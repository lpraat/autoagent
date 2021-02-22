import torch
import numpy as np
import datetime
import os
import random
import time
import pprint
import tqdm
import matplotlib.pyplot as plt

from torch.utils import tensorboard

from autoagent.models.vision.yolo.config.parse_config import parse_params
from autoagent.datasets.pascal_voc import PascalVoc
from autoagent.models.vision.yolo.data_wrapper import YoloDataset
from autoagent.models.vision.yolo.model import Yolo
from autoagent.models.vision.yolo.eval import eval
from autoagent.utils.general import log, fancy_float
from autoagent.utils.torch import get_num_params, warmup_params
from autoagent.utils.optim import EMA
from autoagent.data.sampler import MultiScaleBatchSampler


def train(img_dim, multi_scale, params_file, dset_train, dset_val,
          batch_size, aggregate, seed, ckpt_file, fine_tune, num_workers):
    # Experiment folder
    exp_dir = (
        f"yolo-{datetime.datetime.now().strftime('%d_%m_%y_%H_%M_%S')}"
    )
    exp_dir = os.path.join('./exp', exp_dir)
    os.makedirs(exp_dir)

    # Log file
    log_file = os.path.join(exp_dir, 'log.txt')
    log_files = [log_file]

    # Parse params_file
    yaml_params, params = parse_params(params_file)
    num_epochs = params['num_epochs']

    # Tensorboard
    writer = tensorboard.SummaryWriter(log_dir=exp_dir)

    yolo = Yolo(params)
    yolo.cuda()

    # Optimizer
    # Apply weight decay only to weights
    weights = []
    biases = []
    bn_weights = []
    for k, v in yolo.named_parameters():
        if '.bias' in k:
            biases.append(v)
        elif '.bn' not in k:
            weights.append(v)
        else:
            bn_weights.append(v)

    if params['optim'] == 'adam':
        optim = torch.optim.Adam(bn_weights, lr=params['init_lr'])
    else:
        optim = torch.optim.SGD(bn_weights, lr=params['init_lr'], momentum=params['momentum'], nesterov=True)

    optim.add_param_group({'params': weights, 'weight_decay': params['weight_decay']})
    optim.add_param_group({'params': biases})

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=num_epochs-params['warmup'], eta_min=params['final_lr']
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Data loaders
    dset_train = YoloDataset(dset_train, params, augment=True)
    dset_val = YoloDataset(dset_val, params, augment=False)
    pin_memory = True

    train_scales = list(range(320, 640, 32)) if multi_scale else [img_dim]  # from 320 to 608
    dloader_train = torch.utils.data.DataLoader(
        dset_train,
        batch_sampler=MultiScaleBatchSampler(
            torch.utils.data.sampler.RandomSampler(dset_train),
            batch_size=batch_size, drop_last=True, scales=train_scales,
            multiscale_every=10*aggregate
        ),
        collate_fn=dset_train.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=dset_train.get_worker_init_fn()
    )

    dloader_val = torch.utils.data.DataLoader(
        dset_val,
        batch_sampler=MultiScaleBatchSampler(
            torch.utils.data.sampler.RandomSampler(dset_val),
            batch_size=batch_size, drop_last=False, scales=[img_dim],
            multiscale_every=aggregate
        ),
        collate_fn=dset_val.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=dset_val.get_worker_init_fn()
    )

    # Log training info
    s = f"Seed: {seed}\n"
    log(s, log_files)
    s = f"{yolo.model.__str__()}"
    log(s, log_files)
    s = pprint.pformat(yaml_params, indent=4)
    log(s + "\n", log_files)
    gpu_name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory
    s = f"GPU {gpu_name} with {mem*10**-9 :.2f} GB\n"
    log(s, log_files)
    s = (f"Num epochs: {num_epochs}\n"
         f"Batch size: {batch_size*aggregate}, Sub size: {batch_size}, Aggregate: {aggregate}\n"
         f"Train size: {len(dset_train)}\n"
         f"Train iter per epoch: {len(dset_train)//(batch_size*aggregate)}\n"
         f"Val size: {len(dset_val)}\n"
         f"Val iter per epoch: {len(dset_val)//(batch_size*aggregate)}\n")
    log(s, log_files)
    s = f"Model: {params['version']} with {get_num_params(yolo.model)} params\n"
    log(s, log_files)

    epoch = 0
    best_score = -1

    # Eventually resume training
    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file)
        if fine_tune:
            yolo.load_state_dict(ckpt, fine_tune=True)
        else:
            yolo.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
            scheduler.load_state_dict(ckpt['scheduler'])
            epoch = ckpt['epoch']
            ap05 = ckpt['AP@0.5']
            ap095 = ckpt['AP@0.5:0.95']
            s = (f"Resuming training from {ckpt_file}\n"
                f"Last epoch: {epoch}, AP@0.5: {ap05:.4f}, AP@0.5:0.95: {ap095:.4f}\n")
            log(s, log_files)
            epoch += 1
            best_score = ap05 + ap095

    # Initial params
    iter_per_epoch = len(dset_train)//(batch_size*aggregate)

    if ckpt_file is None:
        # We are not resuming training
        warmup_params(
            curr_iter=epoch*iter_per_epoch, optim=optim, keys=['lr', 'momentum'],
            xp=[0, iter_per_epoch*params['warmup']],
            fps=[[0, params['init_lr']],[params['warmup_momentum'], params['momentum']]])

    # Exponential moving average
    ema_yolo = Yolo(params)
    ema_yolo.ema = EMA(yolo.model, updates=epoch*iter_per_epoch,
                       decay=params['ema_decay'], exp_d=params['ema_exp_d'],
                       mode=params['ema_mode'])

    # Train loop
    while epoch < num_epochs:
        # Track execution time
        t0 = time.time()

        # Train
        yolo.train()
        train_single_losses = 0  # running mean
        single_losses = 0  # tmp aggregate

        # Progress bar
        titles = ['Epoch', 'TrainIter', 'BatchNum', 'Loss', 'LocLoss', 'DetLoss', 'ClsLoss']
        s_titles = "".join(f"{t:<15}" for t in titles)
        print(f"\n{s_titles}")
        pbar = tqdm.tqdm(total=iter_per_epoch, dynamic_ncols=True)

        for train_iter, (x, y, _) in enumerate(dloader_train):
            x = x.cuda(non_blocking=pin_memory)
            y = [[el.cuda(non_blocking=pin_memory) for el in t] for t in y]

            with torch.cuda.amp.autocast():
                preds = yolo(x)
                loss, _single_losses = yolo.get_loss(preds, y)

                if yolo.params['reduction'] == 'sum':
                    # Accumulate scaled loss
                    loss *= 1 / aggregate

            scaler.scale(loss).backward()

            loss, _single_losses = loss.detach(), _single_losses.detach()
            single_losses += _single_losses / aggregate

            if (train_iter+1) % aggregate == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

                ema_yolo.ema.update(yolo.model)

                iter_num = (train_iter+1)//aggregate

                train_single_losses = (train_single_losses * (iter_num-1) + single_losses) / iter_num

                train_iter = iter_num + epoch*(len(dset_train)//(batch_size*aggregate))
                descs = [
                    f"{epoch+1}/{num_epochs}", f"{train_iter}/{iter_per_epoch*num_epochs}",
                    f"{iter_num}/{iter_per_epoch}", f"{fancy_float(torch.sum(train_single_losses).item())}",
                    *(f"{fancy_float(train_single_losses[i].item())}" for i in range(3))
                ]
                s_descs = "".join(f"{d:<15}" for d in descs)
                pbar.set_description(s_descs)

                single_losses = 0

                # Warmup
                if epoch < params['warmup']:
                    warmup_params(
                        curr_iter=train_iter, optim=optim, keys=['lr', 'momentum'],
                        xp=[0, iter_per_epoch*params['warmup']],
                        fps=[[0, params['init_lr']],[params['warmup_momentum'], params['momentum']]])

                # Update progress bar
                pbar.update()

        pbar.close()

        # Lr annealing
        scheduler.step()

        # Speedup eval during warmup
        conf_thresh = 10 if epoch < params['warmup'] else params['confidence_thresh']

        ema_yolo.model = ema_yolo.ema.ema
        eval_statistics, eval_pr_curves = eval(
            ema_yolo, dloader_val, batch_size, aggregate, epoch,
            confidence_thresh=conf_thresh,
            nms_thresh=params['nms_thresh'],
            name='Val'
        )

        eval_statistics = {f'val/{k}':v for k,v in eval_statistics.items()}
        score = eval_statistics['val/AP@0.5'] + eval_statistics['val/AP@0.5:0.95']
        exec_time = time.time() - t0

        statistics = {
            'train/Loss': torch.sum(train_single_losses),
            'train/Loc_loss': train_single_losses[0],
            'train/Det_loss': train_single_losses[1],
            'train/Cls_loss': train_single_losses[2],
            'train/lr': optim.param_groups[0]['lr'],
            'train/momentum': optim.param_groups[0]['momentum'],
            'time': exec_time
        }
        statistics.update(eval_statistics)

        # Write to tensorboard
        for k, v in statistics.items():
            writer.add_scalar(k, v, global_step=epoch)
        for cls_name, fig in eval_pr_curves:
            writer.add_figure(f'val/{cls_name}', fig, global_step=epoch)
            plt.close(fig)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': ema_yolo.state_dict(),
            'optim': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'AP@0.5': eval_statistics['val/AP@0.5'],
            'AP@0.5:0.95': eval_statistics['val/AP@0.5:0.95']
        }

        last_ckpt_path = os.path.join(exp_dir, "last_ckpt.pt")
        best_ckpt_path = os.path.join(exp_dir, "best_ckpt.pt")

        # Last checkpoint
        if os.path.isfile(last_ckpt_path):
            os.remove(last_ckpt_path)
        torch.save(checkpoint, last_ckpt_path)

        # Best checkpoint
        if score > best_score:
            if os.path.isfile(best_ckpt_path):
                os.remove(best_ckpt_path)
            torch.save(checkpoint, best_ckpt_path)
            best_score = score

        epoch += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("YoloTrain")

    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint file to resume training')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune the provided ckpt')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--params', type=str, required=True,
                        help='Path to params config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='Custom data folder')
    parser.add_argument('--img_dim', type=int, required=True,
                        help='Img dimension for training (no multi-scale) and eval')
    parser.add_argument('--multi_scale', action='store_true',
                        help='Use multi scale training')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size')
    parser.add_argument('--aggregate', type=int, required=True,
                        help='Number of batches to accumulate before an update')
    args = parser.parse_args()

    # Seed everything
    if args.seed is None:
        seed = np.random.randint(2**16-1)
    else:
        seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Dataset
    if args.data == 'voc':
        dset_factory = PascalVoc
    else:
        raise NotImplementedError()

    if args.data_folder is not None:
        dset_train = dset_factory(folder=args.data_folder, train=True)
        dset_val = dset_factory(folder=args.data_folder, train=False)
    else:
        dset_train = dset_factory(train=True)
        dset_val = dset_factory(train=False)

    train(
        img_dim=args.img_dim,
        multi_scale=args.multi_scale,
        params_file=args.params,
        dset_train=dset_train,
        dset_val=dset_val,
        batch_size=args.batch_size,
        aggregate=args.aggregate,
        seed=seed,
        ckpt_file=args.ckpt,
        fine_tune=args.fine_tune,
        num_workers=args.num_workers
    )