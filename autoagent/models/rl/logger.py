import pprint
import os

from torch.utils import tensorboard
from tabulate import tabulate


class Logger:
    def __init__(
        self,
        out_dir,
        hyper_fname='hyperparams.txt',
        stat_fname='statistics.csv',
        init_tb_writer=True
    ):

        self.stat_fname = stat_fname
        self.hyper_fname = hyper_fname

        self.out_dir = out_dir
        self.first_write = True

        os.makedirs(out_dir, exist_ok=True)
        self.stat_f = open(os.path.join(out_dir, self.stat_fname),
                           encoding='utf8', mode='a')

        if init_tb_writer:
            self.writer = tensorboard.SummaryWriter(log_dir=out_dir)
        else:
            self.writer = None

    def log_hyperparams(self, hyper_params_dict):
        with open(os.path.join(self.out_dir, self.hyper_fname),
                  encoding='utf8', mode='w') as f:
            f.write(pprint.pformat(hyper_params_dict, indent=4))

    def log(self, stats_dict, step):
        if self.first_write:
            # Write csv header to stat file
            self.stat_f.write(",".join(key for key in stats_dict.keys()) + "\n")
            self.first_write = False

        table = []

        stat_values_str = []
        for stat_key, stat_value in stats_dict.items():
            table.append((stat_key, stat_value))
            self.writer.add_scalar(stat_key, stat_value, global_step=step)
            stat_values_str.append(str(stat_value))

        self.stat_f.write(",".join(stat_values_str))
        self.stat_f.write("\n")
        self.stat_f.flush()

        grid = tabulate(table, headers="firstrow", tablefmt="psql",
                              numalign='right')
        print(grid)

    def log_image(self, name, figure, step):
        self.writer.add_figure(name, figure, global_step=step)

    def close(self):
        self.stat_f.close()
        self.writer.close()