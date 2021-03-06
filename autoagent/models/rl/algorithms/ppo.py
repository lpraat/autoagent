import torch
import gym
import numpy as np
import os
import datetime
import wandb
import time

from autoagent.models.rl.utils import get_env_identifier
from autoagent.models.rl.trajectory import collect_trajectories_with_gae
from autoagent.models.rl.logger import Logger
from autoagent.models.rl.env import GymEnv


class PPO:
    """
    Proximal Policy Optimization
    https://arxiv.org/pdf/1707.06347.pdf
    """
    def __init__(self, epochs, lambda_env, lambda_policy, lambda_vfunc, n_samples,
                 n_env, env_wrappers=[], iters=5, lr=3e-4, ent_bonus=0.01, batch_size=None,
                 clip_eps=0.2, gamma=0.995, lambd=0.98, use_multiprocessing=False, device='cpu',
                 out_dir_name='ppo', seed=None, wandb_proj='RL_Benchmarks'):
        """

        Parameters
        ----------
        epochs : int
            Number of training epochs
        lambda_env : Callable
            Callable to create the non-vectorized version of the environment
        lambda_policy : Callable
            Callable to create the policy
        lambda_vfunc : Callable
            Callable to create the value function approximator
        n_samples : int
            Number of samples to collect per environment
            (Total number of samples collected per epoch = n_samples * n_env)
        n_env : int
            Number of parallel (vectorized) environments
        env_wrappers : list[Callable]
            List of environment wrappers provided as callables, by default []
        iters : int
            Number of iterations on epoch samples per policy/value function update, by default 5
        lr : float
            Learning rate for policy/value function optimization, by default 3e-4
        ent_bonus : float
            Entropy bonus coefficient, by default 0.01
        clip_eps : float
            Epsilon coefficient used in the clipped surrogate objective, by default 0.2
        batch_size : int, optional
            Batch size for epoch iters. If this argument is None, batch size will equal n_samples.
            By default None.
        gamma : float, optional
            Discount factor, by default 0.995
        lambd : float, optional
            GAE lambd, by default 0.98
        use_multiprocessing : bool, optional
            Whether to use asynchronous environments, by default False
        device : str
            Target device where to run tensor computations, by default 'cpu'
        out_dir_name : str, optional
            Statistics output folder, by default 'ppo'
        seed : int, optional
            Random seed, by default None
        wandb_proj : str, optional
            Project name to log to, on Weights & Biases, by default rl_bench.
            Set this to None to avoid logging on W&B.
        """
        self.epochs = epochs
        self.n_env = n_env
        self.n_samples = n_samples
        self.iters = iters
        self.lr = lr
        self.ent_bonus = ent_bonus
        self.clip_eps = clip_eps
        batch_size = n_samples if batch_size is None else batch_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambd = lambd
        self.use_multiprocessing = use_multiprocessing
        self.device = device

        self.gym_env = GymEnv(lambda_env, env_wrappers, n_env, use_multiprocessing)
        self.env = self.gym_env.env
        self.eval_env = self.gym_env.eval_env
        self.ob_shape = np.prod(self.env.observation_space.shape)
        if type(self.env.action_space) is gym.spaces.Discrete:
            self.ac_shape = 1
        else:
            self.ac_shape = self.env.action_space.shape[0]
        env_id = get_env_identifier(self.env)
        # Vectorized env
        self.vec_env = self.gym_env.vec_env

        # Seed everything
        if seed is None:
            seed = np.random.randint(2**16-1)
        np.random.seed(seed)
        self.gym_env.seed(seed)
        torch.manual_seed(seed)

        # Policy
        self.policy = lambda_policy()
        self.policy.to(device)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), self.lr)

        # Value function
        self.vfunc = lambda_vfunc()
        self.vfunc.to(device)
        self.opt_vfunc = torch.optim.Adam(self.vfunc.parameters(), self.lr)

        # Setup the statistics logger
        now_str = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        out_dir_exp_name = f"{now_str}_seed_{seed}"
        self.out_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../runs",
                out_dir_name,
                get_env_identifier(self.env),
                out_dir_exp_name
            )
        )

        self.stat_logger = Logger(self.out_dir)
        hyperparams = {
            'epochs': epochs,
            'n_samples': n_samples,
            'n_env': n_env,
            'gamma': gamma,
            'lambd': lambd,
            'iters': iters,
            'lr': lr,
            'ent_bonus': ent_bonus,
            'clip_eps': clip_eps,
            'batch_size': batch_size,
            'use_multiprocessing': use_multiprocessing,
            'seed': seed,
            'policy': self.policy.__str__(),
            'vfunc': self.vfunc.__str__(),
        }
        self.stat_logger.log_hyperparams(hyperparams)

        # Weights & Biases logging
        self.wandb_proj = wandb_proj
        name = f"{env_id}_ppo_{now_str}"
        if self.wandb_proj is not None:
            wandb.init(
                project=self.wandb_proj,
                name=name,
                config={
                    **hyperparams,
                    'env_id': env_id,
                    'algorithm': 'ppo'
                }
            )

    def eval(self, eval_step=10):
        total_reward = 0
        for _ in range(eval_step):

            episode_r = 0
            s = self.eval_env.reset()
            while True:
                a = self.policy.predict(
                    torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0),
                ).to('cpu').numpy().ravel()
                if type(self.eval_env.action_space) is gym.spaces.Discrete:
                    a = a[0]

                s, r, d, _ = self.eval_env.step(a)
                episode_r += r

                if d:
                    break

            total_reward += episode_r

        return total_reward / eval_step

    def run(self):
        best_average_return = - np.inf
        num_samples = 0
        for epoch in range(self.epochs):

            t0 = time.perf_counter()

            # Collect trajectories
            act_call = lambda s: self.policy.predict(
                torch.tensor(s, dtype=torch.float32, device=self.device)
            ).to('cpu').numpy()
            vfunc_call = lambda s: self.vfunc(
                torch.tensor(s, dtype=torch.float32, device=self.device)
            ).to('cpu')
            res = collect_trajectories_with_gae(
                self.vec_env, act_call, vfunc_call,
                self.gamma, self.lambd, self.n_samples, self.ob_shape, self.ac_shape
            )

            # Reshpe and to tensor for downstream computations
            res = [torch.from_numpy(x.reshape(self.n_env * self.n_samples, -1)) for x in res]
            states, actions, _, targets, advantages = res
            advantages = advantages.view(-1)

            num_samples += states.shape[0]

            if self.env.action_space is gym.spaces.Discrete:
                actions = actions.long()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / advantages.std()

            # Optimization
            pin_memory = self.device == 'cuda'
            dataset = torch.utils.data.TensorDataset(states, actions)
            dloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=states.shape[0] if self.device == 'cpu' else 64,
                shuffle=False,
                drop_last=False,
                pin_memory=pin_memory
            )

            old_log_prob = torch.cat([
              self.policy.log_p(mb_states.to(self.device, non_blocking=pin_memory), mb_actions.to(self.device, non_blocking=pin_memory))[0].to('cpu')
              for mb_states, mb_actions in dloader
            ]).detach()

            dataset = torch.utils.data.TensorDataset(states, actions, old_log_prob, advantages, targets)
            dloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=pin_memory
            )

            policy_losses = []
            vfunc_losses = []
            approx_kls = []
            entropies = []
            for _ in range(self.iters):
                for mb_states, mb_actions, mb_old_log_prob, mb_advantages, mb_targets in dloader:
                    self.opt_vfunc.zero_grad()
                    self.opt_policy.zero_grad()

                    mb_states = mb_states.to(self.device, non_blocking=pin_memory)
                    mb_actions = mb_actions.to(self.device, non_blocking=pin_memory)
                    mb_advantages = mb_advantages.to(self.device, non_blocking=pin_memory)
                    mb_targets = mb_targets.to(self.device, non_blocking=pin_memory)
                    mb_old_log_prob = mb_old_log_prob.to(self.device, non_blocking=pin_memory)

                    # Policy
                    new_log_prob, dist = self.policy.log_p(mb_states, mb_actions)
                    # ratio = target / fixed
                    ratio = torch.exp(new_log_prob - mb_old_log_prob)
                    entropy = dist.entropy().mean()
                    policy_loss = - (
                        torch.mean(torch.min(
                            ratio*mb_advantages,
                            torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps)*mb_advantages
                        )) + self.ent_bonus * entropy
                    )
                    policy_loss.backward()

                    # Value function
                    vfunc_loss = (
                        torch.mean((self.vfunc(mb_states) - mb_targets)**2)
                    )
                    vfunc_loss.backward()

                    self.opt_vfunc.step()
                    self.opt_policy.step()

                    entropies.append(entropy.to('cpu').detach().numpy())
                    policy_losses.append(policy_loss.to('cpu').detach().numpy())
                    vfunc_losses.append(vfunc_loss.to('cpu').detach().numpy())
                    approx_kls.append((mb_old_log_prob - new_log_prob).mean().to('cpu').detach().numpy())

            # Eval current policy
            average_return = self.eval(eval_step=10)

            execution_time = time.perf_counter() - t0

            # Save policy
            if average_return > best_average_return:
                best_average_return = average_return
                torch.save(self.policy.state_dict(), os.path.join(self.out_dir, 'best_policy.pt'))

            torch.save(self.policy.state_dict(), os.path.join(self.out_dir, 'last_policy.pt'))

            # Log statistics
            log_dict = dict(
                Epoch=epoch,
                NumSamples=num_samples,
                ExecutionTime=execution_time,
                AverageReturn=average_return,
                AveragePolicyEntropy=np.mean(entropies),
                AveragePolicyLoss=np.mean(policy_losses),
                AverageVfuncLoss=np.mean(vfunc_losses),
                AveragePolicyKL=np.mean(approx_kls),
                CudaMemory_GB=torch.cuda.memory_allocated() * 1e-9 if self.device == 'cuda' else 0
            )
            self.stat_logger.log(log_dict, step=epoch)
            if self.wandb_proj:
                wandb.log(log_dict)

        self.stat_logger.close()
        wandb.finish()
