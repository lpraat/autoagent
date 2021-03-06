import torch
import torch.nn as nn
import numpy as np
import os
import datetime
import time
import wandb

from autoagent.models.rl.trajectory import ReplayMemory
from autoagent.models.rl.utils import get_env_identifier
from autoagent.models.rl.logger import Logger
from autoagent.models.rl.env import GymEnv

class SAC:
    """
    Soft Actor Critic
    https://arxiv.org/pdf/1812.05905.pdf
    """
    def __init__(self, lambda_env, lambda_qfunc, lambda_policy,
                 epochs, steps_per_epoch, uniform_steps, env_wrappers=[], init_alpha=0.2,
                 batch_size=256, lr=3e-4, tau=0.005, buff_size=1e6, discount=0.99,
                 device='cpu', seed=None, out_dir_name='sac', wandb_proj='RL_Benchmarks'):
        """

        Parameters
        ----------
        lambda_env : Callable
            Callable to create the environment
        lambda_qfunc : Callable
            Callable to create the q-function approximator
        lambda_policy : Callable
            Callable to create the policy
        epochs : int
            Number of training epochs
        steps_per_epoch : int
            Number of training steps per epoch
        uniform_steps : int
            Number of steps where the agent acts randomly
        env_wrappers : list[Callable]
            List of environment wrappers provided as callables, by default []
        init_alpha : float, optional
            Initial alpha (action entropy weight), by default 0.2
        batch_size : int, optional
            Batch size per update, by default 256
        lr : float, optional
            Learning rate for policy, q-function, alpha optimization, by default 3e-4
        tau : float, optional
            Smooth update factor for critic optimization, by default 0.005
        buff_size : , optional
            Size of the replay memory, by default 1e6
        discount : float, optional
            Discount factor, by default 0.99
        device : str
            Target device where to run tensor computations, by default 'cpu'
        seed : int, optional
            Random seed, by default None
        out_dir_name : str, optional
            Statistics output dir name, by default 'sac'
        wandb_proj : str, optional
            Project name to log to, on Weights & Biases, by default RL_Benchmarks.
            Set this to None to avoid logging on W&B.
        """
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.uniform_steps = uniform_steps
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.buff_size = buff_size
        self.discount = discount
        self.seed = seed
        self.device = device

        self.gym_env = GymEnv(lambda_env, env_wrappers, 1, False)
        self.env = self.gym_env.env
        self.eval_env = self.gym_env.eval_env

        # Seed everything
        if seed is None:
            seed = np.random.randint(2**16-1)
        np.random.seed(seed)
        self.gym_env.seed(seed)
        torch.manual_seed(seed)

        # Init replay buffer
        self.buff = ReplayMemory(
            size=buff_size,
            num_features=np.prod(self.env.observation_space.shape),
            action_dim=self.env.action_space.shape[0]
        )

        # Actor
        self.policy = lambda_policy()
        self.policy.to(self.device)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr)

        # Critic
        self.q1 = lambda_qfunc()
        self.q1.to(self.device)
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr)
        self.q2 = lambda_qfunc()
        self.q2.to(self.device)
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr)

        # Target critic
        self.target_q1 = lambda_qfunc()
        self.target_q1.to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2 = lambda_qfunc()
        self.target_q2.to(self.device)
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Alpha - Action entropy weight
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=self.device)
        self.target_entropy = -self.env.action_space.shape[0]
        self.opt_log_alpha = torch.optim.Adam([self.log_alpha], lr)

        # Setup the statistics logger
        now_str = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        out_dir_exp_name = f"{now_str}_seed_{seed}"
        env_id = get_env_identifier(self.env)
        self.out_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../runs",
                out_dir_name,
                env_id,
                out_dir_exp_name
            )
        )
        self.stat_logger = Logger(self.out_dir)
        hyperparams = {
            'epochs': epochs,
            'steps_per_epoch': steps_per_epoch,
            'uniform_steps': uniform_steps,
            'init_alpha': init_alpha,
            'batch_size': batch_size,
            'lr': lr,
            'tau': tau,
            'buff_size': buff_size,
            'discount': discount,
            'seed': seed,
            'policy': self.policy.__str__(),
            'qfunc': self.q1.__str__(),
        }
        self.stat_logger.log_hyperparams(hyperparams)

        # Weights & Biases logging
        self.wandb_proj = wandb_proj
        name = f"{env_id}_sac_{now_str}"
        if self.wandb_proj is not None:
            wandb.init(
                project=self.wandb_proj,
                name=name,
                config={
                    **hyperparams,
                    'env_id': env_id,
                    'algorithm': 'sac'
                }
            )

    def eval(self, eval_step=10, deterministic=True):
        """
        Non stochastic eval, by default
        """
        total_reward = 0
        for _ in range(eval_step):

            episode_r = 0
            s = self.eval_env.reset()
            while True:
                a = self.policy.predict(
                    torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0),
                    deterministic=deterministic
                ).to('cpu').numpy().ravel()
                s, r, d, _ = self.eval_env.step(a)
                episode_r += r

                if d:
                    break

            total_reward += episode_r

        return total_reward / eval_step

    def update_critic(self, samples):
        s, a, r, ns, dones = samples

        # Compute target for q1 and q2
        with torch.no_grad():
            na, log_p, _ = self.policy(ns, get_log_p=True)
            q1_target_value = self.target_q1(torch.cat([ns, na], dim=1))
            q2_target_value = self.target_q2(torch.cat([ns, na], dim=1))
            target = r + self.discount * (1 - dones) * (
                torch.min(q1_target_value, q2_target_value)
                - torch.exp(self.log_alpha) * log_p
            )

        # Update q1
        self.opt_q1.zero_grad()
        q1_loss = ((self.q1(torch.cat([s, a], dim=1)) - target) ** 2).mean()
        q1_loss.backward()
        self.opt_q1.step()

        # Update q2
        self.opt_q2.zero_grad()
        q2_loss = ((self.q2(torch.cat([s, a], dim=1)) - target) ** 2).mean()
        q2_loss.backward()
        self.opt_q2.step()

        return q1_loss.to('cpu').detach().numpy(), q2_loss.to('cpu').detach().numpy()

    def update_target_critic(self):
        # Smooth target1 update
        for p_target, p in zip(self.target_q1.parameters(), self.q1.parameters()):
            p_target.data = (1 - self.tau) * p_target.data + self.tau * p.data

        # Smooth target2 update
        for p_target, p in zip(self.target_q2.parameters(), self.q2.parameters()):
            p_target.data = (1 - self.tau) * p_target.data + self.tau * p.data

    def update_actor(self, samples):
        s, *_ = samples
        a, log_p, _ = self.policy(s, get_log_p=True)

        # Policy update
        self.opt_policy.zero_grad()
        sa_cat = torch.cat([s, a], dim=1)
        alpha = torch.exp(self.log_alpha).detach()
        pol_loss = - (torch.min(self.q1(sa_cat), self.q2(sa_cat)) - alpha * log_p).mean()
        pol_loss.backward()
        self.opt_policy.step()

        # Alpha update
        self.opt_log_alpha.zero_grad()
        alpha_loss = (torch.exp(self.log_alpha) * (-log_p - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.opt_log_alpha.step()

        return pol_loss.to('cpu').detach().numpy(), alpha_loss.to('cpu').detach().numpy()

    def update(self):
        batch = self.buff.sample_batch(self.batch_size)
        samples = [torch.from_numpy(el).to(self.device) for el in batch]
        q1_loss, q2_loss = self.update_critic(samples)
        policy_loss, alpha_loss = self.update_actor(samples)
        self.update_target_critic()
        return q1_loss, q2_loss, policy_loss, alpha_loss

    def run(self):
        s = self.env.reset()
        episode_r = 0
        total_r = 0
        episodes = 0
        best_average_return = -np.inf
        t0 = time.perf_counter()
        q1_losses, q2_losses = [], []
        policy_losses, alpha_losses = [], []

        for step in range(self.epochs * self.steps_per_epoch):
            if step > self.uniform_steps:
                a = self.policy.predict(torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0))
                a = a.to('cpu').numpy().ravel()
            else:
                # Random warmup
                a = self.env.action_space.sample()

            ns, r, d, info = self.env.step(a)
            episode_r += r

            # Bootstrap q-value in case of a time limit
            buff_done = d and not info.get('TimeLimit.truncated', False)
            self.buff.add_sample((s, a, r, ns, buff_done))

            if d:
                s = self.env.reset()

                episodes += 1
                total_r += episode_r

                episode_r = 0
            else:
                s = ns

            # Do not update during warmup
            if step > self.uniform_steps:
                q1_loss, q2_loss, policy_loss, alpha_loss = self.update()
                q1_losses.append(q1_loss)
                q2_losses.append(q2_loss)
                policy_losses.append(policy_loss)
                alpha_losses.append(alpha_loss)

            if ((step+1) % self.steps_per_epoch == 0):
                epoch = int(step / self.steps_per_epoch)

                average_return = self.eval(eval_step=10)
                stoch_average_return = total_r / episodes
                execution_time = time.perf_counter() - t0

                if average_return > best_average_return:
                    best_average_return = average_return
                    torch.save(self.policy.state_dict(), os.path.join(self.out_dir, 'best_policy.pt'))

                torch.save(self.policy.state_dict(), os.path.join(self.out_dir, 'last_policy.pt'))

                log_dict = dict(
                    Epoch=epoch+1,
                    TotalSteps=step+1,
                    AverageReturn=average_return,
                    AverageQ1Loss=np.mean(q1_losses),
                    AverageQ2Loss=np.mean(q2_losses),
                    AveragePolicyLoss=np.mean(policy_losses),
                    AverageAlphaLoss=np.mean(alpha_losses),
                    StochasticAverageReturn=stoch_average_return,
                    ExecutionTime=execution_time,
                    Alpha=torch.exp(self.log_alpha).item()
                )

                self.stat_logger.log(log_dict, step=epoch)
                if self.wandb_proj:
                    wandb.log(log_dict)

                episodes = 0
                total_r = 0
                q1_losses, q2_losses = [], []
                policy_losses, alpha_losses = [], []
                t0 = time.perf_counter()

        self.stat_logger.close()
        wandb.finish()
