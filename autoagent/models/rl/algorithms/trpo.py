import torch
import gym
import numpy as np
import os
import datetime
import wandb
import time

from autoagent.utils.torch import assign_flat_params
from autoagent.models.rl.utils import get_env_identifier
from autoagent.models.rl.trajectory import collect_trajectories_with_gae
from autoagent.models.rl.logger import Logger


class TRPO:
    """
    Trust Region Policy Optimization
    https://arxiv.org/pdf/1502.05477.pdf
    """
    def __init__(self, epochs, lambda_env, lambda_policy, lambda_vfunc, n_samples,
                 n_env, max_epidose_len, gamma=0.995, lambd=0.98, vfunc_lr=1e-2, vfunc_iters=5,
                 vfunc_batch_size=64, cg_iters=10, cg_damping=0.1, kl_thresh=0.01,
                 use_multiprocessing=False, out_dir_name='trpo', seed=None, wandb_proj='RL_Benchmarks'):
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
        max_epidose_len : int
            Maximum steps per episode
        gamma : float, optional
            Discount factor, by default 0.995
        lambd : float, optional
            GAE lambd, by default 0.98
        vfunc_lr : float, optional
            Learning rate for value function optimization, by default 1e-2
        vfunc_iters : int, optional
            Number of iterations on epoch samples per value function update, by default 5
        vfunc_batch_size : int, optional
            Batch size for value function optimization, by default 64
        cg_iters : int, optional
            Conjugate gradient iterations, by default 10
        cg_damping : float, optional
            Conjugate gradient damping factor, by default 0.1
        kl_thresh : float, optional
            KL threshold, by default 0.01
        use_multiprocessing : bool, optional
            Whether to use asynchronous environments, by default False
        out_dir_name : str, optional
            Statistics output folder, by default 'trpo'
        seed : int, optional
            Random seed, by default None
        wandb_proj : str, optional
            Project name to log to, on Weights & Biases, by default rl_bench.
            Set this to None to avoid logging on W&B.
        """
        self.epochs = epochs
        self.max_episode_len = max_epidose_len
        self.n_env = n_env
        self.n_samples = n_samples
        self.gamma = gamma
        self.lambd = lambd
        self.vfunc_lr = vfunc_lr
        self.vfunc_iters = vfunc_iters
        self.vfunc_batch_size = vfunc_batch_size
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.kl_thresh = kl_thresh
        self.use_multiprocessing = use_multiprocessing

        self.env = lambda_env()
        self.eval_env = lambda_env()
        self.ob_shape = self.env.observation_space.shape[0]
        if type(self.env.action_space) is gym.spaces.Discrete:
            self.ac_shape = 1
        else:
            self.ac_shape = self.env.action_space.shape[0]
        env_id = get_env_identifier(self.env)
        # Vectorized env
        self.vec_env = gym.vector.make(
            env_id, num_envs=self.n_env, asynchronous=self.use_multiprocessing
        )

        # Seed everything
        if seed is None:
            seed = np.random.randint(2**16-1)
        np.random.seed(seed)
        self.env.seed(seed)
        self.vec_env.seed(seed)
        torch.manual_seed(seed)

        # Policy
        self.policy = lambda_policy()

        # Value function
        self.vfunc = lambda_vfunc()
        self.opt_vfunc = torch.optim.Adam(self.vfunc.parameters(), self.vfunc_lr)

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
            'max_episode_len': max_epidose_len,
            'gamma': gamma,
            'lambd': lambd,
            'vfunc_lr': vfunc_lr,
            'vfunc_iters': vfunc_iters,
            'vfunc_batch_size': vfunc_batch_size,
            'cg_iters': cg_iters,
            'cg_damping': cg_damping,
            'kl_thresh': kl_thresh,
            'use_multiprocessing': use_multiprocessing,
            'seed': seed,
            'policy': self.policy.__str__(),
            'vfunc': self.vfunc.__str__(),
        }
        self.stat_logger.log_hyperparams(hyperparams)

        # Weights & Biases logging
        self.wandb_proj = wandb_proj
        name = f"{env_id}_trpo_{now_str}"
        if self.wandb_proj is not None:
            wandb.init(
                project=self.wandb_proj,
                name=name,
                config={
                    **hyperparams,
                    'env_id': env_id,
                    'algorithm': 'trpo'
                }
            )

    def eval(self, eval_step=10):
        total_reward = 0
        for _ in range(eval_step):

            episode_r = 0
            s = self.eval_env.reset()
            for _ in range(self.max_episode_len):
                a = self.policy.predict(
                    torch.tensor(s, dtype=torch.float32).unsqueeze(0),
                ).numpy().ravel()
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
            va = lambda s: self.vfunc(torch.tensor(s, dtype=torch.float32))
            res = collect_trajectories_with_gae(
                self.vec_env, self.policy, va,
                self.gamma, self.lambd, self.n_samples, self.ob_shape, self.ac_shape, self.max_episode_len
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

            # TRPO optimization
            # Fixed log probs
            old_log_prob = self.policy.log_p(
                states, actions
            ).detach()

            def compute_gain():
                """
                Computes the gain of the new policy w.r.t the old one
                """
                new_log_prob = self.policy.log_p(
                    states, actions
                )
                gain = torch.mean(
                    torch.exp(new_log_prob - old_log_prob) * (advantages)
                )
                return gain

            if type(self.env.action_space) is gym.spaces.Discrete:
                *_, dist0 = self.policy(states, get_log_p=False)
                p0 = dist0.probs.detach()
            else:
                *_, dist0 = self.policy(states, get_log_p=False)
                mu0 = dist0.loc.detach()
                log_std0 = torch.log(dist0.scale.detach())

            def compute_kl():
                """
                Computes KL(policy_old||policy_new)
                or according to the following notation KL(0||1)
                """
                if type(self.env.action_space) is gym.spaces.Discrete:
                    *_, dist1 = self.policy(states, get_log_p=False)
                    p1 = dist1.probs
                    return (p0*torch.log(p0/p1)).sum(dim=1).mean()
                else:
                    *_, dist1 = self.policy(states, get_log_p=False)
                    mu1 = dist1.loc
                    log_std1 = torch.log(dist1.scale)

                    var0 = torch.exp(log_std0)**2
                    var1 = torch.exp(log_std1)**2
                    return (
                        (0.5 * ((var0 + (mu1-mu0)**2) / (var1 + 1e-7) - 1)
                        + log_std1 - log_std0)
                    ).sum(dim=1).mean()

            def hessian_vector_product(x):
                """
                Computes the product between the Hessian of the KL
                wrt the policy parameters and the tensor provided as input x
                """
                kl = compute_kl()
                grads = torch.autograd.grad(
                    kl, self.policy.parameters(), create_graph=True
                )
                grads = torch.cat(
                    [torch.reshape(grad, (-1,)) for grad in grads], dim=0
                )
                sum_kl_x = torch.sum(grads * x, dim=0)
                grads_2 = torch.autograd.grad(sum_kl_x, self.policy.parameters())
                grads_2 = torch.cat(
                    [torch.reshape(grad, (-1,)) for grad in grads_2], dim=0
                )
                grads_2 += self.cg_damping * x
                return grads_2

            # The trpo optimization problem (using lagrange multipliers) is
            # g.dot(p-p_old) - lagrange_multiplier * (p-p_old)H^-1(p-p_old).T
            # such that (p-p_old)H^-1(p_p_old).T <= kl_thresh
            # where g is the gradient of the gain and H is the hessian of the
            # KL divergence (w.r.t. the policy parameters p)
            # The solution is (p-o_old) = 1/lagrange_mult * H^-1 g
            # The lagrange_mult does not change the search direction
            # so we can compute (p-p_old)=H^-1g
            # Now we can find the scaling so that the KL contrain is satisfied
            # Which is lagrange_mult=sqrt((p-p_old)H^-1(p-o_old).T / (2*kl_thresh))

            # To find the tensor x=(p-p_old) we need to comopute H^1 g
            # Rather than computing and storing the full H which would be huge for
            # a standard network and rather than inverting it we solve
            # the linear problem Hx=g using the conjugate gradient algorithm
            # to obtain x=(p-p_old)=H^-1g
            # The conjugate gradient needs only a function
            # that is able to compute Hx
            # We can provide this function without storing the full hessian
            # by computing directly Hx
            # Which can be done by computing grad(grad(KL)*x)
            gain = compute_gain()
            g = torch.autograd.grad(gain, self.policy.parameters())
            g = torch.cat([torch.reshape(x, (-1,)) for x in g], dim=0)

            x = TRPO.conj_gradient(hessian_vector_product, g, iters=self.cg_iters)

            # 1/lagrange_multiplier is the maximum step we can take
            # along the gradient direction
            lagrange_mult = torch.sqrt(
                torch.dot(x, hessian_vector_product(x)) / (2*self.kl_thresh)
            )

            # Backtracking to ensure improvement and *exact* KL constraint
            # after the update
            success, params, backtrack_iters = TRPO.backtracking(
                self.policy, compute_gain, compute_kl, self.kl_thresh, x, 1/lagrange_mult
            )

            # Update the value function
            dataset = torch.utils.data.TensorDataset(states, targets)
            dloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.vfunc_batch_size,
                shuffle=True,
                drop_last=True
            )
            for _ in range(self.vfunc_iters):
                self.opt_vfunc.zero_grad()
                for mb_states, mb_targets in dloader:
                    self.opt_vfunc.zero_grad()
                    loss = torch.mean((self.vfunc(mb_states) - mb_targets)**2)
                    loss.backward()
                    self.opt_vfunc.step()

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
                BacktrackSuccess=int(success),
                BacktrackIters=backtrack_iters,
            )
            self.stat_logger.log(log_dict, step=epoch)
            if self.wandb_proj:
                wandb.log(log_dict)

        self.stat_logger.close()
        wandb.finish()


    @staticmethod
    def conj_gradient(Ax, b, iters):
        """
        Conjugate gradient algorithm that can be used
        to solve the system of linear equations Ax=b.
        https://en.wikipedia.org/wiki/Conjugate_gradient_method

        Parameters
        ----------
        Ax : Callable
            Callable that takes as input the current guess x
            and returns the tensor product Ax
        b : torch.Tensor
            Constant terms
        iters : int
            Number of iterations

        Returns
        -------
        torch.Tensor
            System's solution x
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()

        i = 0

        while True:
            Ap = Ax(p)

            rr = torch.dot(r, r)
            alpha = rr / torch.dot(p, Ap)
            x += alpha * p

            i += 1
            if i == iters:
                break

            r_new = r - alpha * Ap
            beta = torch.dot(r_new, r_new) / rr
            p = r_new + beta * p
            r = r_new

        return x


    @staticmethod
    def backtracking(model, compute_objective, compute_constraint, constrain_thresh,
                     search_dir, step, max_iters=10, just_check_constraint=False):
        """
        Simple backtracking line search algorithm for constrained optimization.
        Side effects: the function automatically updates the model parameters.
        Parameters
        ----------
        model : torch.nn.Module
            Model whose parameter vector is the one being optimized
        compute_objective : Callable
            Callable that computes the objective function to maximize
        compute_constraint : Callable
            Callable that computes the value of the function to constrain
        constrain_thresh : float
            Constrain threshold
        search_dir : torch.Tensor
            Optimization search direction (e.g. gradient of lagrangian function)
        step : float
            (Maximum) Proposed step
        max_iters : int, optional
            The maximum number of backtrack iterations, by default 10
        Returns
        -------
        tuple[bool, torch.Tensor, int]
            A triplet indicating whether the line search has been successful,
            the updated parameters
            and the number of backtrack iterations used
        """
        old_objective = compute_objective()
        old_params = torch.cat(
            [torch.reshape(w, (-1,)) for w in model.parameters()], dim=0
        )

        for i in range(max_iters):
            alpha = 0.5**i
            new_params = old_params + alpha * step * search_dir

            # Assign parameters after applying proposed step
            assign_flat_params(model, new_params)
            new_objective = compute_objective()

            new_constrain = compute_constraint()

            improvement = new_objective - old_objective

            # If we improve the constrained objective
            valid_update = (just_check_constraint or
                            (torch.isfinite(new_objective) and improvement > 0))

            # If the constraint is satisfied
            valid_update = (valid_update
                            and torch.isfinite(new_constrain)
                            and new_constrain < constrain_thresh)

            if valid_update:
                return True, new_params, i

        # We have not found a suitable step in max_iters
        assign_flat_params(model, old_params)
        return False, old_params, i
