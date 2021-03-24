from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv


class GymEnv:
    def __init__(self, env_callable, wrapper_callables, n_env,
                 use_multiprocessing=False):
        self.env_callable = env_callable
        self.wrapper_callables = wrapper_callables
        self.n_env = n_env
        self.use_multiprocessing = use_multiprocessing
        self.env, self.eval_env, self.vec_env = None, None, None

        self.init_envs()

    def init_envs(self):
        self.env = self.create_env()
        self.eval_env = self.create_env()
        self.vec_env = self.create_env(vectorized=True)

    def seed(self, seed):
        self.env.seed(seed)
        self.eval_env.seed(seed)
        self.vec_env.seed(seed)

    def create_env(self, vectorized=False):
        def env_fn():
            env = self.env_callable()
            for wrapper in self.wrapper_callables:
                env = wrapper(env)
            return env

        if not vectorized:
            return env_fn()

        # Vectorized env
        envs = [env_fn for _ in range(self.n_env)]
        return AsyncVectorEnv(envs) if self.use_multiprocessing else SyncVectorEnv(envs)
