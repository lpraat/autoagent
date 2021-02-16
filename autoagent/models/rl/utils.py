def get_env_identifier(env):
    return (env.unwrapped.spec.id
            if env.unwrapped.spec is not None
            else env.__class__.__name__)