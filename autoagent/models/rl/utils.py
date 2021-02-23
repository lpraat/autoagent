def get_env_identifier(env):
    return (env.unwrapped.spec.id
            if hasattr(env, 'unwrapped')
            else env.__class__.__name__)
