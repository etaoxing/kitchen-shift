from gym.envs.registration import register

try:
    from .kitchen_v0 import Kitchen_v0

    register(
        id='kitchen-v0',
        entry_point='kitchen_env.kitchen_v0:Kitchen_v0',
        max_episode_steps=280,
    )
except:
    pass

try:
    from .kitchen_v1 import Kitchen_v1

    register(
        id='kitchen-v1',
        entry_point='kitchen_env.kitchen_v1:Kitchen_v1',
        max_episode_steps=280,
    )
except:
    pass


from .mujoco.worldgen import MujocoWorldgenKitchenEnvWrapper
