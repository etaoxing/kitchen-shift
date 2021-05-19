from .kitchen_v0 import Kitchen_v0

from gym.envs.registration import register

# Relax the robot
register(
    id='kitchen-v0',
    entry_point='kitchen_env.kitchen_v0:Kitchen_v0',
    max_episode_steps=280,
)
