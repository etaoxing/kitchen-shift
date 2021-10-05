import cv2
import itertools
import numpy as np
import os

import kitchen_shift


rs = (480, 480)

env = kitchen_shift.Kitchen_v1(
    camera_id=6,
    render_size=rs,
    ctrl_mode='abspos',
)

os.makedirs('assets/init_perturb', exist_ok=True)


init_random_steps_set = list(range(0, 2 * env.frame_skip + 1, 10))
print('init_random_steps_set: ', init_random_steps_set)
for i in init_random_steps_set:
    env.set_init_noise_params([i], 0, 0, 'generator')
    state = env.reset()
    img = env.render(mode='rgb_array', height=rs[0], width=rs[1])
    cv2.imwrite(f'assets/init_perturb/init_random_steps_set-{i}.png', img[:, :, ::-1])


init_perturb_robot_ratio = 0.04
init_qpos1 = env.init_qpos[:].copy()
init_qpos2 = env.init_qpos[:].copy()

init_perturb_object_ratio = 0.04
init_qpos3 = env.init_qpos[:].copy()
init_qpos4 = env.init_qpos[:].copy()

for i in range(env.N_DOF_ROBOT):
    init_qpos1[i] += init_perturb_robot_ratio * env.robot.robot_pos_bound[: env.N_DOF_ROBOT, 0][i]
    env.set_init_qpos(init_qpos1)
    env.set_init_noise_params(None, 0, 0, 'generator')

    state = env.reset()
    img = env.render(mode='rgb_array', height=rs[0], width=rs[1])
    cv2.imwrite(f'assets/init_perturb/init_perturb_robot_ratio-low-{i}.png', img[:, :, ::-1])

for i in range(env.N_DOF_ROBOT):
    init_qpos2[i] += init_perturb_robot_ratio * env.robot.robot_pos_bound[: env.N_DOF_ROBOT, 1][i]
    env.set_init_qpos(init_qpos2)
    env.set_init_noise_params(None, 0, 0, 'generator')

    state = env.reset()
    img = env.render(mode='rgb_array', height=rs[0], width=rs[1])
    cv2.imwrite(f'assets/init_perturb/init_perturb_robot_ratio-high-{i}.png', img[:, :, ::-1])


for i in range(env.N_DOF_OBJECT):
    init_qpos3[i + env.N_DOF_ROBOT] += init_perturb_object_ratio
    env.set_init_qpos(init_qpos3)
    env.set_init_noise_params(None, 0, 0, 'generator')

    state = env.reset()
    action = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array', height=rs[0], width=rs[1])
    cv2.imwrite(f'assets/init_perturb/init_perturb_object_ratio-low-{i}.png', img[:, :, ::-1])

for i in range(env.N_DOF_OBJECT):
    init_qpos4[i + env.N_DOF_ROBOT] -= init_perturb_object_ratio
    env.set_init_qpos(init_qpos4)
    env.set_init_noise_params(None, 0, 0, 'generator')

    state = env.reset()
    action = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array', height=rs[0], width=rs[1])
    cv2.imwrite(f'assets/init_perturb/init_perturb_object_ratio-high-{i}.png', img[:, :, ::-1])
