domain_params = (
    [[]]
    + [[('change_microwave', i)] for i in range(1, 4)]
    + [[('change_kettle', i)] for i in range(1, 8)]
    + [[('change_objects_layout', 'microwave', i)] for i in ['closer', 'closer_angled']]
    + [
        [('change_objects_layout', 'kettle', i)]
        for i in ['top_right', 'bot_right', 'bot_right_angled']
    ]
    + [
        [('change_objects_layout', 'slide', 'right_raised')],
        [
            ('change_objects_layout', 'hinge', 'left_lowered'),
            ('change_objects_layout', 'slide', 'right_lowered'),
            ('change_objects_layout', 'ovenhood', 'right_raised'),
        ],
    ]
    + [[('change_camera', 2)]]
    + [[('change_lighting', i)] for i in ['cast_left', 'cast_right', 'brighter', 'darker']]
    + [
        [('change_hinge_texture', i)]
        for i in ['wood1', 'wood2', 'metal1', 'metal2', 'marble1', 'tile1']
    ]
    + [
        [('change_slide_texture', i)]
        for i in ['wood1', 'wood2', 'metal1', 'metal2', 'marble1', 'tile1']
    ]
    + [
        [('change_floor_texture', i)]
        for i in ['white_marble_tile', 'marble1', 'tile1', 'wood1', 'wood2']
    ]
    + [[('change_counter_texture', i)] for i in ['white_marble_tile2', 'tile1', 'wood1', 'wood2']]
    + [[('change_noise_ratio', i)] for i in [0.0, 1.0]]
)


# domain_params = [
#     [
#         ('change_objects_layout', 'hinge', 'left_corner'),
#         ('change_objects_layout', 'slide', 'right_corner'),
#         ('change_objects_layout', 'ovenhood', 'right_raised'),
#     ],
#     [
#         ('change_objects_layout', 'hinge', 'left_corner'),
#         ('change_objects_layout', 'slide', 'right_corner'),
#         ('change_objects_layout', 'ovenhood', 'right_raised'),
#         ('change_object_done', 'switch'),
#     ],
#     [
#         ('change_object_done', 'switch'),
#     ],
#     [
#         ('change_objects_done', ['topknob', 'switch']),
#         ('change_objects_layout', 'kettle', 'bot_right'),
#     ],
# ]


def domain_param_to_tag(domain_param):
    if len(domain_param) == 0:
        return 'unchanged'

    s = ''
    for p in domain_param:
        s += f"{p[0]}({', '.join([repr(x) for x in p[1:]])}),"
    return s[:-1]


import numpy as np

import os

os.makedirs('worldgen', exist_ok=True)

import kitchen_env
import gym

rs = (480, 480)

env = gym.make(
    'kitchen-v1',
    camera_id=6,
    render_size=rs,
)

from kitchen_env import MujocoWorldgenKitchenEnvWrapper

env = MujocoWorldgenKitchenEnvWrapper(env)

for i, ps in enumerate(domain_params):
    env.reset_domain_changes()

    print(i, ps)

    for p in ps:
        fn = getattr(env, p[0])
        fn(*p[1:])

    env.reset(reload_model_xml=True)

    for _ in range(1):
        state, reward, done, info = env.step(np.zeros(9))

    im = env.render(mode='rgb_array', height=rs[0], width=rs[1])

    from PIL import Image

    Image.fromarray(im).save(f'worldgen/{i}.png')
