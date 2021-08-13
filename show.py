import numpy as np
import os
import itertools

import kitchen_env
import gym
from kitchen_env import MujocoWorldgenKitchenEnvWrapper


domain_params = dict(
    train=[[]],
    change_object=(
        [[('change_microwave', i)] for i in range(1, 4)]
        + [[('change_kettle', i)] for i in range(1, 8)]
    ),
    change_object_layout=(
        [[('change_objects_layout', 'microwave', i)] for i in ['closer', 'closer_angled']]
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
    ),
    change_camera=(
        [
            [('change_camera', 2)],
            [('change_camera', 7)],
        ]
    ),
    change_lighting=(
        [[('change_lighting', i)] for i in ['cast_left', 'cast_right', 'brighter', 'darker']]
    ),
    change_texture=(
        [
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
        + [
            [('change_counter_texture', i)]
            for i in ['white_marble_tile2', 'tile1', 'wood1', 'wood2']
        ]
    ),
    change_noise=([[('change_noise_ratio', i)] for i in [0.0, 1.0, 10.0]]),
    change_one_object_done=(
        [
            [('change_object_done', i)]
            for i in ['microwave', 'kettle', 'bottomknob', 'topknob', 'switch', 'slide', 'hinge']
        ]
    ),
)

# domain_params['change_two_object_done'] = [
#     [('change_object_done', i[0]), ('change_object_done', i[1])]
#     for i in itertools.permutations(
#         ['microwave', 'kettle', 'bottomknob', 'topknob', 'switch', 'slide', 'hinge'], 2
#     )
# ]


rs = (480, 480)

env = gym.make(
    'kitchen-v1',
    camera_id=6,
    render_size=rs,
)
env = MujocoWorldgenKitchenEnvWrapper(env)

os.makedirs('worldgen', exist_ok=True)


i = 0
for k, v in domain_params.items():
    for j, ps in enumerate(v):

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

        Image.fromarray(im).save(f'worldgen/{i}-{k}-{j}.png')

        i += 1
