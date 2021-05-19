import numpy as np


CAMERAS = {
    0: dict(distance=4.5, azimuth=-66, elevation=-65),
    1: dict(
        distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
    ),  # original, as in https://relay-policy-learning.github.io/
    2: dict(
        distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-50
    ),  # angled up to get a more top-down view
    3: dict(
        distance=2.65, lookat=[0, 0, 2.0], azimuth=90, elevation=-60
    ),  # similar to appendix D of https://arxiv.org/pdf/1910.11956.pdf
    4: dict(
        distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-60
    ),  # 3-6 are first person views at different angles and distances
    5: dict(
        distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-45
    ),  # problem w/ POV is that the knobs can be hidden by the hinge drawer and arm
    7: dict(
        distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-50
    ),  # move back so less of cabinets
    8: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-35),
    9: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-10),
}

# from https://github.com/rail-berkeley/d4rl/blob/9b1eeab56d2adc5afa0541b6dd95399de9635390/d4rl/kitchen/kitchen_envs.py#L9
OBS_ELEMENT_INDICES = {
    'bottomknob': np.array([11, 12]),  # bottom burner
    'topknob': np.array([15, 16]),  # top burner
    'switch': np.array([17, 18]),  # light switch
    'slide': np.array([19]),  # slide cabinet
    'hinge': np.array([20, 21]),  # hinge cabinet
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25]),  # kettle, position on stove only
    # 'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
}

OBS_ELEMENT_GOALS = {
    'bottomknob': np.array([-0.88, -0.01]),
    'topknob': np.array([-0.92, -0.01]),
    'switch': np.array([-0.69, -0.05]),
    'slide': np.array([0.37]),
    'hinge': np.array([0.0, 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62]),
    # 'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
}
