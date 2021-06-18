import os, sys
import pickle
import functools
import random
import time as timer
from glob import glob
import multiprocessing
import cv2
import numpy as np
import gym
from tqdm import tqdm

import kitchen_env
from kitchen_env.adept_envs.utils.parse_mjl import parse_mjl_logs, viz_parsed_mjl_logs

NUM_WORKERS = 16
ENVS = []  # populated later

DEMO_DIR = 'kitchen_demos_multitask/'
# WHICH_SET = 'postcorl'
# WHICH_SET = 'friday'
WHICH_SET = None

OUT_DIR = 'workdir/'
OUT_DIR_SINGLEOBJ = 'workdir_singleobj/'

CAMERA_ID = 7
FPS = 30.0
FRAME_SKIP = 40
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
VIDEO_EXT = 'avi'

# RENDER_SIZE = (64, 64)
RENDER_SIZE = (128, 128)
# RENDER_SIZE = (120, 160)
# RENDER_SIZE = (256, 256)
# RENDER_SIZE = (240, 320)
# RENDER_SIZE = (480, 640)
# RENDER_SIZE = (1920, 2560)


create_env_fn = functools.partial(
    gym.make,
    'kitchen-v0',
    ctrl_mode='absvel',  # default ctrl mode of demos
    compensate_gravity=True,
    with_obs_ee=True,
    with_obs_forces=True,
    rot_use_euler=True,
)

# TODO: we spawn a separate env if want to convert demos to a different action space

DEMO_OBJ_ORDER = ['microwave', 'kettle', 'bottomknob', 'topknob', 'switch', 'slide', 'hinge']
from kitchen_env.constants import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS

BONUS_THRESH = 0.1  # lowered threshold (guarantees kettle in place when splitting singleobj)
SINGLEOBJ_TIME_EPS = 15


def save_video(render_buffer, filepath):
    vw = cv2.VideoWriter(filepath, FOURCC, FPS, RENDER_SIZE[::-1])
    for frame in render_buffer:
        frame = frame.copy()
        # frame = cv2.resize(frame, RENDER_SIZE[::-1])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vw.write(frame)
    vw.release()


def render_demo(env, data, use_physics=False, log=True):
    render_skip = max(1, round(1.0 / (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    init_qpos = data['qpos'][0].copy()
    init_qvel = data['qvel'][0].copy()

    if use_physics:
        # initialize
        env.reset()

        # prepare env
        env.sim.data.qpos[:] = init_qpos
        env.sim.data.qvel[:] = init_qvel
        env.sim.forward()

    render_buffer = []
    path = dict(observations=[], actions=[])
    N = data['ctrl'].shape[0]
    i_frame = 0

    while True:
        if use_physics:
            # Reset every time step
            # if i_frame % 1 == 0:
            #     qp = data['qpos'][i_frame].copy()
            #     qv = data['qvel'][i_frame].copy()
            #     env.sim.data.qpos[:] = qp
            #     env.sim.data.qvel[:] = qv
            #     env.sim.forward()

            obs = env._get_obs_dict()
        else:
            # w/o physics, not getting terminal obseration, so will have one less obs than w/ physics
            if i_frame == N:
                break

            env.sim.data.qpos[:] = data['qpos'][i_frame].copy()
            env.sim.data.qvel[:] = data['qvel'][i_frame].copy()
            env.sim.forward()

        if i_frame % render_skip == 0:
            curr_frame = env.render(mode='rgb_array', height=RENDER_SIZE[0], width=RENDER_SIZE[1])
            render_buffer.append(curr_frame)
            if log:
                print(f'{i_frame}', end=', ', flush=True)

        if use_physics:
            path['observations'].append(obs)
            if i_frame == N:  # for adding final observation
                break

            # Construct the action
            robot_qp = obs['robot_qp']
            # ctrl = (data['qpos'][i_frame + 1][:9] - robot_qp) / (env.skip * env.model.opt.timestep)
            ctrl = (data['ctrl'][i_frame] - robot_qp) / (env.skip * env.model.opt.timestep)
            act = (ctrl - env.act_mid) / env.act_amp
            act = np.clip(act, -0.999, 0.999)

            next_obs, reward, done, env_info = env.step(act)

            path['actions'].append(act)

        i_frame += 1

    if log:
        print()
        print("physics = %i, time taken = %f" % (use_physics, timer.time() - t0))

    if use_physics:
        # path['observations'] = np.vstack(path['observations'])
        path['actions'] = np.vstack(path['actions'])
        return render_buffer, path
    else:
        return render_buffer


def get_task_info(objects_task, objects_done=[]):
    sorted_objects_task = list(sorted(objects_task, key=DEMO_OBJ_ORDER.index))
    sorted_objects_done = list(sorted(objects_done, key=DEMO_OBJ_ORDER.index))
    task_id = 't-' + ','.join(sorted_objects_task)
    if len(objects_done) > 0:
        task_id += '_d-' + ','.join(sorted_objects_done)
    return task_id, sorted_objects_task


def process_demo(
    filepath,
    save_data=True,
    view_demo=True,  # view demos (physics ignored)
    playback_demo=True,  # playback demos and get data(physics respected)
    split_singleobj=True,
    log=True,
):
    worker_id = multiprocessing.current_process()._identity
    if worker_id == ():
        worker_id = 0  # then not using pool
    else:
        worker_id = worker_id[0] - 1  # subtracting 1 b/c _identity starts from 1
    env = ENVS[worker_id]

    _, task, fn = filepath.split('/')
    which_set = task.split('_')[0]
    task_objects = task.split('_')[1:]
    task_id, sorted_objects_task = get_task_info(task_objects, objects_done=[])
    task_id = which_set + '_' + task_id

    if log:
        print('worker id', worker_id, filepath, task_id)

    os.makedirs(os.path.join(OUT_DIR, task_id), exist_ok=True)
    f = fn.split('.mjl')[0]
    outpath = os.path.join(OUT_DIR, task_id, f)

    data = parse_mjl_logs(filepath, FRAME_SKIP)

    render_meta = f'-c{CAMERA_ID}h{RENDER_SIZE[0]}w{RENDER_SIZE[1]}.{VIDEO_EXT}'
    if view_demo:
        render_buffer = render_demo(env, data, use_physics=False, log=log)
        save_video(render_buffer, outpath + f'_view{render_meta}')

    if playback_demo:
        try:
            render_buffer, path = render_demo(env, data, use_physics=True, log=log)
        except Exception as e:
            print(f'skipped playback {filepath}, {e}')
            return -1
        data['path'] = path
        save_video(render_buffer, outpath + f'_playback{render_meta}')

    if save_data:
        pickle.dump(data, open(outpath + '.pkl', 'wb'))

    if split_singleobj:
        if not playback_demo:
            raise RuntimeError

        process_demo_split_singleobj(
            which_set,
            sorted_objects_task,
            f,
            data,
            render_buffer,
            render_meta,
            save_data=save_data,
            log=log,
        )


def process_demo_split_singleobj(
    which_set, sorted_objects_task, f, data, render_buffer, render_meta, save_data=True, log=True
):
    prev_t = 0
    for i, o in enumerate(sorted_objects_task):
        task_id_singleobj, _ = get_task_info([o], objects_done=sorted_objects_task[:i])
        task_id_singleobj = which_set + '_' + task_id_singleobj
        os.makedirs(os.path.join(OUT_DIR_SINGLEOBJ, task_id_singleobj), exist_ok=True)

        outpath = os.path.join(OUT_DIR_SINGLEOBJ, task_id_singleobj, f)

        obs_indices = OBS_ELEMENT_INDICES[o]
        obs_goals = OBS_ELEMENT_GOALS[o]

        ag = data['qpos'][:, obs_indices]
        d = np.linalg.norm(ag - obs_goals, axis=1)
        traj_obj_success = d < BONUS_THRESH

        # find timestep to split
        i = 1
        while not sum(traj_obj_success) > 0:
            traj_obj_success = d < (BONUS_THRESH * i)
            i += 1
            # raise RuntimeError
        if i != 1:
            if log:
                print(f'increased threshold on {o} to {i}x')
        t = np.argmax(traj_obj_success)

        if not (t - prev_t > 0):
            print('failed on', f)
            print(t, prev_t)
            print(traj_obj_success)
            print(d)
            raise RuntimeError

        # save video and singleobj data
        save_video(
            render_buffer[prev_t : t + SINGLEOBJ_TIME_EPS],
            outpath + f'_singleobj_playback{render_meta}',
        )

        if save_data:
            singleobj_data = {}
            for k, v in data.items():
                if k == 'path':
                    p = dict(
                        observations=v['observations'][prev_t : t + SINGLEOBJ_TIME_EPS],
                        actions=v['actions'][prev_t : t + SINGLEOBJ_TIME_EPS],
                    )
                    singleobj_data[k] = p
                elif isinstance(v, np.ndarray):
                    singleobj_data[k] = v[prev_t : t + SINGLEOBJ_TIME_EPS]
                else:
                    singleobj_data[k] = v
            singleobj_data['obj_timeseg'] = (prev_t, t + SINGLEOBJ_TIME_EPS)
            pickle.dump(singleobj_data, open(outpath + '_singleobj.pkl', 'wb'))

        prev_t = t + SINGLEOBJ_TIME_EPS


def process_demos():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_SINGLEOBJ, exist_ok=True)

    global ENVS
    ENVS = [create_env_fn().env for _ in range(NUM_WORKERS)]  # removing TimeLimit wrapper

    random.seed(0)
    np.random.seed(0)
    for i in range(NUM_WORKERS):
        ENVS[i].seed(i + 1)

    tasks = os.listdir(DEMO_DIR)
    demos = glob(DEMO_DIR + '*/*.mjl')

    if WHICH_SET is not None:
        tasks = list(filter(lambda x: WHICH_SET in x, tasks))
        demos = list(filter(lambda x: WHICH_SET in x, demos))

    print(f'tasks: {tasks}')
    print(f'num tasks: {len(tasks)}')
    print(f'num demos: {len(demos)}')
    print()

    # demos = list(filter(lambda x: 'friday_microwave_kettle_switch_slide' in x, demos))[:1]

    if NUM_WORKERS > 1:
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            process_demo_fn = functools.partial(process_demo, log=False)
            results = tqdm(pool.imap(process_demo_fn, demos), total=len(demos))
            results = list(results)
    else:
        for filepath in tqdm(demos, file=sys.stdout):
            process_demo(filepath)


if __name__ == '__main__':
    process_demos()

    # TODO:
    # - try converting demo to mocapik action space of kitchen-v0
    # - kitchen-v1
    # - try converting demo to 5dim action space of kitchen-v1
