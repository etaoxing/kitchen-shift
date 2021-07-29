import os, sys
import argparse
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
import time

import kitchen_env
from kitchen_env.adept_envs.utils.parse_mjl import parse_mjl_logs, viz_parsed_mjl_logs

parser = argparse.ArgumentParser()
parser.add_argument(
    '--NUM_WORKERS',
    # default=1,
    default=12,
    type=int,
)
parser.add_argument('--RECREATE_ENVS', default=True, type=bool)
ENVS = []  # populated later if recreating envs
parser.add_argument('--DEMO_DIR', default='kitchen_demos_multitask/', type=str)
parser.add_argument('--WHICH_SET', default=None, choices=[None, 'postcorl', 'friday'])
parser.add_argument('--OUT_DIR', default='workdir/', type=str)
# env params
parser.add_argument(
    '--ENV_NAME',
    # default='kitchen-v0',
    default='kitchen-v1',
    type=str,
)
parser.add_argument('--ROBOT', default='franka2', type=str)
parser.add_argument('--CTRL_MODE', default='absvel')
parser.add_argument('--FPS', default=30.0, type=float)
parser.add_argument('--FRAME_SKIP', default=40, type=int)
parser.add_argument(
    '--CAMERA_ID',
    default=6,
    type=int,
)
parser.add_argument(
    '--RENDER_SIZE',
    # default=(64, 64)
    default=(128, 128),
    # default=(120, 160)
    # default=(256, 256)
    # default=(240, 320)
    # default=(480, 640)
    # default=(1920, 2560)
    nargs=2,
    type=int,
)
parser.add_argument(
    '--BONUS_THRESH',
    default=0.1,
    # lowered threshold compared to 0.3 (guarantees kettle in place when splitting singleobj)
    type=float,
)
parser.add_argument('--SINGLEOBJ_TIME_EPS', default=15, type=int)
parser.add_argument('--MAX_ERROR_IGNORE_THRESH', default=0.8, type=float)

parser.add_argument('--disable_tqdm', action='store_true')
args = parser.parse_args()
args.RENDER_SIZE = tuple(args.RENDER_SIZE)
args.FOURCC = cv2.VideoWriter_fourcc(*'XVID')
args.VIDEO_EXT = 'avi'

# sim params unrelated to rendering
env_kwargs = dict(
    frame_skip=args.FRAME_SKIP,
    ctrl_mode=args.CTRL_MODE,
    # compensate_gravity=True,
    with_obs_ee=True,
    with_obs_forces=True,
    rot_use_euler=True,
    robot=args.ROBOT,
    #
    # noise_ratio=0.1,
    # object_pos_noise_amp=0.1,
    # object_vel_noise_qmp=0.1,
)
# TODO: we spawn a separate env if want to convert demos to a different action space

create_env_fn = functools.partial(
    gym.make,
    args.ENV_NAME,
    **env_kwargs,
    camera_id=args.CAMERA_ID,
)


from kitchen_env.constants import (
    ORIG_OBS_ELEMENT_INDICES,
    OBS_ELEMENT_INDICES,
    OBS_ELEMENT_GOALS,
    ORIG_FRANKA_INIT_QPOS,
)
from kitchen_env.utils import get_task_info


def save_video(render_buffer, filepath):
    vw = cv2.VideoWriter(filepath, args.FOURCC, args.FPS, args.RENDER_SIZE[::-1])
    for frame in render_buffer:
        frame = frame.copy()
        # frame = cv2.resize(frame, RENDER_SIZE[::-1])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vw.write(frame)
    vw.release()


def render_demo(env, data, use_physics=False, log=True):
    # render_skip = max(1, round(1.0 / (args.FPS * env.sim.model.opt.timestep * env.frame_skip)))
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

        # if i_frame % render_skip == 0:
        if True:
            curr_frame = env.render(
                mode='rgb_array', height=args.RENDER_SIZE[0], width=args.RENDER_SIZE[1]
            )
            render_buffer.append(curr_frame)
            if log:
                print(f'{i_frame}', end=', ', flush=True)

        if use_physics:
            path['observations'].append(obs)
            if i_frame == N:  # for adding final observation
                break

            # Construct the action
            robot_qp = obs['robot_qp']

            if env.ctrl_mode == 'absvel':
                # ctrl = (data['qpos'][i_frame + 1][:9] - robot_qp) / (env.frame_skip * env.model.opt.timestep)
                ctrl = (data['ctrl'][i_frame] - robot_qp) / (
                    env.frame_skip * env.model.opt.timestep
                )
                act = (ctrl - env.act_mid) / env.act_amp
                act = np.clip(act, -0.999, 0.999)
            elif env.ctrl_mode == 'abspos':
                ctrl = data['ctrl'][i_frame]
                act = (ctrl - env.act_mid) / env.act_amp
                act = np.clip(act, -0.999, 0.999)
            else:
                raise RuntimeError

            # TODO:
            # - converting demo to other action spaces / instantations of the env

            next_obs, reward, done, env_info = env.step(act)

            path['actions'].append(act)

        i_frame += 1

    if log:
        print()
        print("physics = %i, time taken = %f" % (use_physics, timer.time() - t0))

    if use_physics:
        # check the difference b/w the demo path and the playback path and make sure its valid
        errors = []
        for i_frame in range(N):
            view_qpos = data['qpos'][i_frame]
            o = path['observations'][i_frame]
            playback_qpos = np.concatenate([o['robot_qp'], o['obj_qp']])
            f_error = np.linalg.norm(view_qpos - playback_qpos)
            errors.append(f_error)
        path['errors'] = np.array(errors)

        tot_error = np.sum(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        med_error = np.median(errors)

        if log:
            print(f'over {N} frames, view demo to physics playback error:')
            print(
                f'tot_error = {tot_error}, min_error = {min_error}, max_error={max_error}, med_error={med_error}'
            )

        # path['observations'] = np.vstack(path['observations'])
        path['actions'] = np.vstack(path['actions'])
        return render_buffer, path
    else:
        return render_buffer


def swap_obj_indices(data):
    qpos = data['qpos'][:].copy()
    qvel = data['qpos'][:].copy()

    for k, v1 in ORIG_OBS_ELEMENT_INDICES.items():
        v2 = OBS_ELEMENT_INDICES[k]
        assert len(v1) == len(v2)
        if np.any(v1 != v2):
            data['qpos'][:, v2] = qpos[:, v1]
            data['qvel'][:, v2] = qvel[:, v1]
    return data


def splice_mid_reset(filepath, data, log=True):
    # t = np.where(
    #     np.all(
    #         np.isclose(
    #             data['qpos'][:, :9],
    #             ORIG_FRANKA_INIT_QPOS[:9],
    #             atol=0.01,
    #             rtol=0.01,
    #         ),
    #         axis=-1,
    #     )
    # )[0]

    diffs = np.sum(np.diff(data['qpos'][:, :9], axis=0), axis=-1)
    t = np.where(diffs > 0.3)[0]
    if len(t) > 0:
        if log:
            print(filepath, len(t))

        if len(t) != 1:  # should only be one reset t
            raise RuntimeError

        t = t[0]  # pick first timestep

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) == 2:
                    data[k] = v[t + 1 :, :]
                elif len(v.shape) == 1:
                    data[k] = v[t + 1 :]
                else:
                    raise RuntimeError

    return data


def process_demo(
    x,
    save_data=True,
    view_demo=True,  # view demos (physics ignored)
    playback_demo=True,  # playback demos and get data(physics respected)
    split_singleobj=True,
    log=True,
):
    demo_id, filepath = x

    global ENVS, args

    if args.RECREATE_ENVS:
        env = create_env_fn().env  # removing TimeLimit wrapper
    else:
        worker_id = multiprocessing.current_process()._identity
        if worker_id == ():
            worker_id = 0  # then not using pool
        else:
            worker_id = worker_id[0] - 1  # subtracting 1 b/c _identity starts from 1
        env = ENVS[worker_id]
    env.seed(demo_id)

    _, task, fn = filepath.split('/')
    which_set = task.split('_')[0]
    task_objects = task.split('_')[1:]
    task_id, sorted_objects_task = get_task_info(task_objects, objects_done=[])
    task_id = which_set + '_' + task_id

    if log:
        print(filepath, task_id)

    os.makedirs(os.path.join(args.OUT_DIR, 'full', task_id), exist_ok=True)
    f = fn.split('.mjl')[0]
    outpath = os.path.join(args.OUT_DIR, 'full', task_id, f)

    data = parse_mjl_logs(filepath, args.FRAME_SKIP)
    # need to swap qpos and qvel indices since changed body order in xml
    data = swap_obj_indices(data)
    # sometimes, the demos reset to the starting robot pos in the middle.
    # if this occurs, find that timestep and begin the data at that time
    data = splice_mid_reset(filepath, data, log=log)

    render_meta = f'-c{args.CAMERA_ID}h{args.RENDER_SIZE[0]}w{args.RENDER_SIZE[1]}.{args.VIDEO_EXT}'

    if playback_demo:
        try:
            render_buffer, path = render_demo(env, data, use_physics=True, log=log)
        except Exception as e:
            print(f'skipped playback {filepath}, {e}')
            raise e
            return -1

        if args.MAX_ERROR_IGNORE_THRESH is not None:
            errors = path['errors']
            max_error = np.max(errors)
            if max_error > args.MAX_ERROR_IGNORE_THRESH:
                print(f'ignoring demo {filepath}, max_error = {max_error}')

                if args.RECREATE_ENVS:
                    env.close()

                return -1

        data['path'] = path
        save_video(render_buffer, outpath + f'_playback{render_meta}')

    if view_demo:
        render_buffer = render_demo(env, data, use_physics=False, log=log)
        save_video(render_buffer, outpath + f'_view{render_meta}')

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

    if args.RECREATE_ENVS:
        env.close()


def process_demo_split_singleobj(
    which_set, sorted_objects_task, f, data, render_buffer, render_meta, save_data=True, log=True
):
    prev_t = 0
    for i, o in enumerate(sorted_objects_task):
        task_id_singleobj, _ = get_task_info([o], objects_done=sorted_objects_task[:i])
        task_id_singleobj = which_set + '_' + task_id_singleobj
        os.makedirs(os.path.join(args.OUT_DIR, 'singleobj', task_id_singleobj), exist_ok=True)

        outpath = os.path.join(args.OUT_DIR, 'singleobj', task_id_singleobj, f)

        obs_indices = OBS_ELEMENT_INDICES[o]
        obs_goals = OBS_ELEMENT_GOALS[o]

        ag = data['qpos'][:, obs_indices]
        d = np.linalg.norm(ag - obs_goals, axis=1)
        traj_obj_success = d < args.BONUS_THRESH

        # find timestep to split
        i = 1
        while not sum(traj_obj_success) > 0:
            traj_obj_success = d < (args.BONUS_THRESH * i)
            i += 1
            # raise RuntimeError
        if i != 1:
            if log:
                print(f'increased threshold on {o} to {i}x')
        t = np.argmax(traj_obj_success)

        if not (t - prev_t > 0):
            print('failed on', f, which_set, sorted_objects_task)
            print(t, prev_t)
            print(traj_obj_success)
            print(d)
            print(ag)
            # print(np.round(data['qpos'][:, np.array([9, 10, 11, 12, 13, 14, 15, 16])], 3).tolist())
            raise RuntimeError

        # save video and singleobj data
        save_video(
            render_buffer[prev_t : t + args.SINGLEOBJ_TIME_EPS + 1],
            outpath + f'_singleobj_playback{render_meta}',
        )

        if save_data:
            singleobj_data = {}
            for k, v in data.items():
                if k == 'path':
                    p = dict(
                        observations=v['observations'][prev_t : t + args.SINGLEOBJ_TIME_EPS + 1],
                        actions=v['actions'][prev_t : t + args.SINGLEOBJ_TIME_EPS],
                    )
                    singleobj_data[k] = p
                elif isinstance(v, np.ndarray):
                    singleobj_data[k] = v[prev_t : t + args.SINGLEOBJ_TIME_EPS]
                else:
                    singleobj_data[k] = v
            singleobj_data['obj_timeseg'] = (prev_t, t + args.SINGLEOBJ_TIME_EPS)
            pickle.dump(singleobj_data, open(outpath + '_singleobj.pkl', 'wb'))

        prev_t = t + args.SINGLEOBJ_TIME_EPS


def process_demos():
    os.makedirs(args.OUT_DIR, exist_ok=True)

    global ENVS
    if args.RECREATE_ENVS:
        ENVS = []
    else:
        ENVS = [create_env_fn().env for _ in range(args.NUM_WORKERS)]  # removing TimeLimit wrapper

    random.seed(0)
    np.random.seed(0)
    if not args.RECREATE_ENVS:
        for i in range(args.NUM_WORKERS):
            ENVS[i].seed(i + 1)

    tasks = os.listdir(args.DEMO_DIR)
    demos = glob(args.DEMO_DIR + '*/*.mjl')

    if args.WHICH_SET is not None:
        tasks = list(filter(lambda x: args.WHICH_SET in x, tasks))
        demos = list(filter(lambda x: args.WHICH_SET in x, demos))

    print(f'tasks: {tasks}')
    print(f'num tasks: {len(tasks)}')
    print(f'num demos: {len(demos)}')
    print()

    # demos = list(filter(lambda x: 'friday_microwave_kettle_switch_slide' in x, demos))[:1]

    start = time.perf_counter()

    if args.NUM_WORKERS > 1:
        try:
            pool = multiprocessing.Pool(args.NUM_WORKERS)
            process_demo_fn = functools.partial(process_demo, log=False)
            results = tqdm(
                pool.imap(process_demo_fn, enumerate(demos)),
                total=len(demos),
                disable=args.disable_tqdm,
            )
            results = list(results)
        except Exception as e:
            pool.terminate()
            raise e
        pool.close()
        pool.join()
    else:
        for i, filepath in tqdm(
            enumerate(demos),
            file=sys.stdout,
            disable=args.disable_tqdm,
        ):
            process_demo((i, filepath))

    process_time = time.perf_counter() - start
    with open(os.path.join(args.OUT_DIR, 'args.log'), 'w') as f:
        f.write(str(args))
        f.write('\n')
        f.write(str(env_kwargs))
        f.write('\n')
        f.write(f'process_time = {process_time}')


if __name__ == '__main__':
    if args.RECREATE_ENVS:
        multiprocessing.set_start_method('forkserver')
    else:
        multiprocessing.set_start_method('fork')

    process_demos()
