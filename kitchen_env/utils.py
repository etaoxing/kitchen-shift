from kitchen_env.constants import DEMO_OBJ_ORDER


def get_task_info(objects_task, objects_done=[]):
    sorted_objects_task = list(sorted(objects_task, key=DEMO_OBJ_ORDER.index))
    sorted_objects_done = list(sorted(objects_done, key=DEMO_OBJ_ORDER.index))
    task_id = 't-' + ','.join(sorted_objects_task)
    if len(objects_done) > 0:
        task_id += '_d-' + ','.join(sorted_objects_done)
    return task_id, sorted_objects_task
