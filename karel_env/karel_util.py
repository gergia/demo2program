import logging
import pdb

import numpy as np
from karel import state_table, action_table, Karel_world
from KarelTaskEnvironment import KarelTaskEnvironment
import os
from util import log




class color_code:
    HEADER = '\033[95m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    YELLOW = '\033[93m'
    CYAN = '\033[36m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def get_planners_actions(init_state, end_state):
    PLAN2CODE = {"go": "move", "right": "turnRight", "left": "turnLeft", "get": "pickMarker", "put": "putMarker" }
    PLANNER_OUTPUT = "planner_output.txt"
    PLANNER_SAS_FILE = "sas_plan"
    PDDL_PROBLEM_DIR = "pddl/"


    sq_problem_formulation = states2taskDict(init_state, end_state)
    sq_karel_task_environment = KarelTaskEnvironment(sq_problem_formulation)

    #print(sq_karel_task_environment)

    # creating the planner dir if it does not exist
    try:
        os.makedirs(PDDL_PROBLEM_DIR)
    except:
        pass

    domain_name = "karel-hoc-domain"
    task_name = "karel-task"
    pddl_problem_file = os.path.join(PDDL_PROBLEM_DIR, "problem.pddl")
    with open(pddl_problem_file, "w") as pddl_problem:
        problem_dictionary = sq_karel_task_environment.export_problem_hoc(domain_name, task_name)
        pddl_problem.write(problem_dictionary)

    pddl_domain_file = os.path.join(PDDL_PROBLEM_DIR, "domain.pddl")
    with open(pddl_domain_file, "w") as pddl_domain:
        domain_dictionary = KarelTaskEnvironment.export_domain_hoc(domain_name)
        pddl_domain.write(domain_dictionary)

    planning_command = "./do_the_planning.sh {} {} > {}".format(pddl_domain_file, pddl_problem_file,
                                                                PLANNER_OUTPUT)
    log.debug("planning...")
    os.system(planning_command)
    log.debug("planning done")

    extracted_commands = extract_plan(PLANNER_SAS_FILE)
    commands = [PLAN2CODE[c] for c in extracted_commands]
    #print(extracted_commands)

    planners_code_actions = "DEF run m( " + " ".join(commands) + " m)"

    return planners_code_actions, sq_karel_task_environment


def generate_execution_example(s_gen, h, w, wall_prob, dsl, karel_program, usePlanner=False):
    s, _, _, _, _ = s_gen.generate_single_state(h, w, wall_prob)
    karel_world = Karel_world()
    karel_world.set_new_state(s)
    plan_and_program_differ = False

    # karel_util.state2symbol(s)
    try:
        s_h = dsl.run(karel_world, karel_program)
        actions = get_actions(karel_world)
        kw = karel_world
    except RuntimeError as e:
        raise RuntimeError("Error executing program: {}\n{}\n{}\n\n".format(karel_program, state_repr(s), e))
        # raise Exception()
    else:
        if usePlanner:
            try:
                planners_code_actions, karel_task_environment = get_planners_actions(s_h[0], s_h[-1])
                karel_world_p = Karel_world()
                karel_world_p.set_new_state(s)
                s_h_plan = dsl.run(karel_world_p, planners_code_actions)
                planners_actions = get_actions(karel_world_p)

                if not actions == planners_actions:
                    plan_and_program_differ = True

            except RuntimeError as e:
                raise RuntimeError(
                    log.debug("error planning: {}\n{}".format(planners_code_actions, e))
                )

            else:
                kw = karel_world_p
    return kw, plan_and_program_differ


def grid2str(grid):
    assert len(grid) == 16, 'Invalid representation of a grid'
    idx = np.argwhere(grid == np.amax(grid)).flatten().tolist()
    if len(idx) == 1:
        return state_table[idx[0]]
    elif len(idx) == 2:
        return '{} with {}'.format(state_table[idx[0]], state_table[idx[1]])
    else:
        return 'None'

def extract_plan(plan_file_name):
    plan = []
    with open(plan_file_name) as plan_text:
        for line in plan_text:
            if line.startswith(";"):
                return plan
            else:
                if line.find("hint") >= 0:
                    continue
                try:
                    full_command = line.strip("()")
                    command = full_command.split()[0]
                    plan.append(command)
                except:
                    logging.error("Problem extracting plan: {}".format(plan_file_name))
    return plan


def states2taskDict(s_start, s_end):

    assert s_start.shape == s_end.shape
    h = s_start.shape[0]
    w = s_start.shape[1]


    CARDINAL_DIRECTIONS = {0: "north", 1: "east", 2: "south", 3: "west"}
    task_desc = {}

    # subtracting 2 because in the karel-code-generation codebase it is only the inner fields that count, and the outer
    # fields are walls by default
    W = w - 2
    H = h - 2

    task_desc["width"] = W
    task_desc["height"] = H

    # gold = marker in this repo
    gold_fields = []
    obstacles = []
    for j in range(1, w - 1):
        for i in range(1, h -1):
            if np.sum(s_start[j,i, 6:]) > 0:
                gold_fields.append([i,j])
            if np.sum(s_start[j, i, :4]) > 0:
                task_desc["robot-position-start"] = [i, j]

                idx = np.argmax(s_start[j, i])
                task_desc["robot-orientation-start"] = CARDINAL_DIRECTIONS[idx]
            elif np.sum(s_start[j,i,4]) > 0:
                obstacles.append([i,j])

    task_desc["gold-start"] = gold_fields
    task_desc["obstacles"] = obstacles

    gold_fields = []
    for j in range(1, w - 1):
        for i in range(1, h -1):
            if np.sum(s_end[j,i, :4]) > 0:
                task_desc["robot-position-end"] = [i, j]
                idx = np.argmax(s_end[j, i])
                task_desc["robot-orientation-end"] = CARDINAL_DIRECTIONS[idx]

            if np.sum(s_end[j, i, 6:]) > 0:
                gold_fields.append([i,j])
    task_desc["gold-end"] = gold_fields

    return task_desc




# given a karel env state, return a symbol representation
def state_repr(s):
    KAREL = "^>v<#"
    str = ""
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            # if Karel is on that position, facing one of the directions AND the position has more than one marker
            if np.sum(s[i, j, :4]) > 0 and np.sum(s[i, j, 6:]) > 0:
                idx = np.argmax(s[i, j])
                str += color_code.PURPLE+KAREL[idx]+color_code.END

            # if Karel is only on that position
            elif np.sum(s[i, j, :4]) > 0:
                idx = np.argmax(s[i, j])
                str += color_code.BLUE+KAREL[idx]+color_code.END
            # if there is wall at position (i,j)
            elif np.sum(s[i, j, 4]) > 0:
                str += color_code.RED+'#'+color_code.END
            # if there is more than one item at position (i,j)
            elif np.sum(s[i, j, 6:]) > 0:
                str += color_code.GREEN+'o'+color_code.END
            # if position is a regular one
            else:
                str += '.'
        str += "\n"
    return str




def state2symbol(s):
    print(state_repr(s))


def get_actions(w):
    act = []
    for i in range(len(w.a_h)):
        act.append(action_table[w.a_h[i]])
    return act

# given an instance of karel world, print the history of its states and actions
def print_history(w):
    for i in range(len(w.a_h)):
        state2symbol(w.s_h[i])
        print("---> {} --->".format(action_table[w.a_h[i]]))
    state2symbol(w.s_h[-1])


# given a karel env state, return a visulized image
def state2image(s, grid_size=10, root_dir='./'):
    h = s.shape[0]
    w = s.shape[1]
    img = np.ones((h*grid_size, w*grid_size, 3))
    import h5py
    import os.path as osp
    f = h5py.File(osp.join(root_dir, 'asset/texture.hdf5'), 'r')
    wall_img = f['wall']
    marker_img = f['marker']
    # wall
    y, x = np.where(s[:, :, 4])
    for i in range(len(x)):
        img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = wall_img
    # marker
    y, x = np.where(np.sum(s[:, :, 6:], axis=-1))
    for i in range(len(x)):
        img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = marker_img
    # karel
    y, x = np.where(np.sum(s[:, :, :4], axis=-1))
    if len(y) == 1:
        y = y[0]
        x = x[0]
        idx = np.argmax(s[y, x])
        marker_present = np.sum(s[y, x, 6:]) > 0
        if marker_present:
            if idx == 0:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['n_m']
            elif idx == 1:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['e_m']
            elif idx == 2:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['s_m']
            elif idx == 3:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['w_m']
        else:
            if idx == 0:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['n']
            elif idx == 1:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['e']
            elif idx == 2:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['s']
            elif idx == 3:
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['w']
    elif len(y) > 1:
        raise ValueError
    f.close()
    return img
