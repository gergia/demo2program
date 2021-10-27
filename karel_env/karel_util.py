import logging
import pdb

import numpy as np
from karel import state_table, action_table



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
