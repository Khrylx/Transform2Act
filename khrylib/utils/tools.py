import numpy as np
import os
import shutil
import datetime
import subprocess
from os import path
from PIL import Image
from khrylib.utils.math import *
import cv2


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))


def out_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../out'))


def log_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../logs'))


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


def load_img(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        I = Image.open(f)
        img = I.resize((224, 224), Image.ANTIALIAS).convert('RGB')
        return img


def save_screen_shots(window, file_name, transparent=False, autogui=False):
    import glfw
    xpos, ypos = glfw.get_window_pos(window)
    width, height = glfw.get_window_size(window)
    if autogui:
        import pyautogui
        image = pyautogui.screenshot(region=(xpos*2, ypos*2, width*2, height*2))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA if transparent else cv2.COLOR_RGB2BGR)
        if transparent:
            image[np.all(image >= [240, 240, 240, 240], axis=2)] = [255, 255, 255, 0]
        cv2.imwrite(file_name, image)
    else:
        # print(width*2, height*2)
        subprocess.call(['screencapture', '-x', '-m', f'-R {xpos},{ypos},{width},{height}', file_name])


def save_video_ffmpeg(frame_str, out_file, fps=30, start_frame=0, crf=20):
    cmd = ['ffmpeg', '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', f'{start_frame}',
           '-i', frame_str, '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_file]
    subprocess.call(cmd)


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))


def array_to_str(arr, format_str, sep_str=" "):
    return sep_str.join([format_str] * len(arr)).format(*arr)


def index_select_list(x, ind):
    return [x[i] for i in ind]


def get_graph_fc_edges(num_nodes):
    edges = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            edges.append([i, j])
            edges.append([j, i])
    edges = np.stack(edges, axis=1)
    return edges


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count