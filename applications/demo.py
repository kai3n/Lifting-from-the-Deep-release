#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

# import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import time
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os.path import dirname, realpath


DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image2.png'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

_CONNECTION = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
            [15, 16]]

def estimate_image():
    image = cv2.imread(IMAGE_FILE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

    # create pose estimator
    image_size = image.shape

    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    # estimation
    pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

    # close model
    pose_estimator.close()

    # Show 2D and 3D poses
    display_results(image, pose_2d, visibility, pose_3d)

pose = []

def estimate_video():

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer1 = cv2.VideoWriter(PROJECT_PATH + '/data/videos/test_result_{}.mp4'.format(str(time.time())), fourcc, 30.0, (640, 480))
    cap = cv2.VideoCapture(PROJECT_PATH + '/data/videos/test_video.mp4')

    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    pose_list = []

    def animate(pose):


        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)

        ax.clear()
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                       c=col, marker='o', edgecolor=col)
        # smallest = pose.min()
        # largest = pose.max()
        # print('smallest:', smallest)  # -885.36476784
        # print('largest:', largest)  # 809.97294291
        #
        # ax.set_xlim3d(smallest, largest)
        # ax.set_ylim3d(smallest, largest)
        # ax.set_zlim3d(smallest, largest)

        ax.set_xlim3d(-1000, 1000)
        ax.set_ylim3d(-1000, 1000)
        ax.set_zlim3d(-1000, 1000)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=30, metadata=metadata)

    # create pose estimator

    pose_estimator = PoseEstimator((480, 640, 3), SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    count = 0
    while cap.isOpened():
        count += 1
        if count == 300:
            break
        print('count:{}'.format(str(count)))
        ret_val, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))


        try:
            # estimation
            start = time.time()
            pose_2d, visibility, pose_3d = pose_estimator.estimate(frame)
            print(time.time()-start)
        except:
            continue

        pose_list.append(pose_3d[0])
        draw_limbs(frame, pose_2d, visibility)
        video_writer1.write(frame)

    # close model
    pose_estimator.close()


    with writer.saving(fig, PROJECT_PATH + '/data/videos/test_result_3d_{}.mp4'.format(str(time.time())), 100):
        for i in range(len(pose_list)):
            animate(pose_list[i])
            writer.grab_frame()

    # im_ani.save(PROJECT_PATH + '/data/videos/test_result_3d.mp4', writer=writer)
    # video_writer1.release()


def estimate_video_and_save_pkl():

    pose_list = []
    cap = cv2.VideoCapture(PROJECT_PATH + '/data/videos/test_video.mp4')

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # create pose estimator
    pose_estimator = PoseEstimator((480, 640, 3), SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    count = 0
    while cap.isOpened():
        count += 1
        if count == 300:
            break
        print('count:{}'.format(str(count)))
        ret_val, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        try:
            # estimation
            start = time.time()
            pose_2d, visibility, pose_3d = pose_estimator.estimate(frame)
            finish = time.time() - start
            print("time:", finish)
        except:
            continue

        pose_list.append(pose_3d[0])
    # close model
    pose_estimator.close()

    with open(PROJECT_PATH + '/data/videos/' + '3d_joints_{}'.format(str(time.time())), 'wb') as fo:
        pickle.dump(pose_list, fo)


def load_pkl(filename):
    with open(PROJECT_PATH + '/data/videos/' + filename, 'rb') as fi:
        data = pickle.load(fi)
    return data


def joint_color(j):
    """
    TODO: 'j' shadows name 'j' from outer scope
    """

    colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
              (0, 255, 255), (255, 0, 0), (0, 255, 0)]
    _c = 0
    if j in range(1, 4):
        _c = 1
    if j in range(4, 7):
        _c = 2
    if j in range(9, 11):
        _c = 3
    if j in range(11, 14):
        _c = 4
    if j in range(14, 17):
        _c = 5
    return colors[_c]


def main():
    # estimate_video()
    estimate_video_and_save_pkl()

def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
