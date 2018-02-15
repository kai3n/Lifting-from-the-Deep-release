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
    video_writer1 = cv2.VideoWriter(PROJECT_PATH + '/data/videos/test_result.mp4', fourcc, 10.0, (640, 480))
    # video_writer2 = cv2.VideoWriter(PROJECT_PATH + '/data/videos/test_result_3d.mp4', fourcc, 10.0, (640, 480))
    cap = cv2.VideoCapture(PROJECT_PATH + '/data/videos/test_video.mp4')

    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    pose = np.array([[  11.3526683, -82.55155479, -148.66113686, -65.41943422 , 91.45557751,
     53.26781147, 66.86493085, 38.04185689, 7.28628714, -48.41194608,
     -5.47691749, 139.73787422, 223.42598994, 247.67292748, -118.95568127,
   -196.4879065,  -212.15876749],
  [  23.02331676,  124.74562903,    8.44339516,   70.14820979,  -78.69893769,
   -180.41105152,  -83.59408481,   53.64574645,   20.80042024,  -55.12668849,
    -12.24949449,  -72.19321393, -131.84483411, -211.98540937,  136.81573026,
    235.5655179,   152.61293169],
  [-116.93687833, -126.13634777, -473.11945881, -837.70550931, -127.37170391,
   -478.14888198, -885.36476784,  184.5672324,   563.56434648,  603.01735506,
    809.97294291,  454.13285347,  178.52674073,  -67.75306918,  497.82676333,
    232.63506518,   -4.05807492]])
    pose_list = [pose]

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
        smallest = pose.min()
        largest = pose.max()
        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    count = 0
    while cap.isOpened():
        count += 1
        if count > 100:
            break
        ret_val, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        # create pose estimator
        frame_size = frame.shape

        pose_estimator = PoseEstimator(frame_size, SESSION_PATH, PROB_MODEL_PATH)

        # load model
        pose_estimator.initialise()

        try:
            # estimation
            pose_2d, visibility, pose_3d = pose_estimator.estimate(frame)
        except:
            continue

        pose_list.append(pose_3d[0])

        # close model
        pose_estimator.close()

        draw_limbs(frame, pose_2d, visibility)

        video_writer1.write(frame)

    with writer.saving(fig, PROJECT_PATH + '/data/videos/test_result_3d.mp4', 100):
        for i in range(len(pose_list)):
            animate(pose_list[i])
            writer.grab_frame()

    # im_ani.save(PROJECT_PATH + '/data/videos/test_result_3d.mp4', writer=writer)
    # video_writer1.release()


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
    estimate_video()


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
