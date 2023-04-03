import timeit

import numpy as np

import loadData
from tracking import calculate_track
from visualization import show_track

feature_detection_algorithm = 'SIFT'
images = loadData.load_images('./sample/dataset/00/image_0/')
projection_mat, intrinsic_mat = loadData.load_calib('./sample/dataset/00/calib.txt')
ground_truth = loadData.load_poses('./sample/dataset/00/poses.txt')
gt_path = []
es_path = []
#print(ground_truth[0][0])
track = calculate_track(images, feature_detection_algorithm, projection_mat, intrinsic_mat, initial_pose=ground_truth[0])

for i, gt_pose in enumerate(ground_truth):
    gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
for i, pose in enumerate(track):
    es_path.append((pose[0, 3], pose[2, 3]))


show_track(gt_path, es_path)