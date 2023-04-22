import os
import sys
import argparse
import timeit

import numpy as np

import lang
import loadData
from image_distortion import add_blur, add_sp_noise, add_uniform_noise, add_gaussian_noise
from tracking import calculate_track, estimate_path
from visualization import show_track, show_tracks

feature_detection_algorithm_names = ['surf', 'sift', 'orb', 'kaze', 'brisk']
# TODO: эффекты искажения могут измениться/дополниться
distortion_effects = ['blur', 'impulse', 'uniform', 'gauss']


def main():
    path_to_dataset_directory = '/sample/dataset/00/'
    calib_filename = 'calib.txt'
    images_directory_name = '/images/'
    true_position_filename = 'poses.txt'
    feature_detection_algorithm = 'sift'
    output_directory = '/outputs/'
    threshold = 1
    iteration = 1
    image_distortions = []
    parser = argparse.ArgumentParser(prog=lang.program_name, description=lang.program_description)
    parser.add_argument('-n', '--no-asking', dest='no_asking', action='store_const',
                        const=True, default=False, help=lang.argument_no_asking_help)
    parser.add_argument('-q', '--quaternion', dest='quaternion_poses', action='store_const',
                        const=True, default=False)
    parser.add_argument('-inv', '--inverse', dest='inverse_mat', action='store_const',
                        const=True, default=False, help=lang.help_inverse)
    parser.add_argument('-p', '--path_to_dataset', dest='path_to_dataset',
                        required=False, default=path_to_dataset_directory)
    parser.add_argument('-i', '--images_directory_name', dest='images_directory_name',
                        required=False, default=images_directory_name)
    parser.add_argument('-c', '--calib_filename', dest='calib_filename',
                        required=False, default=calib_filename)
    parser.add_argument('-gt', '--positions_filename', dest='true_position_filename',
                        required=False, default=true_position_filename)
    parser.add_argument('-o', '--order', dest='order', required=False, default='xyz')
    parser.add_argument('-a', '--feature_detection_algorithm', dest='feature_detection_algorithm',
                        required=False, default=feature_detection_algorithm)
    parser.add_argument('-t', '--threshold', type=float, dest='threshold',
                        required=False, default=threshold)
    parser.add_argument('-it', '--iteration', type=int, dest='iteration',
                        required=False, default=iteration)
    parser.add_argument('-out', '--output_directory', dest='output_directory',
                        required=False, default=output_directory)
    parser.add_argument('-st', '--start_image', type=int, dest='start_image',
                        required=False, default=0)
    parser.add_argument('-end', '--end_image', type=int, dest='end_image',
                        required=False, default=0)
    parser.add_argument('-count', '--count_image', type=int, dest='count_image',
                        required=False, default=1)
    args = parser.parse_args()

    if not args.no_asking:
        args_from_line = interaction_with_user()
        path_to_dataset_directory = os.getcwd() + '/' + args_from_line[0]
        calib_filename = args_from_line[1]
        images_directory_name = args_from_line[2]
        true_position_filename = args_from_line[3]
        feature_detection_algorithm = args_from_line[4]
        threshold = args_from_line[5]
        iteration = args_from_line[6]
        image_distortions = args_from_line[7]
        output_directory = args_from_line[8]
    else:
        path_to_dataset_directory = os.getcwd() + '/' + args.path_to_dataset
        images_directory_name = args.images_directory_name
        calib_filename = args.calib_filename
        true_position_filename = args.true_position_filename
        feature_detection_algorithm = args.feature_detection_algorithm
        threshold = args.threshold
        iteration = args.iteration
        output_directory = args.output_directory

    algorithms = []
    feature_detection_algorithm = feature_detection_algorithm.split(',')
    for alg in feature_detection_algorithm:
        if alg in feature_detection_algorithm_names:
            algorithms.append(alg)


    projection_mat, intrinsic_mat = loadData.load_calib(path_to_dataset_directory + '/' + calib_filename)
    if args.quaternion_poses:
        ground_truth = loadData.load_quaternion_poses(path_to_dataset_directory + '/' + true_position_filename)
    else:
        ground_truth = loadData.load_poses(path_to_dataset_directory + '/' + true_position_filename)

    for distortion in image_distortions:
        if distortion == 'blur':
            images = add_blur(images)
        elif distortion == 'impulse':
            images = add_sp_noise(images, 0.05)
        elif distortion == 'uniform':
            images = add_uniform_noise(images, 0.5)
        elif distortion == 'gauss':
            images = add_gaussian_noise(images, 0.5)

    if args.start_image != args.end_image:
        images = loadData.load_images(path_to_dataset_directory + '/' + images_directory_name + '/',
                                               args.start_image, args.end_image, args.count_image)
        ground_truth = ground_truth[args.start_image:args.end_image:args.count_image]
    else:
        images = loadData.load_images(path_to_dataset_directory + '/' + images_directory_name + '/')
    init_pose = ground_truth[0]
    #init_pose = np.array([init_pose[0,3], init_pose[1,3], init_pose[2,3]])
    gt_path = []
    x_index = args.order.find('x')
    y_index = args.order.find('y')
    for i, gt_pose in enumerate(ground_truth):
        gt_path.append((gt_pose[x_index, 3], gt_pose[y_index, 3]))

    gt_norm = np.linalg.norm(np.array(gt_path), axis=1)

    track = np.array([])
    result = {}
    for feature_detection_algorithm in algorithms:
        time = 0
        for i in range(iteration):
            start = timeit.default_timer()
            piece = np.array(calculate_track(images, feature_detection_algorithm,
                                    intrinsic_mat, initial_pose=init_pose, ground_truth=ground_truth,
                                             threshold=threshold, inverse_mat=args.inverse_mat))
            if i == 0:
                track = piece
            else:
                track += piece
            stop = timeit.default_timer()
            current_time = stop - start
            time += current_time
            print('Время выполнения вычислений для итерации {}: {} c.'.format(i+1, round(current_time,4)))

        track = track / iteration
        time /= iteration

        print('Среднее время выполнения: {} c.'.format(round(time,4)))
        es_path = []
        for i, pose in enumerate(track):
            es_path.append((pose[x_index, 3], pose[y_index, 3]))
        result[feature_detection_algorithm] = es_path

        error = np.linalg.norm(np.array(gt_path) - np.array(es_path), axis=1)
        rate = error/gt_norm * 100
        average_rate = np.average(rate)
        print("Средний процент ошибок {}: {}%".format(feature_detection_algorithm, round(average_rate, 2)))
    # print(track)
    # for i, pose in enumerate(track):
    #      es_path.append((pose[0], pose[2]))
    show_tracks(gt_path, result, output_directory)

    if not os.path.exists(os.getcwd() + output_directory):
        os.mkdir(os.getcwd() + '/' + output_directory)


def interaction_with_user():
    calib_filename = 'calib.txt'
    images_directory_name = '/images/'
    true_position_filename = 'poses.txt'
    threshold = 1
    iteration = 1
    path_to_dataset_directory = input(lang.greetings_text)
    while path_to_dataset_directory != '' and not os.path.exists(path_to_dataset_directory):
        path_to_dataset_directory = input(lang.fail_input_directory_text)
    if path_to_dataset_directory == '':
        path_to_dataset_directory = '/sample/dataset/00/'
    else:
        calib_filename = input(lang.input_calib_filename_text)
        while calib_filename != '' and not os.path.exists(path_to_dataset_directory + '/' + calib_filename):
            calib_filename = input(lang.fail_input_filename_text)
        if calib_filename == '':
            calib_filename = 'calib.txt'
        images_directory_name = input(lang.input_images_directory_text)
        while not os.path.exists(path_to_dataset_directory + '/' + images_directory_name):
            images_directory_name = input(lang.fail_input_directory_text)
        if images_directory_name == '':
            images_directory_name = '/image_0/'
        true_position_filename = input(lang.input_position_filename_text)
        while true_position_filename != '' and not os.path.exists(path_to_dataset_directory + '/' + true_position_filename):
            true_position_filename = input(lang.fail_input_filename_text)
        if true_position_filename == '':
            true_position_filename = 'poses.txt'

    feature_detection_algorithm = input(lang.input_choose_feature_detection_algorithm_text)
    while feature_detection_algorithm != '' and not is_feature_detection_alg(feature_detection_algorithm):
        feature_detection_algorithm = input(lang.fail_input_choose_feature_detection_algorithm_text)
    if feature_detection_algorithm == '':
        feature_detection_algorithm = 'sift'

    threshold = input(lang.input_threshold)
    while threshold != '' and not is_float(threshold):
        threshold = input(lang.fail_input_threshold)
    if threshold == '':
        threshold = 1
    else:
        threshold = float(threshold)

    iteration = input(lang.input_iteration)
    while iteration != '' and not iteration.isnumeric():
        iteration = input(lang.fail_input_iteration)
    if iteration == '':
        iteration = 1
    else:
        iteration = int(iteration)

    image_distortions = set()
    user_answer = input(lang.question_distortion_text)
    print(user_answer)
    if user_answer == "да":
        while True:
            if image_distortions.__len__() != 0:
                print(lang.current_distortions_text + ", ".join(image_distortions))
            distortion = input(lang.choose_distortian_text)
            if distortion == '':
                break
            if distortion in distortion_effects:
                image_distortions.add(distortion)
            else:
                print(lang.fail_choose_distortion_text)
            if input(lang.ask_continue_text) != 'да':
                break

    output_directory = input(lang.ask_output_directory_text)
    if output_directory == '':
        output_directory = '/outputs/'
    return path_to_dataset_directory, calib_filename, images_directory_name, true_position_filename, \
           feature_detection_algorithm, threshold, iteration, image_distortions, output_directory


def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def is_feature_detection_alg(string):
    feature_detection_algorithm = string.split(',')
    for alg in feature_detection_algorithm:
        if alg not in feature_detection_algorithm_names:
            return False
    return True


if __name__ == '__main__':
    sys.exit(main())
