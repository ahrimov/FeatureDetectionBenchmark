import os
import sys
import numpy as np
import cv2 as cv
import timeit


import lang

feature_detection_algorithm_names = ['surf', 'sift', 'orb', 'kaze', 'brisk']
# TODO: эффекты искажения могут измениться/дополниться
distortion_effects = ['размытие', 'шум']


def main():
    calib_filename = ''
    images_directory_name = ''
    true_position_filename = ''
    path_to_dataset_directory = input(lang.greetings_text)
    while path_to_dataset_directory != '' and not os.path.exists(path_to_dataset_directory):
        path_to_dataset_directory = input(lang.fail_input_directory_text)

    if path_to_dataset_directory == '':
        path_to_dataset_directory = os.getcwd() + '/sample/'
        calib_filename = 'calib.txt'
        images_directory_name = '/images/'
        true_position_filename = 'poses.txt'
    else:
        calib_filename = input(lang.input_calib_filename_text)
        while calib_filename != '' and not os.path.exists(calib_filename):
            calib_filename = input(lang.fail_input_filename_text)
        images_directory_name = input(lang.input_images_directory_text)
        while not os.path.exists(images_directory_name):
            images_directory_name = input(lang.fail_input_directory_text)
        true_position_filename = input(lang.input_position_filename_text)
        while true_position_filename != '' and not os.path.exists(true_position_filename):
            true_position_filename = input(lang.fail_input_filename_text)

    feature_detection_algorithm = input(lang.input_choose_feature_detection_algorithm_text)
    while feature_detection_algorithm != '' and feature_detection_algorithm not in feature_detection_algorithm_names:
        feature_detection_algorithm = input(lang.fail_input_choose_feature_detection_algorithm_text)
    if feature_detection_algorithm == '':
        feature_detection_algorithm = 'sift'

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

    start = timeit.default_timer()

    # TODO: вычисления местоположения робота
    # calculate()

    stop = timeit.default_timer()

    print('Время выполнения вычислений: ', stop - start)

    if not os.path.exists(os.getcwd() + output_directory):
        os.mkdir(os.getcwd() + '/' + output_directory)


if __name__ == '__main__':
    sys.exit(main())
