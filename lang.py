program_name = "Feature Detection Benchmark"
program_description = "Бенчмарк алгоритмов извлечения особых точек для визуальной одометрии"
argument_no_asking_help = "Быстрый запуск программы. Если остальные флаги не установлены, берёт данные по умолчанию."
greetings_text = """
Приветсвую в Feature Detection Benchmark!
Укажите папку с датасетом:
"""
fail_input_directory_text = """
Такой папки нет. Пожалуйста, укажите действительную папку:
"""
fail_input_filename_text = """
Такого файла нет. Пожалуста, укажите существующий файл:
"""

input_calib_filename_text = """
Укажите файл для калибровки:
"""
input_images_directory_text = """
Укажите папку с изображениями:
"""
input_position_filename_text = """
Укажите файл с данными о местоположении робота:
"""

input_choose_feature_detection_algorithm_text = """
Укажите алгоритм извлечения особых rgb-точек(sift/surf/kaze/brisk/orb): 
"""
fail_input_choose_feature_detection_algorithm_text = """
Такого алгоритма нет. Пожалуста выбере из предложенных: sift, surf, kaze, brisk, orb.
"""

input_threshold = """
Укажите пороговое значение для точек: 
"""
fail_input_threshold = """
Данное значение не является числом. Пожалуйста, укажите пороговое значение для точек: 
"""

input_iteration = """
Укажите количество итераций: 
"""
fail_input_iteration = """
Данное значение не является числом. Пожалуйста, укажите количество итераций: 
"""

question_distortion_text = """
Применить к изображениям искажения?(да/нет)
"""
current_distortions_text = """"
Сейчас применены искажения: 
"""
choose_distortian_text = """
Выберете искажения из предложенных: blur, uniform, impulse, gauss. Внимание: применение искажений может занять продолжительное время.
"""
fail_choose_distortion_text = """
Такого искажения нет.
"""
ask_continue_text = """
Продолжить выбор?(да/нет)
"""

ask_output_directory_text = """
Укажите папку, в которую будет сохранён результат программы:
"""
