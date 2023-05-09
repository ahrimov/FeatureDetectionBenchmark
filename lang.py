program_name = "Feature Detection Benchmark."
program_description = "Бенчмарк алгоритмов извлечения особых точек для визуальной одометрии."
argument_no_asking_help = "Быстрый запуск программы. Если остальные флаги не установлены, берёт данные по умолчанию."
argument_quaternion_help = "Входные данные позиций представлены в виде кватернионов."
argument_dataset_directory_help = "Путь к основной папке с набором данных."
argument_images_help = "Путь к папке с изображениями."
argument_calib_help = "Название файла с настройками камеры."
argument_poses_help = "Название файла с позициями."
argument_order_help = "Порядок координат в файле с позициями."
argument_algorithms_help = "Выбор алгоритмов извлечения и описания особых точек на RGB-изображении."
argument_threshold_help = "Оценка 'хороших совпадений' по Лоу."
argument_iteration_help = "Количество итераций."
argument_output_help = "Название директории, в которой будет лежать результат."
argument_start_help = "Индекс первого изображения."
argument_end_help = "Индекс последнего изображения."
argument_step_help = "Шаг загрузки изображений."
argument_effects_help = "Добавление эффектов на изображения."
greetings_text = """Приветсвую в Feature Detection Benchmark!
Укажите папку с датасетом:
"""
fail_input_directory_text = """Такой папки нет. Пожалуйста, укажите действительную папку:
"""
fail_input_filename_text = """Такого файла нет. Пожалуста, укажите существующий файл:
"""

input_calib_filename_text = """Укажите файл c параметрами внутренней калибровки камеры:
"""
input_images_directory_text = """Укажите папку с изображениями:
"""
input_position_filename_text = """Укажите файл с данными о истинном местоположении робота:
"""

input_choose_feature_detection_algorithm_text = """Укажите алгоритмы извлечения особых точек из предложенных(sift/surf/kaze/brisk/orb): 
"""
fail_input_choose_feature_detection_algorithm_text = """Такого алгоритма нет. Пожалуста выбере из предложенных: sift, surf, kaze, brisk, orb.
"""

input_threshold = """Укажите пороговое значение для оценки совпадений по Лоу от 0 до 1: 
"""
fail_input_threshold = """Данное значение не является числом. Пожалуйста, укажите пороговое значение оценки совпадений по Лоу от 0 до 1: 
"""

input_iteration = """Укажите количество итераций: 
"""
fail_input_iteration = """Данное значение не является числом. Пожалуйста, укажите количество итераций (целое число): 
"""

question_distortion_text = """Применить к изображениям искажения?(да/нет)
"""
current_distortions_text = """"Сейчас применены искажения: 
"""
choose_distortian_text = """Выберете искажения из предложенных: blur, uniform, impulse, gauss. 
Внимание: применение искажений может занять продолжительное время.
"""
fail_choose_distortion_text = """Такого искажения нет."""
ask_continue_text = """Продолжить выбор?(да/нет)
"""

ask_output_directory_text = """Внимание: после данного этапа начнуться вычисления, которые могут знаять продолжительное время.
Укажите папку, в которую будет сохранён результат программы:
"""

help_inverse = "Инвертировать матрицу трансформации."

error_out_of_range_load_images = """Ошибка. Вышли за пределы доступных изображений."""