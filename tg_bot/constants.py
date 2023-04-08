from os.path import abspath


# путь до папки с проектом "OrdoHereticus_bot"
project_name = 'OrdoHereticus_bot'
path_list = abspath('constants.py').split('/')
PROJECT_PATH = '/'.join(path_list[:path_list.index(project_name) + 1])
# print(PROJECT_PATH)
