from os.path import abspath

# Путь до папки с проектом Windows
# PROJECT_PATH = ""
# for i in abspath('constants.py').split("\\")[:-1]:
#     PROJECT_PATH = PROJECT_PATH + f"{i}\\"
# print(PROJECT_PATH)

# Путь до папки с проектом Ubuntu
PROJECT_PATH = ""
for i in abspath('constants.py').split("/")[:-1]:
    PROJECT_PATH += f"{i}/"
# print(PROJECT_PATH)
