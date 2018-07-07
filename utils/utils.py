import os

def create_directories(list_of_directories):
    try:
        for dir_ in list_of_directories:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as e:
        print("Directory Creation error: {0}", e)
        exit(0)
