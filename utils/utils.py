import os

def create_directories(list_of_directories):
    try:
        for dir in list_of_directories:
            os.makedirs(dir)
    except Exception as e:
        print("Directory Creation error: {0}", e)
        exit(0)
