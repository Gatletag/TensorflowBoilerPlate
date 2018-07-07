from tensorflow.python.client import device_lib
import tensorflow as tf

class Device(object):
    def __init__(self):
        self.devices = device_lib.list_local_devices()
        self.device_names = []
        for device in self.devices:
            self.device_names.append(device.name)

    def get_device_list(self):
        return self.device_names

    def get_device_details(self):
        return self.devices

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        return session
