import sys
import os
sys.path.append(os.getcwd())

import tensorflow as tf

from device.device import Device

def main():
    # change to get arguments from command line
    device = Device()
    session = device.get_session()

if __name__ == '__main__':
    main()
