# import random
# import carla

# class CarlaEnvironment:
#     def __init__(self):
#         pass


###########################################################################
###########################################################################
# Class Project
# Travis Bonneau, Sennan Liu
# CS 6364.002
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import cv2 as cv
# import random
# import carla

# IM_WIDTH = 640
# IM_HEIGHT = 480

# def process_img(image):
#     i = np.array(image.raw_data)  # convert to an array
#     i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # was flattened, so we're going to shape it.
#     i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
#     cv.imshow("Reinforcment Learning using Carla", i3)  # show it.
#     cv.waitKey(1)
#     return i3/255.0  # normalize

# calra_client = carla.Client("localhost", 2000)
# carla_world =  calra_client.get_world()
# carla_blueprint = carla_world.get_blueprint_library()
# carla_tesla = carla_blueprint.filter("model3")[0]
# spawn_point = random.choice(carla_world.get_map().get_spawn_points())
# vehicle = carla_world.spawn_actor(carla_tesla, spawn_point)

# carla_rgb = carla_blueprint.find('sensor.camera.rgb')
# carla_rgb.set_attribute('image_size_x', f'{IM_WIDTH}')
# carla_rgb.set_attribute('image_size_y', f'{IM_HEIGHT}')
# carla_rgb.set_attribute('fov', '110')

# # Adjust sensor relative to vehicle
# spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

# # spawn the sensor and attach to vehicle.
# sensor = carla_world.spawn_actor(carla_rgb, spawn_point, attach_to=vehicle)
# sensor.listen(lambda data: process_img(data))

import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 640*2
IM_HEIGHT = 480*2

BATCH_SIZE = 128
NUM_ACTIONS = 4
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(10)
    return i3/255.0

actor_list = []
try:
    client = carla.Client(host='127.0.0.1', port=2000)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor
    sensor.listen(lambda data: process_img(data))

    for i in range(0, 10000000000):
        world.tick()

    # time.sleep(10)


finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')