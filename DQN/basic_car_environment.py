# import matplotlib.pyplot as plt
import queue
import carla
import time
import random
import glob
import os
import sys
import numpy as np
# import subprocess

carla_abs_path = "/home/stan/Documents/Assighments/CS6364/final_project"
try:
    sys.path.append(glob.glob(os.path.join(carla_abs_path, '/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

IM_WIDTH = 600
IM_HEIGHT = 400
SCAN_FREQ = 0.0

SHOW_PREVIEW = True
SECONDS_PER_EPISODE = 60

LOW_SPEED_BOUNDARY = 50
LOW_SPEED_PUNISHMENT = -5


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0  # we could change the steer as a conntineous value

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    actor_list = []

    front_camera = None
    collision_hist = []

    def __init__(self, queue):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        self.blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        # print(blueprint_library.filter('vehicle'))
        self.model_3 = self.blueprint_library.filter('model3')[0]
        self.queue = queue

    def reset(self):
        self.vehicle = None
        self.collision_hist = []
        self.actor_list = []
        self.sensor_list = []  # use a sensor slist to stop before the object is recycled

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(
                    self.model_3, self.transform)
            except:
                pass

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.sensor = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        self.sensor_list.append(self.sensor)

        self.sensor.listen(lambda data: self.process_img(data))
        # initially passing some commands seems to help with time. Not sure why.
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))

        # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        # while self.front_camera is None:
        time.sleep(2)

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.sensor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        self.lane_cross_hist = []

        lane_cross_sensor = self.world.get_blueprint_library().find(
            'sensor.other.lane_invasion')
        self.lane_cross_sensor = self.world.spawn_actor(
            lane_cross_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lane_cross_sensor)
        self.sensor_list.append(self.lane_cross_sensor)
        self.lane_cross_sensor.listen(
            lambda event: self.lane_cross_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(
            carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_cross_data(self, event):
        self.lane_cross_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM and self.queue.qsize() <= 1024:
            self.queue.put(i3)
            # cv2.imshow("",i3)
            # cv2.waitKey(1)
        self.front_camera = i3.reshape(3, self.im_height, self.im_width)

    def step(self, action):
        # print("steped. action {}".format(action))
        # TODO: have more actions. Control throttle, steer and break saperatly.
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''
        if action == 0:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=-1*self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -1000

        if len(self.lane_cross_hist) != 0:
            # We need to punish but there's no need to stop the game
            done = True if len(self.collision_hist) > 0 else False
            reward = -200

        elif kmh < 50:
            done = False
            reward = - (kmh / 10)

        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            print("time is over")
            done = True

        return self.front_camera, reward, done, None  # state, reward, done, other_info

    def __del__(self):
        print("destroying actors")
        print("done.")
