# import matplotlib.pyplot as plt
import queue
import carla
import time
import random
import glob
import os
import sys
import cv2 as cv
import numpy as np
# import subprocess

# carla_abs_path = "/home/stan/Documents/Assighments/CS6364/final_project"
# try:
#     sys.path.append(glob.glob(os.path.join(carla_abs_path, '/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
# except IndexError:
#     pass

IM_WIDTH = 600
IM_HEIGHT = 400
SCAN_FREQ = 0.0

SHOW_PREVIEW = True
SECONDS_PER_EPISODE = 100

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

    def __init__(self, queue=None):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        self.frame = None

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

        # if self.SHOW_CAM:
        # TODO : add a command line parameter so we could start different view script for different agent
        # self.p = subprocess.Popen(["python3.6", "imshow.py"])

    def reset(self):
        self.vehicle = None
        self.collision_hist = []
        self.lane_cross_hist = []
        self.actor_list = []
        self.sensor_list = []  # use a sensor slist to stop before the object is recycled
        self._queues = []

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1/20.0))

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(
                    self.model_3, self.transform)
            except:
                pass

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=1.2))

        self.sensor = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        self.sensor_list.append(self.sensor)

        # self.sensor.listen(lambda data: self.process_img(data))
        # initially passing some commands seems to help with time. Not sure why.
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))

        # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        # while self.front_camera is None:
        #     time.sleep(0.01)
        time.sleep(4)

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.sensor_list.append(self.colsensor)
        # self.colsensor.listen(lambda event: self.collision_data(event))

        lane_cross_sensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_cross_sensor = self.world.spawn_actor(
            lane_cross_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lane_cross_sensor)
        self.sensor_list.append(self.lane_cross_sensor)
        # self.lane_cross_sensor.listen(lambda event: self.lane_cross_data(event))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensor_list:
            make_queue(sensor.listen)

        # while self.front_camera is None:
        #     time.sleep(0.01)


        self.episode_start = time.time()

        self.vehicle.apply_control(
            carla.VehicleControl(brake=0.0, throttle=0.0))

        # return self.front_camera

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x is None or x.frame == self.frame for x in data)
        return tuple(data)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            try:
                data = sensor_queue.get(timeout=timeout)
                if data.frame == self.frame:
                    return data
            except:
                return None


    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_cross_data(self, event):
        lane_type = str(event.crossed_lane_markings[0].type)
        if "Solid" in lane_type or "NONE" in lane_type or "Grass" in lane_type or "Curb" in lane_type or "Other" in lane_type:
            self.lane_cross_hist.append(event)

    def is_invalid(self, event):
        lane_type = str(event.crossed_lane_markings[0].type)
        return "Solid" in lane_type or "NONE" in lane_type or "Grass" in lane_type or "Curb" in lane_type or "Other" in lane_type

    def process_img(self, image):
        if image is None:
            return None
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        # cv.imshow("Carla Env", self.process_image(i3, "semantic_segmentation"))
        # cv.waitKey(1)
        return i3

    def process_image(self, image, type="rgb"):
        frame = image.copy()
        if type == "rgb":
            pass
        elif type == "semantic_segmentation":
            frame[np.where((frame == [0, 0, 1]).all(axis=2))] = [70, 70, 70]
            frame[np.where((frame == [0, 0, 2]).all(axis=2))] = [153, 153, 190]
            frame[np.where((frame == [0, 0, 3]).all(axis=2))] = [160, 170, 250]
            frame[np.where((frame == [0, 0, 4]).all(axis=2))] = [60, 20, 220]
            frame[np.where((frame == [0, 0, 5]).all(axis=2))] = [153, 153, 153]
            frame[np.where((frame == [0, 0, 6]).all(axis=2))] = [50, 234, 157]
            frame[np.where((frame == [0, 0, 7]).all(axis=2))] = [128, 64, 128]
            frame[np.where((frame == [0, 0, 8]).all(axis=2))] = [232, 35, 244]
            frame[np.where((frame == [0, 0, 9]).all(axis=2))] = [35, 142, 107]
            frame[np.where((frame == [0, 0, 10]).all(axis=2))] = [142, 0, 0]
            frame[np.where((frame == [0, 0, 11]).all(axis=2))] = [156, 102, 102]
            frame[np.where((frame == [0, 0, 12]).all(axis=2))] = [0, 220, 220]

        return frame

    def get_screen(self):
        pass

    def step(self, action):
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

        world, next_state, obj_col, lane_col = self.tick(1.0)

        if obj_col is not None:
            done = True
            reward = -200
        elif lane_col is not None and self.is_invalid(lane_col):
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.process_img(next_state), reward, done, None  # state, reward, done, other_info

    def __del__(self):
        print("destroying actors")

        # if self.SHOW_CAM:
        #     self.p.kill()

        print("done.")
