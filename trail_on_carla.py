# import matplotlib.pyplot as plt
import carla
import time
import random
import glob
import os
import sys
import numpy as np
import cv2
import subprocess
import multiprocessing
# my_queue = Manager().Queue()


carla_abs_path = "/home/stan/Documents/Assighments/CS6364/final_project"
try:
    sys.path.append(glob.glob(os.path.join(carla_abs_path, '/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

IM_WEIGHT = 640
IM_HEIGHT = 480
SCAN_FREQ = 0.1


class Consumer(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            image = self.queue.get()
            if type(image) is str:
                break
            if not image is None and len(image) > 0:
                cv2.imshow("Frame", image)
                cv2.waitKey(1)


class Producer(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):
        actor_list = []

    # display = pygame.display.set_mode(
    #     (IM_WEIGHT, IM_HEIGHT),
    #     pygame.HWSURFACE | pygame.DOUBLEBUF)

        try:
            # p = subprocess.Popen(["python3.6", "imshow.py"])
            client = carla.Client('localhost', 2000)
            client.set_timeout(1.0)

            world = client.get_world()
            blueprint_library = world.get_blueprint_library()

            bp_model3 = blueprint_library.filter('model3')[0]

            spawn_point = random.choice(world.get_map().get_spawn_points())

            vehicle = world.spawn_actor(bp_model3, spawn_point)

            # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            vehicle.set_autopilot(True)

            actor_list.append(vehicle)

            bp_cam = blueprint_library.find("sensor.camera.rgb")
            bp_cam.set_attribute("image_size_x", f"{IM_WEIGHT}")
            bp_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
            bp_cam.set_attribute("fov", "110")
            # bp_cam.set_attribute("sensor_tick", f"{SCAN_FREQ}")
            bp_transform = carla.Transform(carla.Location(
                x=2.5, z=1.7))  # relative location to the car
            camera = world.spawn_actor(bp_cam, bp_transform, attach_to=vehicle)

            camera.listen(lambda data: self.view_img(data))
            # camera.listen(lambda image: image.save_to_disk(
            #     'output/%06d.png' % image.frame))

            time.sleep(10)

        finally:
            print("destroying actors")
            camera.stop()
            # p.kill()
            # subprocess.run("kill -s 9 " + + "", shell=True)
            camera.destroy()
            client.apply_batch([carla.command.DestroyActor(x)
                               for x in actor_list])
            self.queue.put("end")
            print("done.")

    def view_img(self, img):

        i = np.array(img.raw_data)
        img_0 = i.reshape((IM_HEIGHT, IM_WEIGHT, 4))  # rgb a
        img_1 = img_0[:, :, :3]
        if self.queue.qsize() <= 1024:
            self.queue.put(img_1)
        # cv2.imwrite("image.pgm", cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY))
        # plt.show()
        # cv2.waitKey(100)
        return img_1 / 255.0


if __name__ == "__main__":
    queue = multiprocessing.Queue()
    process_producer = Producer(queue)
    process_consumer = Consumer(queue)
    process_producer.start()
    process_consumer.start()
    process_producer.join()
    process_consumer.join()
