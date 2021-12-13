# import matplotlib.pyplot as plt
import ctypes
from trail_car_environment import CarEnv
from trail_DQN_agent import DQNAgent, MODEL_NAME
from threading import Thread
# import keras.backend.tensorflow_backend as backend
# import tensorflow as tf
import torch
import random
import glob
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import carla
import warnings
import cv2
import multiprocessing
warnings.filterwarnings("ignore")
# ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)

carla_abs_path = "/home/stan/Documents/Assighments/CS6364/final_project"
try:
    sys.path.append(glob.glob(os.path.join(carla_abs_path, '/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass


# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

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
        MEMORY_FRACTION = 0.3
        EPISODES = 100

        epsilon = 1.0
        EPSILON_DECAY = 0.95  # 0.9975 99975
        MIN_EPSILON = 0.001

        AGGREGATE_STATS_EVERY = 10
        MIN_REWARD = -200

        FPS = 60
        # For stats
        ep_rewards = [-200]

        # For more repetitive results
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        # Memory fraction, used mostly when training multiple agents
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
        agent = DQNAgent()
        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')
        else:
            agent.model.load_state_dict()
        # Create agent and environment
        env = CarEnv(self.queue)

        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        # Initialize predictions - forst prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        agent.get_qs(np.ones((3, env.im_height, env.im_width)))

        # Iterate over episodes
        for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
            # try:
            env.collision_hist = []

            # Update tensorboard step every episode
            agent.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()

            # Play for given number of seconds only
            while True:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, 3)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)

                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory(
                    (current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            for sensor in env.sensor_list:
                sensor.stop()
            env.client.apply_batch([carla.command.DestroyActor(x)
                                    for x in env.actor_list])

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(
                    ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.writer.add_scalars('reward', {'reward_avg': average_reward,
                                                    'reward_min': min_reward,
                                                    'reward_max': max_reward}, agent.step)
                # agent.writer.add_scalars(
                #     'epsilon', torch.IntTensor([epsilon]), agent.step)
                # agent.tensorboard.update_stats(
                #     reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    torch.save(agent.model.state_dict(),
                               f'models/{MODEL_NAME}--{max_reward:_>7.2f}max_\
                                   {average_reward:_>7.2f}-avg_{min_reward:_>7.2f}-min_{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        # Set termination flag for training thread and wait for it to finish
        print("Sent flag")
        agent.terminate = True
        trainer_thread.join()
        torch.save(agent.model.state_dict(), f'models/{MODEL_NAME}--{max_reward:_>7.2f}max_\
            {average_reward:_>7.2f}-avg_{min_reward:_>7.2f}-min_{int(time.time())}.model')


if __name__ == "__main__":
    queue = multiprocessing.Queue()
    process_producer = Producer(queue)
    process_consumer = Consumer(queue)
    process_producer.start()
    process_consumer.start()
    process_producer.join()
    process_consumer.join()
