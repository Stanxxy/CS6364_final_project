# import matplotlib.pyplot as plt
import ctypes
from torch.autograd import Variable
from complete_car_environment import CarEnv
from policy_net_agent import PolicyAgent, MODEL_NAME
import random
import glob
import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
import carla
import warnings
import cv2
import multiprocessing
warnings.filterwarnings("ignore")
ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)

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
            try:
                image = self.queue.get(timeout=30)
            except:
                return
            if type(image) is str:
                return
            if len(image) > 0:
                cv2.imshow("Frame", image)
                cv2.waitKey(1)


class Producer(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue

    def run(self):
        MAX_SIZE = 32
        MEMORY_FRACTION = 1.0
        EPISODES = 100

        epsilon = 0.1
        EPSILON_DECAY = 0.95  # 0.9975 99975
        MIN_EPSILON = 0.001

        AGGREGATE_STATS_EVERY = 10
        MIN_REWARD = -200

        FPS = 60
        # For stats
        ep_rewards = [-200]

        # For more repetitive results
        # random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        # Memory fraction, used mostly when training multiple agents
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
        agent = PolicyAgent(device)
        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')
        elif not model_path is None:
            agent.model.load_state_dict(torch.load(model_path))
        # Create agent and environment
        env = CarEnv(self.queue)

        # This time is not possible to use replay method since we are using a policy net
        # trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        # trainer_thread.start()

        agent.init_model()
        # agent.training_initialized = True

        # Initialize predictions - forst prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        # agent.get_action(np.ones((env.im_height, env.im_width, 3)))

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
            agent.rewards = []
            agent.probs = []
            # Play for given number of seconds only
            while len(agent.probs) < MAX_SIZE:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action, ln_prob = agent.get_action(current_state)

                else:
                    # Get random action
                    # modify action to have contineous result
                    action, ln_prob = agent.get_action(
                        current_state, is_random=True)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)

                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update record memory
                agent.update_record_memory(reward, ln_prob)

                agent.train()

                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            env.clear_all()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(
                    ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.writer.add_scalars('reward', {'avg': average_reward,
                                                    'min': min_reward,
                                                    'max': max_reward}, agent.step)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    torch.save(agent.model.state_dict(),
                               f'models/{MODEL_NAME}--{max_reward:_>7.2f}max_{average_reward:_>7.2f}-avg_{min_reward:_>7.2f}-min_{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        agent.terminate = True
        # trainer_thread.join()
        # print("training thread joined")
        torch.save(agent.model.state_dict(
        ), f'models/{MODEL_NAME}--{max_reward:_>7.2f}max_{average_reward:_>7.2f}-avg_{min_reward:_>7.2f}-min_{int(time.time())}.model')
        self.queue.put("end")


if __name__ == "__main__":
    try:
        model_path = sys.argv[1]
    except:
        model_path = None
    queue = multiprocessing.Queue()
    process_producer = Producer(queue)
    process_consumer = Consumer(queue)
    process_producer.start()
    process_consumer.start()
    process_producer.join()
    process_consumer.join()
