import ctypes
from threading import Thread
from car_environment import CarEnv
from critic_agent import DQNAgent, MODEL_NAME_CRITIC
from actor_agent import PolicyAgent, MODEL_NAME_ACTOR
from collections import deque
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

    REPLAY_MEMORY_SIZE = 16
    MAX_ROUND = 128  # max duration for running in each episode
    MEMORY_FRACTION = 1.0
    EPISODES = 100

    epsilon = 0.25
    EPSILON_DECAY = 0.95  # 0.9975 99975
    MIN_EPSILON = 0.001

    AGGREGATE_STATS_EVERY = 10
    MIN_REWARD = -200

    FPS = 60

    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def run(self):

        # For stats
        ep_rewards = [-200]

        # For more repetitive results
        # random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        # Memory fraction, used mostly when training multiple agents
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_per_process_memory_fraction(self.MEMORY_FRACTION, 0)
        actor_agent = PolicyAgent(device, self.replay_memory)
        critic_agent = DQNAgent(device, self.replay_memory)
        actor_agent.set_critic(critic_agent)
        critic_agent.set_actor_net(actor_agent)
        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')
        else:
            if not actor_model_path is None:
                actor_agent.model.load_state_dict(
                    torch.load(actor_model_path))
            if not critic_model_path is None:
                critic_agent.model.load_state_dict(
                    torch.load(critic_model_path))
        # Create agent and environment
        env = CarEnv(self.queue)

        # This time is not possible to use replay method since we are using a policy net
        critic_trainer_thread = Thread(
            target=critic_agent.train_in_loop, daemon=True)
        actor_trainer_thread = Thread(
            target=actor_agent.train_in_loop, daemon=True)

        # while not critic_agent.training_initialized:
        #     time.sleep(0.01)

        critic_trainer_thread.start()
        actor_trainer_thread.start()

        # actor_agent.init_model()

        # Initialize predictions - forst prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        critic_agent.get_q(
            np.ones((3, env.im_height, env.im_width)), np.random.rand(2))

        # Iterate over episodes
        for episode in tqdm(range(1, self.EPISODES + 1), unit='episodes'):
            # try:
            env.collision_hist = []

            # Update tensorboard step every episode
            actor_agent.step = critic_agent.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            # actor_agent.reward = None
            # actor_agent.prob = None
            # Play for given number of seconds only
            while True:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action, ln_prob = actor_agent.get_action(
                        current_state, if_predict=True)
                else:
                    # Get random action
                    # modify action to have contineous result
                    action, ln_prob = actor_agent.get_action(
                        current_state, is_random=True)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/self.FPS)

                new_state, reward, done, _ = env.step(action)

                self.update_replay_memory(
                    (current_state, action, reward, new_state, done))
                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update record memory
                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            env.clear_all()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % self.AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(
                    ep_rewards[-self.AGGREGATE_STATS_EVERY:])/len(ep_rewards[-self.AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-self.AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-self.AGGREGATE_STATS_EVERY:])
                actor_agent.writer.add_scalars('reward', {'avg': average_reward,
                                                          'min': min_reward,
                                                          'max': max_reward}, actor_agent.step)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= self.MIN_REWARD:
                    torch.save(actor_agent.model.state_dict(),
                               f'models/{MODEL_NAME_ACTOR}:{MODEL_NAME_CRITIC}--{max_reward:_>7.2f}' +
                               f'max_{average_reward:_>7.2f}-avg_{min_reward:_>7.2f}-min_{int(time.time())}.model')

            # Decay epsilon
            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon = max(self.MIN_EPSILON, self.epsilon)

        actor_agent.terminate = True
        # trainer_thread.join()
        # print("training thread joined")
        torch.save(actor_agent.model.state_dict(),
                   f'models/{MODEL_NAME_ACTOR}:{MODEL_NAME_CRITIC}--{max_reward:_>7.2f}' +
                   f'max_{average_reward:_>7.2f}-avg_{min_reward:_>7.2f}-min_{int(time.time())}.model')
        self.queue.put("end")


if __name__ == "__main__":
    try:
        actor_model_path = sys.argv[1]
    except:
        actor_model_path = None
    try:
        critic_model_path = sys.argv[1]
    except:
        critic_model_path = None

    queue = multiprocessing.Queue()
    process_producer = Producer(queue)
    process_consumer = Consumer(queue)
    process_producer.start()
    process_consumer.start()
    process_producer.join()
    process_consumer.join()
