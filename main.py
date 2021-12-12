# Class Project
# Travis Bonneau, Sennan Liu
# CS 6364.002
import time
import carla
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN, ReplayMemory, Transition
import math
import random
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from trail_car_environment import CarEnv
import cv2 as cv
import numpy as np
from multiprocessing import Queue
from itertools import count

print("Torch Version:", torch.__version__)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 50
NUM_EPISODES = 10
MAX_TIME_STEP = 10000

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

queue = Queue()
env = CarEnv()

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape
screen_height, screen_width = env.im_height, env.im_width


# Get number of actions from gym action space
# n_actions = env.action_space.n
n_actions = 3

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-5)
memory = ReplayMemory(10000)


steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            output = policy_net(torch.tensor(state, dtype=torch.float).to(device))
            return output.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []
def plot_durations():
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.figure()
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(np.arange(len(episode_durations)), episode_durations, color="blue", marker="o")

    # plt.pause(0.01)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)


    # print("State Batch:", state_batch)
    # print("State Batch Device:", state_batch.device)
    # print("Action Batch:", action_batch)
    # print("Action Batch Device:", action_batch.device)
    # print("Reward Batch:", reward_batch)
    # print("Reward Batch Device:", reward_batch.device)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_net_output = policy_net(state_batch)
    state_action_values = policy_net_output.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = next_state_values.reshape((-1, 1))
    # Compute the expected Q values
    # print("Next State Values Shape:", next_state_values.shape)
    # print("Rewards Batch Shape:", reward_batch.shape)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # print(state_action_values.shape)
    # print(expected_state_action_values.shape)
    # print(expected_state_action_values.unsqueeze(1).shape)
    loss = criterion(state_action_values, expected_state_action_values)

    # print(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def process_image(image, type="rgb"):
    frame = image
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


episode_durations = []
for episode in range(1, NUM_EPISODES + 1):
    # print("="*100)
    # Initialize the environment and state
    start_time = time.time()
    curr_state = env.reset()
    time.sleep(1)

    for t in range(MAX_TIME_STEP):
        curr_state = curr_state.reshape((1, 3, 400, 600))
        action_tensor = select_action(curr_state)
        action = action_tensor.item()
        next_state, reward, done, _ = env.step(action)

        # Convert step output to tensors
        curr_state_tensor = torch.tensor(curr_state.reshape((1, 3, 400, 600)), dtype=torch.float)
        next_state_tensor = torch.tensor(next_state.reshape((1, 3, 400, 600)), dtype=torch.float)
        action_tensor = action_tensor.reshape((1, -1))
        reward_tensor = torch.tensor([reward], dtype=torch.float).reshape((1, -1))
        # Save state data to memory
        memory.push(curr_state_tensor, action_tensor, next_state_tensor, reward_tensor)

        # Update State
        curr_state = next_state

        # cv.imshow("Image", process_image(curr_state, "semantic_segmentation"))
        # cv.waitKey(1)

        optimize_model()
        if t + 1 == MAX_TIME_STEP:
            done = True
        if done:
            stop_time = time.time()
            delta_time = stop_time - start_time
            print(f"Episode: {episode}, Durations: {t+1:3d}, Time: {delta_time:.3f} secs", flush=True)
            episode_durations.append(t + 1)
            break

    # Destroy an actor at end of episode
    for sensor in env.sensor_list:
        sensor.stop()
    env.client.apply_batch([carla.command.DestroyActor(x)
                            for x in env.actor_list])
    # Update the target network, copying all weights and biases in DQN
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plot_durations()
plt.show()

















#     last_screen = env.get_screen()
#     current_screen = get_screen()
#     state = current_screen - last_screen
#     for t in count():
#         # Select and perform an action
#         action = select_action(state)
#         _, reward, done, _ = env.step(action.item())
#         reward = torch.tensor([reward], device=device)

#         # Observe new state
#         last_screen = current_screen
#         current_screen = get_screen()
#         if not done:
#             next_state = current_screen - last_screen
#         else:
#             next_state = None

#         # Store the transition in memory
#         memory.push(state, action, next_state, reward)

#         # Move to the next state
#         state = next_state

#         # Perform one step of the optimization (on the policy network)
#         optimize_model()
#         if done:
#             episode_durations.append(t + 1)
#             plot_durations()
#             break
#     # Update the target network, copying all weights and biases in DQN
#     if i_episode % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

# print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show()



























###################################################################################
###################################################################################
###################################################################################
# NUM_ROWS = 720
# NUM_COLS = 1280
# NUM_ACTIONS = 4
# NUM_EPISODES = 100
# MAX_TIME_STEPS = 1000
# BATCH_SIZE = 128
# EPSILON = 1.0
# EPSILON_END = 0.05
# EPSILON_DECAY = 0.0001
# UPDATE_RATE = 10
# UPDATE_RATE = 10

# def copy_state(policy_net, target_net):
#     target_net.load_state_dict(policy_net.state_dict())

# def get_action(state, network):
#     rand = random.random()
#     if rand > EPSILON:
#         with torch.no_grad():
#             action = network(state).max(1)
#     else:
#         action = torch.tensor([random.randrange(NUM_ACTIONS)])

#     print(action)
#     EPSILON = EPSILON - EPSILON_DECAY
#     return action

# def train_model():
#     pass

# # Define networks, criterion and optimizer
# policy_net = DeepQNetwork(rows=NUM_ROWS, cols=NUM_COLS, outputs=NUM_ACTIONS)
# target_net = DeepQNetwork(rows=NUM_ROWS, cols=NUM_COLS, outputs=NUM_ACTIONS)
# criterion = nn.SmoothL1Loss()
# optimizer = optim.Adam(policy_net.parameters())

# # Target net does not need to calculate gradients, so we set it in eval mode
# copy_state(policy_net, target_net)
# target_net.eval()

# replay_memory = ReplayMemory(10000)



# # Train network
# duration_hist = []
# for episode in range(1, NUM_EPISODES+1):
#     last_img = carla_env.get_frame()
#     curr_img = carla_env.get_frame()
#     for time in range(MAX_TIME_STEPS):
#         curr_action = get_action(state, policy_net)

#         reward, is_done = carla_env.step(curr_action)


#         last_img = curr_img
#         curr_img = carla_env.get_frame()
#         if not is_done:
#             next_state = curr_img - last_img
#         else:
#             next_state = None

#         replay_memory.push((state, action, next_state, reward))
#         state = next_state

#         train_model()
#         if is_done:
#             duration_hist.append(time + 1)
#             break

#     if episode % UPDATE_RATE == 0:
#         copy_state(policy_net, target_net)