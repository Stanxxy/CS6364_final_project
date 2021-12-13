
from collections import deque
import glob
import os
import sys
import time
import random
import torch
from torch import nn

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torchvision.models as models
from basic_car_environment import IM_HEIGHT, IM_WIDTH


carla_abs_path = "/home/stan/Documents/Assighments/CS6364/final_project"
try:
    sys.path.append(glob.glob(os.path.join(carla_abs_path, '/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

REPLAY_MEMORY_SIZE = 4
MODEL_NAME_CRITIC = "XCeption"
MIN_REPLAY_MEMORY_SIZE = 2
MINIBATCH_SIZE = 2
PREDICTION_BATCH_SIZE = 1
# TRAINING_BATCH_SIZE = MINIBATCH_SIZE  # // 4
UPDATE_TARGET_EVERY = 5

EPISODES = 10

DISCOUNT = 0.90


class DQNAgent:

    def __init__(self, device, alpha=0.001, action_state_num=3):
        self.alpha = alpha
        self.device = device
        self.model = self.create_model(action_state_num)
        self.target_model = self.create_model(action_state_num)
        self.target_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.target_model.load_state_dict(self.model.state_dict())
        # self.target_model.set_weights(self.model.get_weights())

        # memory replay
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.writer = SummaryWriter(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # will track when it's time to update the target model
        self.target_update_counter = 0

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.step = 0
        self.training_initialized = False  # waiting for TF to get rolling

    def create_model(self, output_dim):
        model = models.inception_v3(
            pretrained=False, num_classes=output_dim, aux_logits=False)
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        else:
            self.model.train()
            if self.terminate:
                print("terminated")
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
            if self.terminate:
                print("sampled before terminated")
            current_states = torch.FloatTensor(np.array(
                [transition[0] for transition in minibatch])/255)

            current_qs_list = self.model.forward(current_states)

            new_current_states = torch.FloatTensor(np.array(
                [transition[3] for transition in minibatch])/255)

            future_qs_list = self.target_model.forward(new_current_states)
            if self.terminate:
                print("forward to obtain q values before terminated")
            y_pred = current_qs_list.max(1)[0]
            y_target = future_qs_list.max(1)[0]

            for index, (_, _, reward, _, done) in enumerate(minibatch):
                if not done:
                    # max_future_q = future_qs_list[torch.argmax(future_qs_list[index])]
                    y_target[index] = reward + DISCOUNT * y_target[index]
                else:
                    y_target[index] = reward

            log_this_step = False
            # log_this_step = True
            if self.step > self.last_logged_episode:
                log_this_step = True
                self.last_logged_episode = self.step

            if self.terminate:
                print("loss computed before terminated")

            loss = self.criterion(y_pred, y_target)
            l = loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            if self.terminate:
                print("backward done before terminated")

            # Update the parameters
            self.optimizer.step()

            if log_this_step:
                self.writer.add_scalar(
                    "training_loss", l, self.last_logged_episode)
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_update_counter = 0

    def get_qs(self, state):
        self.model.eval()
        return self.model.forward(torch.FloatTensor(np.array(state).reshape(-1, *state.shape)/255)).detach().numpy()

    def train_in_loop(self):
        # first get warm up
        X = torch.FloatTensor(np.random.uniform(
            size=(1, 3, IM_HEIGHT, IM_WIDTH)).astype(np.float32))
        y_target = torch.FloatTensor(
            np.random.uniform(size=(1, 1)).astype(np.float32))

        y_pred = self.model.forward(X)
        # print(y_pred)
        loss = self.criterion(y_pred, y_target.max(1)[0])

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        self.optimizer.step()

        self.training_initialized = True

        while True:
            # print(self.terminate)
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
