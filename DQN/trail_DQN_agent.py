
from collections import deque
import glob
import os
import sys
import time
import random
# import tensorflow as tf
import torch
from torch import nn

import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from myTensorboard import ModifiedTensorBoard

# from keras.applications.xception import Xception
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.optimizers import Adam
from torch.optim import Adam
# from keras.models import Model
import torchvision.models as models
from trail_car_environment import IM_HEIGHT, IM_WIDTH

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)


carla_abs_path = "/home/stan/Documents/Assighments/CS6364/final_project"
try:
    sys.path.append(glob.glob(os.path.join(carla_abs_path, '/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

REPLAY_MEMORY_SIZE = 4
MODEL_NAME = "XCeption"
MIN_REPLAY_MEMORY_SIZE = 2
MINIBATCH_SIZE = 2
PREDICTION_BATCH_SIZE = 1
# TRAINING_BATCH_SIZE = MINIBATCH_SIZE  # // 4
UPDATE_TARGET_EVERY = 5

EPISODES = 100

DISCOUNT = 0.90


class DQNAgent:

    def __init__(self, alpha=0.001, action_state_num=3):
        self.alpha = alpha
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
        # when do image processing do remember to modify image to 3 * H * W not H * W * 3
        # Xception(weights=None, include_top=False,
        #   input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        # x = base_model.output
        # x = GlobalAveragePooling2D()(x)

        # predictions = Dense(3, activation="linear")(x)
        # model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        else:
            self.model.train()
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
            current_states = torch.FloatTensor(np.array(
                [transition[0] for transition in minibatch])/255)  # .to('cuda')

            current_qs_list = self.model.forward(current_states)

            new_current_states = torch.FloatTensor(np.array(
                [transition[3] for transition in minibatch])/255)  # .to('cuda')

            future_qs_list = self.target_model.forward(new_current_states)

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

            loss = self.criterion(y_pred, y_target)
            l = loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

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

        # with self.graph.as_default():
        #     self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
