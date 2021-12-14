import glob
import os
import sys
import time
import random
import torch
from torch import nn
from torch.nn.modules.linear import Linear
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torchvision.models as models
from car_environment import IM_HEIGHT, IM_WIDTH


carla_abs_path = "/home/stan/Documents/Assighments/CS6364/final_project"
try:
    sys.path.append(glob.glob(os.path.join(carla_abs_path, '/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass


MODEL_NAME_CRITIC = "XCeption"
MIN_REPLAY_MEMORY_SIZE = 8
MINIBATCH_SIZE = 2
PREDICTION_BATCH_SIZE = 1
# TRAINING_BATCH_SIZE = MINIBATCH_SIZE  # // 4
UPDATE_TARGET_EVERY = 1

EPISODES = 10

DISCOUNT = 0.90


class DQNNet(models.Inception3):
    def __init__(self, action_dim=1000, aux_logits=False):
        super(DQNNet, self).__init__(
            num_classes=action_dim, aux_logits=aux_logits)

        self.action_encoder = Linear(action_dim, self.fc.out_features)
        self.relu = torch.nn.ReLU()
        self.action_encoder2 = Linear(self.fc.out_features, 256)
        self.action_encoder3 = Linear(256, 128)
        self.reward_dense = Linear(128, 1)

    def forward(self, x, action):
        hidden_x = super().forward(x)
        encoded_action = self.relu(self.action_encoder(action))
        encoded_action = self.relu(
            self.action_encoder2(encoded_action + hidden_x))
        encoded_action = self.relu(self.action_encoder3(encoded_action))
        reward = self.reward_dense(encoded_action)
        return reward


class DQNAgent:

    def __init__(self, device, replay_q, alpha=0.001, action_state_num=2):
        # 1 is the reward value
        self.alpha = alpha
        self.device = device
        self.model = self.create_model(action_state_num)
        self.target_model = self.create_model(action_state_num)
        self.target_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.target_model.load_state_dict(self.model.state_dict())

        # memory replay
        self.replay_memory = replay_q
        self.writer = SummaryWriter(
            log_dir=f"logs/{MODEL_NAME_CRITIC}-{int(time.time())}")

        # will track when it's time to update the target model
        self.target_update_counter = 0

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.step = 0
        self.training_initialized = False  # waiting for TF to get rolling

    def create_model(self, action_dim):
        model = DQNNet(action_dim=action_dim, aux_logits=False)
        return model

    def set_actor_net(self, actor_net):
        self.actor_net = actor_net

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        else:
            self.model.train()
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

            current_states = torch.FloatTensor(np.array(
                [transition[0] for transition in minibatch])/255)

            actions = torch.FloatTensor(np.array(
                [transition[1] for transition in minibatch])/255)

            current_qs_list = self.model.forward(current_states, actions)

            new_current_states = torch.FloatTensor(np.array(
                [transition[3] for transition in minibatch])/255)

            new_actions = torch.FloatTensor(np.array([
                self.actor_net.get_action(transition[3], if_predict=True)[0] for transition in minibatch]))

            future_qs_list = self.target_model.forward(
                new_current_states, new_actions)

            y_pred = current_qs_list
            y_target = future_qs_list

            for index, (_, _, reward, _, done) in enumerate(minibatch):
                if not done:
                    y_target[index] = reward + DISCOUNT * y_target[index]
                else:
                    y_target[index] = reward

            log_this_step = False
            if self.step > self.last_logged_episode:
                log_this_step = True
                self.last_logged_episode = self.step

            loss = self.criterion(y_pred, y_target)
            l = loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()
            print("critic backward called")
            # Update the parameters
            self.optimizer.step()

            if log_this_step:
                self.writer.add_scalar(
                    "critic_training_loss", l, self.last_logged_episode)
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_update_counter = 0

    def get_q(self, state, action):

        self.model.eval()
        with torch.no_grad():
            res = self.model.forward(
                torch.FloatTensor(
                    np.array(state).reshape(-1, *state.shape)/255),
                torch.FloatTensor(np.array(action).reshape(-1, *action.shape))).detach().numpy()
            print("reward: {}".format(res[0][0][0]))
            return res

    def train_in_loop(self):
        # first get warm up
        X = torch.FloatTensor(np.random.uniform(
            size=(1, 3, IM_HEIGHT, IM_WIDTH)).astype(np.float32))
        X_action = torch.FloatTensor(np.random.uniform(
            size=(1, 2)).astype(np.float32))
        y_target = torch.FloatTensor(
            np.random.uniform(size=(1, 1)).astype(np.float32))

        y_pred = self.model.forward(X, X_action)
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
