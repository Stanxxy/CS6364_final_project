
from collections import deque
import glob
import os
import sys
import time
import random
import torch
from torch import nn
import numpy as np
from torch.nn.modules.linear import Linear
from torch.distributions import Normal
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torchvision.models as models
from complete_car_environment import IM_HEIGHT, IM_WIDTH

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

REPLAY_MEMORY_SIZE = 1
MODEL_NAME = "XCeption"
MIN_HISTORY_SIZE = 10
MINIBATCH_SIZE = 4
PREDICTION_BATCH_SIZE = 1
# TRAINING_BATCH_SIZE = MINIBATCH_SIZE  # // 4
UPDATE_TARGET_EVERY = 5
# LOG_SIG_MIN = -2
# LOG_SIG_MAX = -20
DISCOUNT = 0.90


class PolicyNet(models.Inception3):
    def __init__(self, output_dim=1000, aux_logits=False):
        super(PolicyNet, self).__init__(
            num_classes=output_dim, aux_logits=aux_logits)

        self.gaussian_mean = Linear(self.fc.out_features, output_dim)
        self.log_std = Linear(self.fc.out_features, output_dim)

    def forward(self, x):
        hidden_x = super().forward(x)
        mean = self.gaussian_mean(hidden_x)
        log_std = self.log_std(hidden_x)
        std = log_std.exp()
        return mean, std


class PolicyAgent:

    def __init__(self, device, alpha=0.001, action_state_num=2):
        self.alpha = alpha
        self.device = device
        self.model = self.create_model(action_state_num)  # .to(self.device)
        # self.target_model = self.create_model(action_state_num)
        # self.target_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        # self.target_model.load_state_dict(self.model.state_dict())

        # memory replay
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.writer = SummaryWriter(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # will track when it's time to update the target model
        self.target_update_counter = 0
        self.probs, self.rewards = [], []  # Variable(torch.Tensor()), []

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.step = 0
        self.training_initialized = False  # waiting for TF to get rolling

    def create_model(self, output_dim):
        model = PolicyNet(output_dim=output_dim)
        return model

    def update_record_memory(self, reward, prob):
        self.probs.append(prob)
        self.rewards.append(reward)

    def train(self):
        if len(self.probs) < MIN_HISTORY_SIZE:
            return
        else:
            policy_loss = None

            R = 0
            returns = []
            for r in self.rewards[::-1]:
                R = r + DISCOUNT * R
                returns.insert(0, R)

            returns = torch.FloatTensor(returns)
            # print(returns)
            policy_loss = []
            for log_prob, R in zip(self.probs, returns):
                policy_loss.append(- log_prob * R)

            # print(policy_loss)
            policy_loss = torch.stack(policy_loss).sum()  # .to(self.device)
            # self.model .to(self.device)
            # policy_loss = (torch.sum(torch.mul(self.probs,
            #                                    Variable(returns)).mul(-1), -1))
            # update policy weights
            self.optimizer.zero_grad()
            policy_loss.backward()
            print("backward called ")
            self.optimizer.step()

            # log_this_step = False
            if self.step > self.last_logged_episode:
                # log_this_step = True
                self.last_log_episode = self.step

            self.rewards = []
            self.agent.probs = []
            # if log_this_step:
            #     self.target_update_counter += 1

            # if self.target_update_counter > UPDATE_TARGET_EVERY:
            #     self.target_model.set_weights(self.model.get_weights())
            #     self.target_update_counter = 0

    def get_action(self, state, is_random=False):
        if is_random:
            # print("random gen")
            mean = torch.randn(2).float().unsqueeze(0)
            log_std = torch.randn(2).float().unsqueeze(0)
            # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = log_std.exp()
        else:
            # print("normal gen")
            mean, std = self.model(Variable(torch.FloatTensor(
                np.array(state).reshape(-1, *state.shape)/255)))  # .to(self.device))

        # print(mean, std)
        normal = Normal(mean, std)
        # the contineous action value for throttle and steer
        c = normal.sample()
        ln_prob = normal.log_prob(c)

        sum_prob = ln_prob.sum()
        # squeeze action into [-1,1]
        action = torch.tanh(c)

        action = action.cpu().numpy()
        return action[0], sum_prob

    def init_model(self):
        # first get warm up
        X = torch.FloatTensor(np.random.uniform(
            size=(1, 3, IM_HEIGHT, IM_WIDTH)).astype(np.float32))  # .to(self.device)
        y_target = torch.FloatTensor(
            np.random.uniform(size=(1, 1)).astype(np.float32))  # .to(self.device)

        pred_mean, pred_std = self.model(X)

        normal = Normal(pred_mean, pred_std)
        # the contineous action value for throttle and steer
        action = normal.sample()
        ln_prob = normal.log_prob(action)

        ln_prob = ln_prob.sum()
        # squeeze action into [-1,1]
        policy_loss = - ln_prob * y_target[0]

        self.optimizer.zero_grad()
        policy_loss.backward()
        # print("backward called ")
        self.optimizer.step()
        # return action, ln_prob

        self.training_initialized = True

        while not self.training_initialized:
            time.sleep(0.01)
        # while True:
        #     if self.terminate:
        #         return
        #     self.train()
        #     time.sleep(0.01)
