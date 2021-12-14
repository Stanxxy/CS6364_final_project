import random
import glob
import os
import sys
import time
import torch
from torch import nn
import numpy as np
from torch.nn.modules.linear import Linear
from torch.distributions import Normal
from torch.autograd import Variable
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

MODEL_NAME_ACTOR = "XCeption"
MIN_REPLAY_MEMORY_SIZE = 8
MINIBATCH_SIZE = 4
PREDICTION_BATCH_SIZE = 1
LOG_SIG_MIN = -20
LOG_SIG_MAX = -2
UPDATE_TARGET_EVERY = 5
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
        log_std = torch.clamp(
            log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()
        return mean, std


class PolicyAgent:

    def __init__(self, device, replay_q, alpha=0.001, action_state_num=2):
        self.alpha = alpha
        self.device = device
        self.model = self.create_model(action_state_num)  # .to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

        # memory replay
        self.replay_memory = replay_q
        self.writer = SummaryWriter(
            log_dir=f"logs/{MODEL_NAME_ACTOR}-{int(time.time())}")

        # will track when it's time to update the target model
        self.target_update_counter = 0
        self.prob, self.reward = None, None

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.step = 0
        self.training_initialized = False

    def create_model(self, output_dim):
        model = PolicyNet(output_dim=output_dim)
        return model

    def set_critic(self, critic_net):
        self.critic_net = critic_net

    def train(self):
        self.model.train()
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        for transition in minibatch:
            current_state = transition[0]
            action, self.prob = self.model.forward(
                torch.FloatTensor(current_state / 255).unsqueeze(0))
            self.reward = self.critic_net.get_q(
                current_state, action)
            if self.prob is None or self.reward is None:
                print(self.prob, self.reward)
            policy_loss = - \
                torch.mul(self.prob, torch.FloatTensor(self.reward))

            policy_loss = policy_loss.sum()
            l = policy_loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            policy_loss.backward()
            print("actor backward called ")
            self.optimizer.step()

        log_this_step = False
        if self.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.step

        if log_this_step:
            self.writer.add_scalar(
                "critic_training_loss", l, self.last_logged_episode)
        #     self.target_update_counter += 1

        # if self.target_update_counter > UPDATE_TARGET_EVERY:
        #     self.target_model.set_weights(self.model.get_weights())
        #     self.target_update_counter = 0

    def get_action(self, state, is_random=False, if_predict=False):
        if if_predict:
            # print("do predict")
            self.model.eval()
            with torch.no_grad():
                mean, std = self.model(Variable(torch.FloatTensor(
                    np.array(state).reshape(-1, *state.shape)/255)))
        else:
            if is_random:
                # print("random gen")
                mean = torch.randn(2).float().unsqueeze(0)
                log_std = torch.randn(2).float().unsqueeze(0)
                log_std = torch.clamp(
                    log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                std = log_std.exp()
            else:
                # print("normal gen")
                mean, std = self.model(Variable(torch.FloatTensor(
                    np.array(state).reshape(-1, *state.shape)/255)))
        # print(mean, std)
        normal = Normal(mean, std)
        # the contineous action value for throttle and steer
        c = normal.sample()
        ln_prob = normal.log_prob(c)

        sum_prob = ln_prob.sum()

        action = torch.tanh(c)

        action = action.cpu().numpy()
        return action[0], sum_prob

    def train_in_loop(self):
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

        policy_loss = - ln_prob * y_target[0]

        self.optimizer.zero_grad()
        policy_loss.backward()

        self.optimizer.step()

        self.training_initialized = True

        while True:
            # print(self.terminate)
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
