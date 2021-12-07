
from collections import deque
import glob
import os
import sys
import time
import random
import tensorflow as tf
import numpy as np
from myTensorboard import ModifiedTensorBoard
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
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

REPLAY_MEMORY_SIZE = 1
MODEL_NAME = "XCeption"
MIN_REPLAY_MEMORY_SIZE = 1
MINIBATCH_SIZE = 1
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE  # // 4
UPDATE_TARGET_EVERY = 5

EPISODES = 100

DISCOUNT = 0.99


class DQNAgent:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # memory replay
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # will track when it's time to update the target model
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.training_initialized = False  # waiting for TF to get rolling

    def create_model(self):
        base_model = Xception(weights=None, include_top=False,
                              input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        else:
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
            current_states = np.array([transition[0]
                                      for transition in minibatch])/255
            with self.graph.as_default():
                current_qs_list = self.model.predict(
                    current_states, PREDICTION_BATCH_SIZE)

            new_current_states = np.array(
                [transition[3] for transition in minibatch])/255
            with self.graph.as_default():
                future_qs_list = self.target_model.predict(
                    new_current_states, PREDICTION_BATCH_SIZE)

            X = []
            y = []

            for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                X.append(current_state)
                y.append(current_qs)

            log_this_step = False
            if self.tensorboard.step > self.last_logged_episode:
                log_this_step = True
                self.last_log_episode = self.tensorboard.step

            with self.graph.as_default():
                self.model.fit(
                    np.array(X)/255,
                    np.array(y),
                    batch_size=TRAINING_BATCH_SIZE,
                    verbose=0,
                    shuffle=False,
                    callbacks=[self.tensorboard] if log_this_step else None)

            if log_this_step:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        # first get warm up
        X = np.random.uniform(
            size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)

        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
