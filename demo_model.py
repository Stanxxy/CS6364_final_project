from trail_DQN_agent import MEMORY_FRACTION
from trail_car_environment import CarEnv
from keras.models import load_model
import keras.backend.tensorflow_backend as backend
import tensorflow as tf
import time
import cv2
import numpy as np
from collections import deque
import carla
import os
import sys
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please input model path.")
        exit()

    MODEL_PATH = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print("cannot find model. Check your path")
        exit()

    # 'models/Xception__-118.00max_-179.10avg_-250.00min__1566603992.model'
    # Memory fraction
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv()
    env.SHOW_CAM = True

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.ones((1, env.im_height, env.im_width, 3)))

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            # image.save_to_disk('output/%06d.png' % image.frame)
            # cv2.imshow(f'Agent - preview', current_state)
            # cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = model.predict(
                np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(
                f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: \
                  [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for sensor in env.sensor_list:
            sensor.stop()
        env.client.apply_batch([carla.command.DestroyActor(x)
                                for x in env.actor_list])
