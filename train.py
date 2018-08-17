from environment import *
import tensorflow as tf
from actorCritic import ActorCritic
import pygame
from datetime import datetime
import os


env = Environment()
session = tf.Session()
agent = ActorCritic(session)

pygame.init()  # intializes the pygame

# makes the folder where the tensorflow log will be written for Tensorboard visualization
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs/new_model'
if not os.path.isdir(root_logdir):
    os.makedirs(root_logdir)

log_dir = '{}/run-{}/'.format(root_logdir, now)
file_writer = tf.summary.FileWriter(log_dir, session.graph)  # log file for Tensorboard


running = True
action = 0
done = False
iteration = 0
training_interval = 2
n_steps = 4000000  # total number of training steps
n_games = 0
mean_score = 0

with session as sess:
    training_start = agent.start()

    while running:
        iteration += 1
        env.render()

        if done:  # game over, start a new game
            env.reset()
            n_games += 1
            mean_score = env.total_rewards / n_games

        for event in pygame.event.get():  # stops the program if we quit the game
            if event.type == pygame.QUIT:
                running = False

        observation = env.screenshot()
        cur_state = env.get_last_frames(observation)
        step = agent.global_step.eval()

        action = agent.act(cur_state, step)
        new_state, reward, done = env.step(action)
        agent.remember(cur_state, action, reward, new_state, done)

        # Only train at regular intervals
        if iteration < training_start or iteration % training_interval != 0:
            continue

        # train the agent
        agent.train()

        if iteration % 200 == 0:
            print("\rTraining step {}/{} ({:.1f})%\tMean score {:.2f} ".format(
                step, n_steps, step * 100 / n_steps, mean_score), end="")
            agent.write_summary(file_writer, mean_score, step)  # write the loss summary in Tensorboard

        if step > n_steps:
            break
