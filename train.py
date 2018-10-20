from environment import *
import tensorflow as tf
from actorCritic import ActorCritic
import pygame
from datetime import datetime
import os
import argparse


parser = argparse.ArgumentParser(description='DQN-snake testing.')
parser.add_argument('--learningRate', type=float, required=False, default=0.0001,
                    help='Learning rate of the training of the agent.')
parser.add_argument('--memorySize', type=int, required=False, default=100000,
                    help='The number of past events remembered by the agent.')

parser.add_argument('--discountRate', type=float, required=False, default=0.95,
                    help={'The discount rate is the parameter that indicates how many actions '
                          'will be considered in the future to evaluate the reward of a given action.'
                          'A value of 0 means the agent only considers the present action, '
                          'and a value close to 1 means the agent considers actions very far in the future.'})

parser.add_argument('--epsilonMin', type=float, required=False, default=0.05,
                    help='The percentage of random actions take by the agent.')


args = parser.parse_args()
learning_rate = args.learningRate
memory_size = args.memorySize
discount_rate = args.discount_rate
eps_min = args.epsilonMin

env = Environment()
session = tf.Session()
agent = ActorCritic(sess=session, learning_rate=learning_rate,
                    memory_size=memory_size, discount_rate=discount_rate, eps_min=eps_min)


pygame.init()  # Intializes the game

# Makes the folder where the tensorflow log will be written for Tensorboard visualization
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs/new_model'
if not os.path.isdir(root_logdir):
    os.makedirs(root_logdir)

log_dir = '{}/run-{}/'.format(root_logdir, now)
file_writer = tf.summary.FileWriter(log_dir, session.graph)  # Log file for Tensorboard


running = True
action = 0
done = False
iteration = 0
training_interval = 2
n_steps = 5000000  # Total number of training steps
n_games = 0
mean_score = 0

with session as sess:
    training_start = agent.start()

    while running:
        iteration += 1
        env.render()

        if done:  # Game over, start a new game
            env.reset()
            n_games += 1
            mean_score = env.total_rewards / n_games

        for event in pygame.event.get():  # Stop the program if we quit the game
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

        # Train the agent
        agent.train(file_writer, mean_score)

        if iteration % 500 == 0:
            print("\rTraining step {}/{} ({:.1f})%\tMean score {:.2f} ".format(
                step, n_steps, step * 100 / n_steps, mean_score), end="")

        if step > n_steps:
            break
