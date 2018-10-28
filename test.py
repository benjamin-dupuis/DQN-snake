from environment import *
import tensorflow as tf
from actorCritic import ActorCritic
import time
import pygame
import os
import argparse

env = Environment()
session = tf.Session()
agent = ActorCritic(session)

pygame.init()   # Intializes the pygame
checkpoint_path = './models/new_model/dqn.ckpt'   # Path of the trained model
games_scores = []  # List that will contain the score of each game played by the gamebot


parser = argparse.ArgumentParser(description='DQN-snake testing.')

parser.add_argument('--numberOfGames', type=int, required=False, default=10,
                    help='Number of test games.')

parser.add_argument('--slowDownFactor', type=float, required=False, default=0.06,
                    help='The factor to make the game slow down. A value of 0 means the games is at full speed.')

parser.add_argument('--modelName', type=str, required=True,
                    help='The name of the model.')


def test_network(n_games, slow_down_factor):
    episode = 0
    iterations_without_progress = 0
    max_without_progress = 175
    best_total = 0

    with session as sess:
        agent.saver.restore(sess, checkpoint_path)   # Restore the model

        while episode < n_games:  # Number of games that we want the robot to play
            time.sleep(slow_down_factor)     # Make the game slow down

            env.render(display=True)
            observation = env.screenshot()
            cur_state = env.get_last_frames(observation)

            q_values = agent.predict(cur_state)
            action = np.argmax(q_values)

            new_state, reward, done = env.step(action)

            # Check if the snake makes progress in the game
            if snake.total > best_total:
                best_total = snake.total
                iterations_without_progress = 0
            else:
                iterations_without_progress += 1
            # If the snake gets stuck, the game is over
            if iterations_without_progress >= max_without_progress:
                done = True

            if done:   # Game over, start a new game
                time.sleep(1)
                games_scores.append(snake.total)
                env.reset()
                episode += 1  # Increment the number of games played
                iterations_without_progress = 0
                best_total = 0

    return games_scores


if __name__ == '__main__':

    args = parser.parse_args()

    if os.path.isfile(checkpoint_path + ".index"):  # Check to see if the model exists
        games_scores = test_network(args.numberOfGames, args.slowDownFactor)
        mean_score = np.mean(games_scores)
        std = np.std(games_scores)
        max_score = np.max(games_scores)

        print("Max score {:.2f}\tMean score {:.2f}\tStandard deviation {:.2f} ".format(max_score, mean_score, std))

    else:
        raise ValueError('Model file does not exist : a model file is required for testing')
