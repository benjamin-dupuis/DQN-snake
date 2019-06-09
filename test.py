import argparse
import os
import time

import tensorflow as tf

from actorCritic import ActorCritic
from environment import *
from utils import get_checkpoint_path

env = Environment()
session = tf.Session()
agent = ActorCritic(session)

pygame.init()   # Intializes the pygame
games_scores = []  # List that will contain the score of each game played by the gamebot


parser = argparse.ArgumentParser(description='DQN-snake testing.')

parser.add_argument('--numberOfGames', type=int, required=False, default=10,
                    help='Number of test games.')

parser.add_argument('--slowDownFactor', type=float, required=False, default=0.06,
                    help='The factor to make the game slow down. A value of 0 means the games is at full speed.')

parser.add_argument('--modelName', type=str, required=True,
                    help='The name of the model.')


def make_agent_play_games(n_games, slow_down_factor):
    """
    Make the agent play a given number of games

    :param n_games: The number of games to play.
    :param slow_down_factor: Throttling to make the snake move less rapidly.
    :return: A list containing the score of each game played.
    """
    episode = 0
    iterations_without_progress = 0
    max_without_progress = 200
    best_total = 0

    with session as sess:
        agent.saver.restore(sess, checkpoint_path)   # Restore the model

        while episode < n_games:  # Number of games that we want the robot to play
            time.sleep(slow_down_factor)     # Make the game slow down

            env.render(display=True)
            observation = env.screenshot()
            cur_state = env.get_last_frames(observation)

            q_values = agent.predict(cur_state)
            action = np.argmax(q_values)  # Optimal action

            new_state, reward, done = env.step(action)

            # Check if the snake makes progress in the game
            if env.snake.total > best_total:
                best_total = env.snake.total
                iterations_without_progress = 0
            else:
                iterations_without_progress += 1
            # If the snake gets stuck, the game is over
            if iterations_without_progress >= max_without_progress:
                done = True

            if done:   # Game over, start a new game
                time.sleep(1)
                games_scores.append(env.snake.total)
                env.reset()
                episode += 1  # Increment the number of games played
                iterations_without_progress = 0
                best_total = 0

    return games_scores


if __name__ == '__main__':

    args = parser.parse_args()

    n_games = args.numberOfGames
    slow_down_factor = args.slowDownFactor
    model_name = args.modelName
    checkpoint_path = get_checkpoint_path(model_name=model_name)

    if os.path.isfile(checkpoint_path + ".index"):  # Check to see if the model exists
        games_scores = make_agent_play_games(n_games, slow_down_factor)
        mean_score = np.mean(games_scores)
        std = np.std(games_scores)
        max_score = np.max(games_scores)

        print("Max score {:.2f}\tMean score {:.2f}\tStandard deviation {:.2f} ".format(max_score, mean_score, std))

    else:
        raise ValueError('Model file does not exist : a model file is required for testing')
