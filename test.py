from environment import *
import tensorflow as tf
from actorCritic import ActorCritic
import time
import pygame

env = Environment()
session = tf.Session()
agent = ActorCritic(session)

pygame.init()   # intializes the pygame
checkpoint_path = './models/model4/dqn.ckpt'   # path of my trained model
games_scores = []  # list that will contain the score of each game played by the bot


def test_network(n_games):
    episode = 0
    iterations_without_progress = 0
    max_without_progress = 175
    best_total = 0

    with session as sess:
        agent.saver.restore(sess, checkpoint_path)   # restore the model

        while episode < n_games:  # number of games that we want the robot to play
            time.sleep(600.0 / 10000.0)     # to make the game slow down

            env.render(display=True)
            observation = env.screenshot()
            cur_state = env.get_last_frames(observation)

            q_values = agent.predict(cur_state)
            action = np.argmax(q_values)

            new_state, reward, done = env.step(action)

            # check if the snake makes progress in the game
            if snake.total > best_total:
                best_total = snake.total
            else:
                iterations_without_progress += 1
            # if the snake gets stuck, the game is over
            if iterations_without_progress >= max_without_progress:
                done = True

            if done:   # game over, start a new game
                time.sleep(1)
                games_scores.append(snake.total)
                env.reset()
                episode += 1  # increment the number of games played
                iterations_without_progress = 0
                best_total = 0

    return games_scores


if __name__ == '__main__':

    games_scores = test_network(10)
    mean_score = np.mean(games_scores)
    std = np.std(games_scores)
    max_score = np.max(games_scores)

    print("Max score {:.2f}\tMean score {:.2f}\tStandard deviation {:.2f} ".format(max_score, mean_score, std))