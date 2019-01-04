from environment import *
import tensorflow as tf
from actorCritic import ActorCritic
import pygame
import argparse
from utils import get_checkpoint_path, get_file_writer


parser = argparse.ArgumentParser(description='DQN-snake testing.')

parser.add_argument('--modelName', type=str, required=True,
                    help='The name of the model.')

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

parser.add_argument('--trainingInterval', type=int, required=False, default=2,
                    help='The interval between two training steps.')

parser.add_argument('--numberOfSteps', type=int, required=False, default=5000000,
                    help='The total number of training steps.')


args = parser.parse_args()
model_name = args.modelName
learning_rate = args.learningRate
memory_size = args.memorySize
discount_rate = args.discountRate
eps_min = args.epsilonMin
training_interval = args.trainingInterval
n_steps = args.numberOfSteps


session = tf.Session()


def train(env, agent):
    file_writer = get_file_writer(model_name=model_name, session=session)
    checkpoint_path = get_checkpoint_path(model_name=model_name)

    running = True
    done = False
    iteration = 0
    n_games = 0
    mean_score = 0

    with session:
        training_start = agent.start(checkpoint_path)

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
            agent.train(checkpoint_path, file_writer, mean_score)

            if iteration % 500 == 0:
                print("\rTraining step {}/{} ({:.1f})%\tMean score {:.2f} ".format(
                    step, n_steps, step * 100 / n_steps, mean_score), end="")

            if step > n_steps:
                break


if __name__ == '__main__':
    pygame.init()  # Intializes the game

    environment = Environment()
    training_agent = ActorCritic(sess=session, training_steps=n_steps, learning_rate=learning_rate,
                                 memory_size=memory_size, discount_rate=discount_rate,
                                 eps_min=eps_min)

    train(environment, training_agent)
