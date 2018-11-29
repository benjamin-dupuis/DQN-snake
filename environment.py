import pygame
import random
from scipy.spatial import distance
import numpy as np
from PIL import Image
from collections import deque


WHITE = (255, 255, 255)
OUT_PENALTY = -1
LIFE_REWARD = 0   
APPLE_REWARD = 1
INPUT_HEIGHT = 84 
INPUT_WIDTH = 84   

pygame.init()


def image_transform(image_path, image_width, image_heigth):
    """
    Loads an image and reshapes it.
    
    :param image_path: The path of the image to load
    :param image_width: The desired image width
    :param image_heigth: The desired image height
    :return: Image with dimension (image_width, image_height)
    """
    image = pygame.image.load(image_path).convert()
    image = pygame.transform.scale(image, (image_width, image_heigth))
    return image


class Snake:
    """
    Represents the snake and his interactions with his environment.
    The initial configuration of the snake is obtained by the reset function.
    """

    def __init__(self, lenght=3, speed=25):
        self.lenght = lenght
        self.size = speed
        self.speed = speed
        self.direction = None
        self.x = None
        self.y = None
        self.total = None
        self.tail = None

    def _is_moving_backwards(self, action):
        """
        Checks if the snake is trying to move backwards (which you can't do in the game)
        :param action: The action selected by the agent
        :return: True is the action is the inverse of the snake's direction and False otherwise
        """
        # If the action selected and the direction are opposites
        if self.direction == 0 and action == 1:
            return True
        if self.direction == 1 and action == 0:
            return True
        if self.direction == 3 and action == 2:
            return True
        if self.direction == 2 and action == 3:
            return True
        else:
            return False

    def move(self, action):
        # If the snake tries to go backwards, it keeps his original direction
        if self._is_moving_backwards(action):
            action = self.direction
        else:
            self.direction = action

        if action == 0:  # LEFT
            self.x -= self.speed
        if action == 1:  # RIGHT
            self.x += self.speed
        if action == 2:  # UP
            self.y -= self.speed
        if action == 3:  # DOWN
            self.y += self.speed

        self.tail.appendleft([self.x, self.y])
        self.tail.pop()

    def eat(self):
        self.total += 1
        self.tail.appendleft([self.x, self.y])

    def dead(self, screen_width, screen_height):
        self.total = 0
        self.lenght = 3
        self.x = screen_width/2
        self.y = screen_height/2
        self.tail = deque([self.x + i * self.speed, self.y] for i in range(self.lenght))
        self.direction = 0

    def draw(self, screen, image):
        """
        Function that draws every part of the snake body.
        
        :param screen: pygame screen
        :param image: image that we want to draw on the screen
        """
        for i in range(len(self.tail)):
            screen.blit(image, (self.tail[i][0], self.tail[i][1]))


class Apple:
    """
    Represents the Apple entity, that obtains a new position when eaten.
    The initial configuration of the apple object is obtained by the reset function.
    """

    def __init__(self, size=25):
        self.size = size
        self.x = None
        self.y = None

    def reset(self, screen_width, screen_height):
        self.x = int(screen_width/3)
        self.y = int(screen_height/3)

    def get_new_position(self, screen_width, screen_height, snake_tail):
        """
        Gets a new position for the apple. 
        Checks to be sure the apple is not placed inside the snake's body.
        
        :param screen_width: The width of the game screen
        :param screen_height: The height of the game screen
        :param snake_tail: The list representing the tail of the snake
        """
        all_positions = [[x, y] for x in range(self.size, screen_width - self.size)
                         for y in range(self.size, screen_height - self.size)]
        allowed_positions = [coord for coord in all_positions if coord not in snake_tail]
        self.x = random.choice(allowed_positions)[0]
        self.y = random.choice(allowed_positions)[1]

    def draw(self, screen, image):
        screen.blit(image, (self.x, self.y))


class Environment:
    """
    Represents the RL environment where the agent interacts and obtains rewards associated with is actions.
    """

    screen_width = 250
    screen_height = 250
    apple_size = 25

    def __init__(self, screen_width=screen_width, screen_height=screen_height):
        self.total_rewards = 0
        self._screen = pygame.display.set_mode((screen_width, screen_height))
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._frames = None
        self._num_last_frames = 4
        self.apple = Apple(self.apple_size)
        self.snake = Snake()

        self.reset()
        self._game_reward = 0

    def reset(self):
        self.snake.dead(self._screen_width, self._screen_height)
        self.apple.reset(self._screen_width, self._screen_height)
        self._frames = None
        self._game_reward = 0

    def get_last_frames(self, observation):
        """
        Gets the 4 previous frames of the game as the state.
        Credits goes to https://github.com/YuriyGuts/snake-ai-reinforcement.
        
        :param observation: The screenshot of the game
        :return: The state containing the 4 previous frames taken from the game
        """
        frame = observation
        if self._frames is None:
            self._frames = deque([frame] * self._num_last_frames)
        else:
            self._frames.append(frame)
            self._frames.popleft()
        state = np.asarray(self._frames).transpose()  # Transpose the array so the dimension of the state is (84,84,4)
        return state

    def render(self, display=False):
        """
        Shows and updates the game on the screen.
        
        :param display: True if we want to show the score in the title of the screen
        """
        self._screen.fill(WHITE)

        image_snake = image_transform('./images/bloc.png', self.snake.size, self.snake.size)
        image_apple = image_transform('./images/apple.jpg', self.apple.size, self.apple.size)

        self.apple.draw(self._screen, image_apple)
        self.snake.draw(self._screen, image_snake)

        if display is True:
            pygame.display.set_caption('Score : ' + str(self.snake.total))
        pygame.display.update()

    def screenshot(self):
        """
        Takes a screenshot of the game , converts it to grayscale, reshapes it to size INPUT_HEIGHT, INPUT_WIDTH,
        and returns a np.array.
        Credits goes to https://github.com/danielegrattarola/deep-q-snake/blob/master/snake.py
        """
        data = pygame.image.tostring(self._screen, 'RGB')  # Take screenshot
        image = Image.frombytes('RGB', (self._screen_width, self._screen_height), data)
        image = image.convert('L')  # Convert to greyscale
        image = image.resize((INPUT_HEIGHT, INPUT_WIDTH)) 
        matrix = np.asarray(image.getdata(), dtype=np.uint8)
        matrix = (matrix - 128)/(128 - 1)  # Normalize from -1 to 1
        return matrix.reshape(image.size[0], image.size[1])
    
    def step(self, action):
        """
        Makes the snake move according to the selected action.
        
        :param action: The action selected by the agent
        :return: The new state, the reward, and the done value
        """
        done = False
        self.snake.move(action)

        reward = LIFE_REWARD   # Reward given to stay alive

        # IF SNAKE QUITS THE SCREEEN
        if self.snake.x in [-self.snake.size, self._screen_width] or self.snake.y in [-self.snake.size, self._screen_height]:
            reward = OUT_PENALTY
            done = True

        snake_position = (self.snake.x, self.snake.y)
        apple_position = (self.apple.x, self.apple.y)
        dst = distance.euclidean(snake_position, apple_position)

        # IF SNAKES EATS THE APPLE
        if dst <= self.apple.size:
            self.snake.eat()
            self.apple.get_new_position(self._screen_width, self._screen_height, self.snake.tail)
            self.total_rewards += 1
            self._game_reward += APPLE_REWARD
            reward = self._game_reward

        # IF SNAKE EATS ITSELF
        head_pos = (self.snake.tail[0][0], self.snake.tail[0][1])
        for i in range(2, len(self.snake.tail)):
            body_part_pos = (self.snake.tail[i][0], self.snake.tail[i][1])
            dst_body = distance.euclidean(head_pos, body_part_pos)  
            if dst_body < self.snake.size:
                done = True
                reward = -1
                break

        new_observation = self.screenshot()
        new_state = self.get_last_frames(new_observation)
        return new_state, reward, done
