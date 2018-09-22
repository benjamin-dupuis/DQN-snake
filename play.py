from environment import *
import pygame
import time


pygame.init()   # Intialize the game
snake = Snake()
apple = Apple()
env = Environment(screen_width=400, screen_height=400)   # Make the screen larger so it's easier to play
running = True
done = True
action = 0  # Initialize the action to be 0 (LEFT)

while running:
    time.sleep(0.08)   # Make the game slow down

    if done:   # Game over, start a new game
        time.sleep(1)
        env.reset()
        action = 0

    env.render(display=True)

    _, _, done = env.step(action)  # Make the snake move according to the chosen action

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 0
            if event.key == pygame.K_RIGHT:
                action = 1
            if event.key == pygame.K_UP:
                action = 2
            if event.key == pygame.K_DOWN:
                action = 3

        if event.type == pygame.QUIT:  
            running = False
