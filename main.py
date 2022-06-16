# Imports
import pygame 
import time
import random 
import keyboard
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Initialize pygame 
pygame.init()

# Game Variables
WIDTH = 500
HEIGHT = 700
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
TARGET_SIZE_1 = (200, 200)
TARGET_SIZE_2 = (400, 400)

pipe_x = WIDTH
pipe_2_x = WIDTH + 120
pipe_1_top_y = random.randint(-240, 0)
pipe_1_bottom_y = random.randint(HEIGHT - 250, HEIGHT)
pipe_2_top_y = random.randint(-240, 0)
pipe_2_bottom_y = random.randint(HEIGHT - 250, HEIGHT)
pipe_x_change = 4 

player_y = HEIGHT / 2
player_x = 50 
player_y_change = 4

# Make the display 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

jump_index = 0
normal_index = 0


# Loading in the model
model = keras.models.load_model("flappy-bird-5-epoch-v1.0/")
# model = keras.models.load_model("flappy-bird-player-neural-net-v1.0")
class_names = ["jump", "normal"]

# Main Game Loop 
running = True 
clock = pygame.time.Clock()

while running:
    # Fill the screen with black
    screen.fill(BLACK)
    clock.tick(200)

    # Creating the bird 
    player = pygame.draw.rect(screen, WHITE, (player_x, player_y, 30, 30))
    
    # Creating the pipes 
    pipe_1_top = pygame.draw.rect(screen, WHITE, (pipe_x, pipe_1_top_y, 30, 250))
    pipe_1_bottom = pygame.draw.rect(screen, WHITE, (pipe_x, pipe_1_bottom_y, 30, 250))

    pipe_2_top = pygame.draw.rect(screen, WHITE, (pipe_2_x, pipe_2_top_y, 30, 250))
    pipe_2_bottom = pygame.draw.rect(screen, WHITE, (pipe_2_x, pipe_2_bottom_y, 30, 250))

    # Prediction Code
    test_img = "prediction_image.png"
    img = image.load_img(test_img, target_size=TARGET_SIZE_2)

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=[0])
    images = np.vstack([X])

    prediction = model.predict(images)
    text_prediction = class_names[np.argmax(prediction)]
    print(text_prediction)

    if text_prediction == "jump":
       keyboard.press("space")

    else:
       keyboard.release("space")



    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        keys = pygame.key.get_pressed()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Take screen shot of the game
                # pygame.image.save(screen, f"dataset/jump/{jump_index}.png")
                jump_index += 1
                player_y_change = -4


        if event.type == pygame.KEYUP:
            player_y_change = 4

    # Boundary Checkiing 
    if pipe_x < -33:
        pipe_x = WIDTH + 120

    if pipe_2_x < -33:
        pipe_2_x = WIDTH + 120

    if player_y >= HEIGHT - 30:
        player_y = HEIGHT - 30

    # Collision 
    if player.colliderect(pipe_1_top) or player.colliderect(pipe_1_bottom) or player.colliderect(pipe_2_top) or player.colliderect(pipe_2_bottom):
        time.sleep(0.05)
        break

    if not keys[pygame.K_SPACE]:
        # Update the display
        # pygame.image.save(screen, f"dataset/normal/{normal_index}.png")
        normal_index += 1

    pygame.image.save(screen, "prediction_image.png")
    
    pipe_x -= pipe_x_change
    pipe_2_x -= pipe_x_change
    player_y += player_y_change
    pygame.display.update()
