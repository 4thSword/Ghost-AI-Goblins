import retro
import neat
import numpy as np
import cv2
import pickle

env = retro.make('GhostsnGoblins-Nes',)

confing = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,'config-feedforward')

env.reset()

done = False
if __name__ == "__main__":
    
    while not done:
     #env.render()
     action = env.action_space.sample()
     #action = [0, 1, 1, 0, 0, 1, 1, 1, 1]

     ob, reward, done, info = env.step(action)
     print("Action:",action)
     print('Image:',ob.shape, "reward:",reward, "done",done)
     print('Info:',info)