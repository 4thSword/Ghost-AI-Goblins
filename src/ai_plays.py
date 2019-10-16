import retro
import neat
import cv2
import pickle
import numpy as np
import visualize
import glob, os

def image_to_array(image,inx,iny):
    # Converts image to an inx * iny size, change color to greyscale and flatten it into 1D ndarray
    image = cv2.resize(image,(inx,iny)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = np.reshape(image,(inx,iny))
    return np.ndarray.flatten(image)

def eval_genomes(genomes, config):  


    for genome_id, genome in genomes:
        
        observ = env.reset() # First image observed
        random_action = env.action_space.sample()

        # Takes a tuple with the size of the image and; inc = color
        inx,iny,inc = env.observation_space.shape 
        # Image reduction for faster processing
        inx = int(inx/8)
        iny = int(iny/8)
        # 20 Networks
        net = neat.nn.RecurrentNetwork.create(genome,config)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        prev_lives = 3
        
        done = False

        while not done:
            # Shows the game playing, comment it to a fas training.
            env.render() 
            frame+=1

            #Converts the curent frame to grayscale, reduces it and flatten into an array.
            oned_image = image_to_array(observ,inx,iny)

            # Give an output for current frame from neural network
            neuralnet_output = net.activate(oned_image)

            # Try given output from network in the game and takes the new response from the environment.
            observ, reward, done, info = env.step(neuralnet_output)
            
            if prev_lives>info['lives']:
                fitness_current -= 1000
                prev_lives = info['lives']
        
            fitness_current += reward
            
            if fitness_current>current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter+=1
                # count the frames until it successful

            # Train for max 750 frames
            if done or counter == 750:
                done = True 
                print(genome_id,fitness_current)
            
            genome.fitness = fitness_current

def load_last_checkpoint():
    try:
        os.chdir('../checkpoints')
        checkpoints = [f for f in glob.glob('neat-checkpoint-*')]
        checkpoints = [int(f[16:])for f in checkpoints]
        checkpoints.sort()
        return neat.Checkpointer.restore_checkpoint('neat-checkpoint-{}'.format(checkpoints[-1]))
    except:
        print('No checkpoints in our folder, starting training from generation 0')
        return neat.Population(config)

if __name__ == "__main__":
    # Creates our ghosts and goblings environment:
    #env = retro.make('GhostsnGoblins-Nes','Level1')
    env = retro.make(game='GhostsnGoblins-Nes', record='../records')
    # Loads our selected configuration for our Neat neural network:
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,'config-feedforward')

    '''
    with open('winner.pkl', 'rb') as input_file:
        genome = pickle.load(input_file)
    '''

    # Restore the last checkpoint if exist, else starts from zero:
    p = load_last_checkpoint()

    # Uncomment to restore a selected checkpoint if don't want to restore last checkpoint 
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-803')

    

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    
    # Save the process after each 10 Generations
    p.add_reporter(neat.Checkpointer(10,filename_prefix='../checkpoints/neat-checkpoint-'))

    winner = p.run(eval_genomes)


    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
    
    
    