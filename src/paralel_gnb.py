import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle
import glob

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        
        self.env = retro.make('GhostsnGoblins-Nes')
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        current_max_fitness = 0
        fitness_current = 0
        counter = 0
        prev_lives = 3

        imgarray = []
        
        while not done:
            #self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            actions = net.activate(imgarray)
            
            ob, reward, done, info = self.env.step(actions)
            
            if prev_lives>info['lives']:
                reward -= 1000
                prev_lives = info['lives']

            fitness_current += reward
            if fitness_current>current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter+=1
                # count the frames until it successful

            # Train for maxs
            if done or counter == 850:
                done = True 
            
                         
        print(self.genome.key,fitness_current)
        return fitness_current

def eval_genomes(genome, config):
    
    worky = Worker(genome, config)
    return worky.work()

def load_last_checkpoint():
    try:
        checkpoints = [f for f in glob.glob('neat-checkpoint-*')]
        checkpoints = [int(f[16:])for f in checkpoints]
        checkpoints.sort()
        return neat.Checkpointer.restore_checkpoint('neat-checkpoint-{}'.format(checkpoints[-1]))
    except:
        print('No checkpoints in our folder, starting training from generation 0')


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

#Loads the last checkpoint if exists:
p = load_last_checkpoint()


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(8, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
