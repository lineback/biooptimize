import numpy as np
import random
import math
from bitstring import BitArray

class GA(object):
    def __init__(self, length, popSize, fitFunc, pc, pm, seed, learning=False):
        """
        Constructor
        ------------------------------------------------------------------------
        Creates a genetic algorithm object where each member of the population 
        has a length of l and the fitness of a string is determined by fitness
        The population size is determined by popSize
        ------------------------------------------------------------------------
        Fitness should be a function that takes a bit string of length,l, and 
        returns a real number
        """
        random.seed(seed)
        np.random.seed(seed)
        self.length = length
        self.maxString = pow(2,length-1)
        self.popSize = popSize
        self.fitFunc = fitFunc
        self.learning = learning
        self.pc = pc
        self.pm = pm
        self.fitness = np.zeros(popSize)
        self.normFitness = np.zeros(popSize)
        self.cumFitness = np.zeros(popSize)
        self.population = []
        for i in xrange(popSize):
            randInt = random.randint(0,self.maxString)
            bitarrayString = 'uint:%d=%d' %(self.length, randInt)
            self.population.append(BitArray(bitarrayString))
        
    def Respawn(self):
        """
        respawns a random population of bit stings, any population already
        extant will be overwritten
        """
        self.population = []
        for i in xrange(self.popSize):
            randInt = random.randint(0,self.maxString)
            bitarrayString = 'uint:%d=%d' %(self.length, randInt)
            self.population.append(BitArray(bitarrayString))

    def calcFitness(self):
        """
        Calculates the absolute and normalized fitness of each member of the 
        population
        """
        # calculate fitness
        for i in xrange(self.popSize):
            self.fitness[i] = self.fitFunc(self.population[i])
        # calculate norm fitness
        self.normFitness = self.fitness / np.sum(self.fitness)
        # calculate cumulative fitness
        self.cumFitness = np.cumsum(self.normFitness)
        # get best and average
        best = self.fitness.max()
        worst = self.fitness.min()
        ave = self.fitness.mean()
        num_ones = (self.population[(self.fitness).argmax()]).count(True)
        return (best, worst, ave, float(num_ones))

    def crossOver(self,idxa, idxb):
        """
        takes two bit strings and performa a cross over
        """
        if not self.learning:
            # create two new bit strings
            newA = BitArray(self.length)
            newB = BitArray(self.length)
            # select a cross over point
            crsovrpt = random.randint(1, self.length - 1)
            # create new strings by slicing the old strings
            newA[0:crsovrpt] = self.population[idxa][0:crsovrpt]
            newA[crsovrpt:self.length] = self.population[idxb][crsovrpt:self.length]
            newB[0:crsovrpt] = self.population[idxb][0:crsovrpt]
            newB[crsovrpt:self.length] = self.population[idxa][crsovrpt:self.length]
            return (newA, newB)
        else:
            # get the odd ones
            oddA = self.population[idxa][::2]
            oddB = self.population[idxb][::2]
            # create two new bit strings for odds
            newOddA = BitArray(self.length/2)
            newOddB = BitArray(self.length/2)
            # and two new for the entire strings
            newA = BitArray(self.length)
            newB = BitArray(self.length)
            # cross over point
            crsovrpt = random.randint(1, self.length/2 -1)
            newOddA[0:crsovrpt] = oddA[0:crsovrpt]
            newOddA[crsovrpt:self.length/2] = oddB[crsovrpt:self.length/2]
            newOddB[0:crsovrpt] = oddB[0:crsovrpt]
            newOddB[crsovrpt:self.length/2] = oddA[crsovrpt:self.length/2]
            # assign odd bitstrings to the odd position of the entire string
            newA[::2] = newOddA
            newB[::2] = newOddB
            return (newA, newB)
            

    def guessTheRest(self, genome):
        """
        guess the even numbered bits
        """
        bestString = BitArray(genome)
        bestScore = self.fitFunc(bestString)
        current = BitArray(genome)
        halfLength = self.length/2
        halfMax = pow(2,halfLength)
        for i in xrange(20):
            randInt = random.randint(0,halfMax - 1)
            bitarrayString = 'uint:%d=%d' %(halfLength, randInt)
            evenString = BitArray(bitarrayString)
            current[1::2] = evenString
            score = self.fitFunc(current)
            if score > bestScore:
                bestString = BitArray(current)
                bestScore = score
        return bestString

    def findNearest(self, value):
        """
        returns the index of the array that has a value closest to value
        """
        idx = (np.abs(self.cumFitness - value)).argmin()
        if self.cumFitness[idx] < value:
            idx = idx + 1
        return idx

    def CreateNextGen(self):
        """
        create the next generation
        """
        # calculate the fitness and get best and average
        (best, worst, ave, num_ones) = self.calcFitness()
        nextPop = []
        
        for i in xrange(self.popSize/2):
            # create two empty bit strings
            indA = BitArray(self.length)
            indB = BitArray(self.length)
            # pick two random floats in (0,1)
            randA = np.random.rand()
            randB = np.random.rand()
            # get the index of the two individuals they correspond to
            idxA = self.findNearest(randA)
            idxB = self.findNearest(randB)
            # decide whether to crossover or just copy
            if np.random.rand() < self.pc:
                 (indA,indB) = self.crossOver(idxA,idxB)
            else:
                indA = BitArray(self.population[idxA])
                indB = BitArray(self.population[idxB])
            if not self.learning:
                # for each bit in new bit strings decide whether to mutate
                for j in xrange(self.length):
                    if np.random.rand() < self.pm:
                        indA[j] = not indA[j]
                    if np.random.rand() < self.pm:
                        indB[j] = not indB[j]
            else:
                for j in xrange(0,self.length, 2):
                    if np.random.rand() < self.pm:
                        indA[j] = not indA[j]
                    if np.random.rand() < self.pm:
                        indB[j] = not indB[j]
                indA = self.guessTheRest(indA)
                indB = self.guessTheRest(indB)
            # add new bitstrings to new population
            nextPop.append(BitArray(indA))
            nextPop.append(BitArray(indB))
        # copy the population     
        self.population = nextPop
        return (best, worst, ave, num_ones)
    
    def SetFitFunction(self, newFunc):
        """
        sets the fitness function to newFunc
        """
        self.fitFunc = newFunc

    
