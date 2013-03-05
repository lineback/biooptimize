from numpy import *
from math import *
import matplotlib.pyplot as plt

class Swarm():
    def __init__(self, bounds, numParticles,v_max,fitFunc, inertia, cog, social, radius=0, decCog=False, doGlobal=True):
        self.size = len(bounds)
        self.bounds = bounds
        self.particles = numParticles
        #self.tolerance = args.tol 
        #self.radius = radius
        self.doGlobal = doGlobal
        self.v_max = v_max
        self.fitFunc = fitFunc
        self.inertia = inertia
        self.cog = cog
        self.social = social
        self.position = random.rand(self.particles,self.size)
        for i in xrange(self.particles):
            for j in xrange(self.size):
                self.position[i][j] = self.position[i][j]*(bounds[j][1] - bounds[j][0]) + bounds[j][0]
        #self.position = self.position*200 -100
        self.velocity = random.uniform(0,1,(self.particles,self.size))
        self.best_position = zeros((self.particles,self.size))
        self.curr_fitness = zeros(self.particles)
        self.best_fitness = zeros(self.particles)
        self.global_best = 0.0
        self.global_best_idx = 0
        if not self.doGlobal:
            self.local_best = arange(self.particles)
            self.radius = radius
        self.decCog = decCog
        self.calc_fitness()
        
    def calc_fitness(self):
        for i in xrange(self.particles):
            #temp = 0
            #temp = (self.position[i]+5)
            self.curr_fitness[i] = self.fitFunc(self.position[i])
            if self.curr_fitness[i] > self.best_fitness[i]:
                self.best_fitness[i] = copy(self.curr_fitness[i])
                self.best_position[i] = copy(self.position[i])
            if self.curr_fitness[i] > self.global_best:
                self.global_best = copy(self.curr_fitness[i])
                self.global_best_idx = i
        if not self.doGlobal:
            self.set_local_best()

    def set_local_best(self):
        for idx in xrange(self.particles):
            for i in xrange(idx - self.radius , idx + self.radius +1):
                j = i%self.particles
                if self.curr_fitness[j] > self.curr_fitness[self.local_best[idx]]:
                    self.local_best[idx] = j
    
    def update_velocity(self):
        if not self.doGlobal:
            for i in xrange(self.particles):
                self.velocity[i] += (self.inertia*self.velocity[i] + 
                                     (self.cog*random.rand()*(self.best_position[i]-self.position[i]) + 
                                     self.social*random.rand()*(self.best_position[self.local_best[i]]-self.position[i])))
                mag_vel = sqrt(dot(self.velocity[i],self.velocity[i]))
                if mag_vel>self.v_max:
                    self.velocity[i] = self.velocity[i]*self.v_max/mag_vel
        else:
            for i in xrange(self.particles):
                self.velocity[i] = (self.inertia*self.velocity[i] + 
                                    (self.cog*random.rand()*(self.best_position[i]-self.position[i]) +
                                     self.social*random.rand()*(self.best_position[self.global_best_idx]-self.position[i])))
                mag_vel = sqrt(dot(self.velocity[i],self.velocity[i]))
                if mag_vel>self.v_max:
                    self.velocity[i] = self.velocity[i]*self.v_max/mag_vel

    def update_position(self):
        self.position += self.velocity
        if self.decCog:
            self.cog = 0.95*self.cog
        
    def get_best_fitness(self):
        return self.global_best

    def get_best_position(self):
        return self.best_position[self.global_best_idx]
    
    def plot_particles(self, fileName):
        """
        makes a scatter plot of the particle position
        NOTE: this will only work if the size of the system is 2
        """
        x_pos = zeros(self.particles)
        y_pos = zeros(self.particles)
        for particle in xrange(self.particles):
            x_pos[particle] = self.position[particle][0]
            y_pos[particle] = self.position[particle][1]

        plt.scatter(x_pos, y_pos)
        plt.ylim(self.bounds[1][0], self.bounds[1][1])
        plt.xlim(self.bounds[0][0], self.bounds[0][1])
        plt.savefig(fileName)
        plt.clf()
    
    def get_ave_pos(self):
        return mean(self.position, 0)

    def get_inertia(self):
        return self.inertia

    def set_inertia(self, inertia):
        self.inertia = inertia
