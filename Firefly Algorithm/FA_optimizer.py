####*************************************####
####         FIREFLY ALGORITHM           ####
####*************************************####

# NOTES:
# This is the implemention of the basic unmodified firefly algorithm

from dataclasses import dataclass
import numpy as np
from random import random as rnd
import random
import copy
import math as mth

g_l_rand_buffer = []
g_n_eval = 0

#Problem Class - very much related to the objective function
class Problem:
    m_h_costFunction = None
    m_n_dim = 2
    m_f_lb = -10 
    m_f_ub = 10
    def Create(self, h_costFunction, n_dim, f_lb, f_ub):
        self.m_h_costFunction = h_costFunction
        self.m_n_dim = n_dim
        self.m_f_lb = f_lb
        self.m_f_ub = f_ub
        return

#Param Class - related  to FireFlyAlgo with the effective parametrs to solve the corresponding problem
class Params:
    m_n_maxIt = 500
    m_n_pop = 25
    m_f_alpha = 0.2
    m_f_gamma = 1
    def Create(self, n_maxIt, n_pop, f_alpha, f_gamma):
        self.m_n_maxIt = n_maxIt
        self.m_n_pop = n_pop
        self.m_f_alpha = f_alpha
        self.m_f_gamma = f_gamma

#FireFly class
#@dataclass
class FireFly:
    m_l_position = []
    m_f_cost = -mth.inf

#Results class - has list of real values as the best solution and the best fitness/cost
class Results:
    m_l_bestSol = None
    m_f_bestCost = -mth.inf
    def Create(self, l_bestSol, f_bestCost):        
        self.m_l_bestSol = l_bestSol
        self.m_f_bestCost = f_bestCost

class RandomBufferGenerator:
    @staticmethod
    def Generate (n_count = 1000):
        global g_l_rand_buffer
        g_l_rand_buffer = [0]*n_count
        for ind in range(n_count):
            g_l_rand_buffer[ind] = round(rnd(),3)


## CREATION OF ALPHA BETA FOR FIREFLYALGO
class Helper:     
    @staticmethod
    def CreateAlphaMatrix(n_dim, alpha_value, upper_limit, lower_limit, sub_val, precision):
        a_alpha= [round((rnd() - sub_val)*(upper_limit-lower_limit)+lower_limit,precision) for x in range(n_dim)]
        return a_alpha
    @staticmethod
    def CreateBetaMatrix(n_dim, beta_value, upper_limit, lower_limit, precision):
        a_beta= [round((rnd()*beta_value)*(upper_limit-lower_limit)+lower_limit,precision) for x in range(n_dim)]
        return a_beta
    @staticmethod
    def CreatePosition(n_dim, upper_limit, lower_limit, precision):
        a_position = [round((rnd())*(upper_limit-lower_limit)+lower_limit,precision) for x in range(n_dim)]
        return a_position
    
    ## FITNESS EVALUATION OF AN Individual 
    #(CHANGE/ENTER THE OBJECTIVE FUNCTION HERE)
    #Michalewicz function global minima at m f∗ ≈ −1.801 in 2-D occurs at (2.20319, 1.57049)
    @staticmethod
    def fitness_calculation_f1(individual):
        global g_n_eval
        g_n_eval += 1
        count = len(individual)
        fitness_value = 0
        m = 10
        for i in range(count):
            fitness_value += (mth.sin(individual[i]))*mth.pow(mth.sin((i+1)*mth.pow(individual[i],2)/mth.pi), 2*m)
        return fitness_value
    #Ref 112 Multimodal, continuous, non-differentiable,separable, non-scalable

    @staticmethod
    def fitness_calculation_f2(individual):

        count = len(individual)
        fitness_value = 0
        for i in range(count):
            fitness_value += abs(individual[i]*mth.sin(individual[i]) + 0.1*individual[i])
        return -fitness_value
        #Ref 112 unimodal, continuous, nondifferentiable, scalable, separable, stochastic
    @staticmethod
    def fitness_calculation_f5(individual):
        count = len(individual)
        fitness_value = 0
        global g_l_rand_buffer
        for i_count in range(count):
            fitness_value += g_l_rand_buffer[i_count]*(abs(individual[i_count])**i_count)
        return -fitness_value

##FireFly Algorithm
class FireFlyAlgorithm:
    def RunFA (self,problem, params):
        print('Starting FA ...')
        h_costFunction = problem.m_h_costFunction        # Cost Function
        n_dim = problem.m_n_dim          # Number of Decision Variables
        f_lb = problem.m_f_lb      # Lower Bound of Variables
        f_ub = problem.m_f_ub      # Upper Bound of Variables
        f_gamma = params.m_f_gamma            
        f_alpha = params.m_f_alpha         
        f_m = 2.5
        f_beta_0 = 1 
        f_extremeMinVal = -mth.inf
        ## PSO Parameters
        n_maxGen = params.m_n_maxIt      # Maximum Number of Iterations
        n_pop = params.m_n_pop
        # best sol and cost
        # generate firefly array and initialize randomly
        o_bestSol = FireFly()
        o_bestSol.m_f_cost = -mth.inf
        l_fireFlyPop = [FireFly() for x in range(n_pop)]
        for i_pop in range(n_pop): #initialize
            l_fireFlyPop[i_pop].m_l_position = Helper.CreatePosition(n_dim, f_ub, f_lb, 3)
            l_fireFlyPop[i_pop].m_f_cost = f_extremeMinVal#h_costFunction(l_fireFlyPop[i_pop].m_l_position)
            if l_fireFlyPop[i_pop].m_f_cost >= o_bestSol.m_f_cost:
                o_bestSol = l_fireFlyPop[i_pop]
        #l_fireFlyPop = sorted(l_fireFlyPop, key=lambda firefly: firefly.m_f_cost, reverse=True)       
        l_bestCost = [f_extremeMinVal for x in range(n_maxGen)]
        for i_gen in range(n_maxGen):    
            for i_pop in range(n_pop):
                o_newFireFly = copy.deepcopy(l_fireFlyPop[i_pop])
                #b_updated = False
                for i_pop2 in range(0, i_pop):
                    if l_fireFlyPop[i_pop].m_f_cost < l_fireFlyPop[i_pop2].m_f_cost:
                        a_dist = np.subtract(l_fireFlyPop[i_pop].m_l_position, l_fireFlyPop[i_pop2].m_l_position)#.multiply(FireflyPop[i].m_l_position - FireflyPop[j].m_l_position)
                        a_dij = np.multiply(a_dist, a_dist)
                        f_dij = np.sum(a_dij)
                        f_dij = np.sqrt(f_dij)
                        #f_dij = abs(l_fireFlyPop[i_pop2].m_f_cost - l_fireFlyPop[i_pop].m_f_cost)
                        f_beta = f_beta_0 * (mth.exp(-f_gamma * mth.pow (f_dij, f_m)))
                        a_e_i = Helper.CreateAlphaMatrix (n_dim, f_alpha, 1, 0, 0.5, 3) 
                        a_beta_i = Helper.CreateBetaMatrix (n_dim, f_beta, 1, 0, 3)                                   
                        o_newFireFly.m_l_position = l_fireFlyPop[i_pop].m_l_position + np.multiply(a_beta_i, a_dist) +  a_e_i                          
                        #b_updated = True
                        # check whether the fireflies are within the given range                    
                        for i_dim in range(n_dim):
                            if o_newFireFly.m_l_position[i_dim] < f_lb:
                                o_newFireFly.m_l_position[i_dim] = f_lb
                        for i_dim in range(n_dim):
                            if o_newFireFly.m_l_position[i_dim] > f_ub:
                                o_newFireFly.m_l_position[i_dim] = f_ub 
                #if b_updated == True:                      
                o_newFireFly.m_f_cost = h_costFunction(o_newFireFly.m_l_position) #find cost
                if o_newFireFly.m_f_cost > l_fireFlyPop[i_pop].m_f_cost: 
                    l_fireFlyPop[i_pop].m_l_position = o_newFireFly.m_l_position
                    l_fireFlyPop[i_pop].m_f_cost = o_newFireFly.m_f_cost  
            #outside the outer for loop - this is for Nth firefly to move randomly
 
            l_fireFlyPop = sorted(l_fireFlyPop, key=lambda firefly: firefly.m_f_cost, reverse=True)# sort 2*nPop fireflies
            l_fireFlyPop = l_fireFlyPop[:n_pop] #select best nPop out of 2*nPop
            l_bestCost[i_gen] = l_fireFlyPop[0].m_f_cost
            #update best solution position and cost
            if l_fireFlyPop[0].m_f_cost >= o_bestSol.m_f_cost:
                o_bestSol.m_l_position =   l_fireFlyPop[0].m_l_position
                o_bestSol.m_f_cost = l_fireFlyPop[0].m_f_cost
            print(f'Generation {i_gen+1} Best Cost =  {l_bestCost[i_gen]} and X[] = {l_fireFlyPop[0].m_l_position}')  
            print(f'***Best Across Genrations**** {o_bestSol.m_f_cost}')
        results = Results()
        results.Create(o_bestSol.m_l_position, o_bestSol.m_f_cost)

        print('Ending FA ...')
        return results

def main1():
    global g_n_eval
    g_n_eval = 0
    problem = Problem()
    RandomBufferGenerator.Generate()
    problem.Create(Helper.fitness_calculation_f1, 2, 0, mth.pi)
    params = Params()
    params.Create(10, 40, 0.2, 1)
    FA = FireFlyAlgorithm()
    results = FA.RunFA(problem, params)
        # Get Results
    print(f'Individual with best fitness: {results.m_l_bestSol}')
    print(f'Best fitness value: {results.m_f_bestCost}')
    print(f'Number of Evaluations: {g_n_eval}')

if __name__ == "__main__":
    main1()