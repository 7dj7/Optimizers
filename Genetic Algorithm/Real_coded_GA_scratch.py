####*************************************####
####    REAL CODED GENETIC ALGORITHM     ####
####*************************************####

# NOTES:
# This code does take into account constraint handling
# Availabe methods for:
# Selection: Tournament
# Pairing: Random
# Crossover: SinglePoint
# Mutation: Average, Random 

import numpy as np
from random import random as rnd
import random
import copy

## CREATION OF AN INDIVIDUAL
def individual(number_of_genes, upper_limit, lower_limit):
    individual= [round(rnd()*(upper_limit-lower_limit)+lower_limit,1) for x in range(number_of_genes)]
    return individual

## CREATION OF INITIAL POPULATION (GROUP OF INDIVIDUALS)
def population(number_of_individuals, number_of_genes, upper_limit, lower_limit):
    return [individual(number_of_genes, upper_limit, lower_limit) for x in range(number_of_individuals)]

## FITNESS EVALUATION OF AN INDIVIDUAL 
"""(CHANGE/ENTER THE OBJECTIVE FUNCTION HERE)"""
def fitness_calculation(individual):
    fitness_value = -((1.5 - individual[0] + individual[0]*individual[1])**2 + 
    (2.25 - individual[0] + individual[0]*individual[1]**2)**2 +
    (2.625-individual[0]+individual[0]*individual[1]**3)**2)      #BEALE FUNCTION----minimisation
    return fitness_value

    #fitness_value= -(individual[0]**2 + individual[1]**2 + individual[2]**2 + individual[3]**2)      #SPHERE FUNCTION----minimisation
    #return fitness_value

## GETIING THE INDIVIDUAL WITH BEST FITNESS
def get_best(ind_list):
    fit_values= []
    for i in range(len(ind_list)):
        fit_values.append(fitness_calculation(ind_list[i]))
    pos= fit_values.index(max(fit_values))
    return (ind_list[pos],fit_values[pos])

## GETTING FITNESS VALUES OF A GROUP OF INDIVIDUALS
def get_fits(ind_list):
    fit_values=[]
    for i in ind_list:
        fit_values.append(fitness_calculation(i))
    return fit_values

## SELECTION FROM POPULATION TO GENERATE MATING POOL
def selection(population, Method= 'Tournament'):
    pop_length= len(population)
    selected_individuals= []
    selected_fitness= []
    if Method == 'Tournament':
        """CHANGE THE TOURNAMENT SIZE HERE"""
        tour_size= 20              
        for i in range(pop_length):
            ind_list= random.sample(population, k= tour_size)
            (x,y)= get_best(ind_list)
            selected_individuals.append(x)
            selected_fitness.append(y)
    selected= {'Selected_Individuals':selected_individuals,'Selected_Fitness':selected_fitness}
    return selected

## CHOOSE TWO PARENTS THAT MAY PARTICIPATE IN CROSSOVER/ CHOOSE A PAIR OF INDIVIDUALS
def get_pair(pop):
    parents= random.sample(pop, k=2)
    return parents

## CROSSOVER OF THE PARENTS IN THE MATING POOL
def crossover(selected, pc, Method= 'SinglePoint'):
    children=[]
    counter=0
    pop_length= len(selected['Selected_Individuals'])
    number_of_genes= len(selected['Selected_Individuals'][0])
    if Method == 'SinglePoint':
        while counter<pop_length//2:
            parents= get_pair(selected['Selected_Individuals'])
            r= round(rnd(),1)
            if r<=pc:
                pos= random.randint(1,number_of_genes-1)
                temp= copy.deepcopy(parents[0])
                parents[0][pos:number_of_genes] = parents[1][pos:number_of_genes]
                parents[1][pos:number_of_genes] = temp[pos:number_of_genes]
                children.append(parents[0])
                children.append(parents[1])
                counter+=1
    return children

## MUTATION OF THE CHILDREN
def mutation(child, pm, Method= 'Average'):
    pop_length= len(child)
    number_of_genes= len(child[0])
    mutated= copy.deepcopy(child)
    if Method== 'Average':
        for x in range(pop_length):
            for y in range(number_of_genes):
                r= round(random.uniform(0,0.009),3)
                if r<=pm:
                    rm= random.uniform(-5,5)
                    mutated[x][y]= (mutated[x][y] + rm)/2
    if Method== 'Random':
        """CHANGE THE DELTA VALUE HERE"""
        delta= -1      
        for x in range(pop_length):
            for y in range(number_of_genes):
                r= round(random.uniform(0,0.009),3)     # CHANGE MUTATION RANDOM NUMBER FORMAT HERE 
                if r<=pm:
                    rm= random.uniform(0,1)
                    mutated[x][y]= mutated[x][y] + (rm-0.5)*delta
    return mutated

## CREATION OF NEW GENERATION
def new_gen(pop, children, mutated):
    complete= pop + children + mutated
    number_of_genes= len(pop[0])
    pop_length= len(pop)
    fit_values= get_fits(complete)
    n= len(complete)
    complete= np.asarray(complete)
    fit_values= np.asarray(fit_values)
    fit_values= fit_values.reshape((n,1))
    temp= np.append(complete,fit_values,axis=1)
    temp= temp[temp[:,number_of_genes].argsort()]
    temp= np.delete(temp,number_of_genes,1)
    temp= temp.tolist()
    newgen= temp[-pop_length:-1]
    newgen.append(temp[-1])
    return newgen

#######################
#    MAIN FUNCTION    #
#######################

pop_size= 26            # CHANGE POPULATION SIZE HERE
number_of_genes= 2      # CHANGE NUMBER OF GENES HERE
pc= 0.8                 # CHANGE Pc HERE
pm= 0.003               # CHANGE Pm HERE
gen= 10000                # CHANGE NUMBER OF GENERATIONS HERE

init_pop= population(pop_size,number_of_genes,5,-5)

new_pop= copy.deepcopy(init_pop)

for i in range(gen):
    
    sel= selection(new_pop)
    child= crossover(sel, pc)
    mut= mutation(child, pm)
    newgen= new_gen(new_pop, child, mut)
    
    (ind,fit)= get_best(newgen)
    print(f'Generation number: {i+1}')
    print(f'Individual with best fitness: {ind}')
    print(f'Best fitness value: {fit}')
    print('\n'*2)

    new_pop= copy.deepcopy(newgen)