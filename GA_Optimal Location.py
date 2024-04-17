import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/USER/Downloads/latitude and longitude.csv")
campus_bounds = {'min_x': data['x'].min(),'max_x': data['x'].max(),'min_y': data['y'].min(),'max_y': data['y'].max()}


class iiserb():

    def __init__(self, buildings_df, campus_bounds):
        self.buildings_df = buildings_df
        
        self.campus_bounds = campus_bounds

    def euclidean_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def initialize_population(self, pop_size):
        initial_population = []
        for _ in range(pop_size):
            x = np.random.uniform(self.campus_bounds['min_x'], self.campus_bounds['max_x'])
            y = np.random.uniform(self.campus_bounds['min_y'], self.campus_bounds['max_y'])
            initial_population.append([x, y])
        return np.array(initial_population)

    def calculate_fitness(self, hospital_location):
        distances = []
        for _, building in self.buildings_df.iterrows():
            dist = self.euclidean_distance(hospital_location[0], hospital_location[1], building['x'], building['y'])
            distances.append(dist)
        return np.mean(distances)

    def selection_roulette_wheel(self, population, fitness_scores):
        probabilities = 1 / (fitness_scores + 1e-6)
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(np.arange(len(population)), size=len(population), replace=True, p=probabilities)
        return population[selected_indices]

    def crossover(self, parents, crossover_rate):
        children = []
        for i in range(0, len(parents), 2):
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(2)
                child1 = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
                child2 = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
                children.append(child1)
                children.append(child2)
            else:
                children.append(parents[i])
                children.append(parents[i+1])
        return np.array(children)

    def mutation(self, population, mutation_rate):
        for i in range(len(population)):
            if np.random.rand() < mutation_rate:
                population[i] += np.random.normal(0, 0.001, size=2)  
        return population

    def optimize_hospital_location(self, pop_size=100, num_generations=100, crossover_rate=0.8, mutation_rate=0.01):
        population = self.initialize_population(pop_size)
        for gen in range(num_generations):
            fitness_scores = np.zeros(pop_size)
            for i, hospital_location in enumerate(population):
                fitness_scores[i] = self.calculate_fitness(hospital_location)
            population = self.selection_roulette_wheel(population, fitness_scores)
            population = self.crossover(population, crossover_rate)
            population = self.mutation(population, mutation_rate)
        best_index = np.argmin(fitness_scores)
        best_hospital_location = population[best_index]
        return best_hospital_location

iiserb_optimization = iiserb(buildings_df=data, campus_bounds=campus_bounds)
optimal_location = iiserb_optimization.optimize_hospital_location()
print("Optimal hospital location:", optimal_location)
