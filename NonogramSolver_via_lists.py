import random
import numpy as np

class NonogramSolverGA:
    def __init__(self, rows, columns, row_clues, col_clues, population_size=200, mutation_rate=0.2):
        self.rows = rows
        self.columns = columns
        self.row_clues = row_clues
        self.col_clues = col_clues
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.positions = [[]]

    def generate_individual(self) -> list:
        ### Guarantees the following all row clues!!! ###
        # Returns the list of lists
        grid = []
        for i in range (self.columns):
            row = [0 for _ in range (self.columns)]
            i_position = []
            cursor = 0
            common_length = sum(self.row_clues[i])
            amount = len(self.row_clues[i]) # amount of strings
            n = amount
            for j in range(amount):
                index = np.random.randint(cursor, (rows - common_length - n + 2))
                i_position.append(index)
                for k in range(self.row_clues[i][j]):
                    row[k + index] = 1
                common_length -= self.row_clues[i][j]
                cursor = (index + self.row_clues[i][j] + 1) 
                n -= 1
            self.positions.append(i_position)
            grid.append(row)
        return grid
    

    def get_column_indexes(self, individual):
        grid = []
        for i in range (self.columns):
            column = []
            for j in range (self.rows):
                column.append(individual[j][i])
            grid.append(column)
        return grid
    
    def fitness(self, individual):
        grid = self.get_column_indexes(individual)
        positions = []
        score = 0

#-----------------------------------------------------------------------------
        # for i, clue in enumerate(self.col_clues):
        #     column = grid[i]

        #     n_clue = len(clue) # needed number of words
        #     n_column = 0 # actual number of words
        #     word = False
        #     start = 0
            # for j in range(self.columns):
            #     if column[j] == 1 and not word:
            #         n_column += 1
            #         start = j
            #         word = True
            #     if column[j] == 0 and word:
            #         positions.append((start, j))
            #         word = False
            # if word:
            #     positions.append((start, self.columns))
            
        #     if n_clue != n_column:
        #         score += 1000 * abs(n_clue - n_column)
            
        #     score += abs(sum(clue) - sum(column))

            # for j in range (min(n_column, n_clue)):
            #     score += abs(positions[j][1] - positions[j][0] - clue[j])

#--------------------------------------------------------------------------------
        for i, clue in enumerate(self.col_clues):
            column = grid[i]
            score += abs(sum(column) - sum(clue))
            positions = []

            n_column = 0
            for j in range(len(column) - 1):
                if j == 0 and column[j] == 1:
                    n_column += 1
                if column[j] == 0 and column[j+1] == 1:
                    n_column += 1

            score += abs(len(clue) - n_column)

            start = -1
            word = False
            for j in range(self.columns):
                if column[j] == 1 and not word:
                    start = j
                    word = True
                if column[j] == 0 and word:
                    positions.append((start, j))
                    word = False
            if word:
                positions.append((start, self.columns))
            for j in range (min(n_column, len(clue))):
                score += abs(positions[j][1] - positions[j][0] - clue[j])
        
        return score

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, individual):
        positions = []
        for row in individual:
            position_i = []
            if row[0] == 1:
                position_i.append(0)
            for j in range(len(row) - 1):
                if row[j] == 0 and row [j + 1] == 1:
                    position_i.append(j + 1)
            positions.append(position_i)

        initial_index = new_index = -1
        previous_allowed_index = next_allowed_index = -1

        while initial_index == new_index or previous_allowed_index == next_allowed_index:
            rand_row_index = random.randint(0, self.rows - 1) # random row

            if sum(self.row_clues[rand_row_index]) + len(self.row_clues[rand_row_index]) - 1 == self.columns:
                continue
            
            n = len(self.row_clues[rand_row_index])
            rand_word = random.randint(0, n - 1) # random word in the row
            initial_index = positions[rand_row_index][rand_word]

            if rand_word == 0: # first word in the row
                previous_allowed_index = 0
            else:
                previous_allowed_index = positions[rand_row_index][rand_word - 1] + self.row_clues[rand_row_index][rand_word - 1] + 1

            if rand_word == n - 1: # last word in the row 
                next_allowed_index = self.columns - self.row_clues[rand_row_index][rand_word]
            else:
                next_allowed_index = positions[rand_row_index][rand_word + 1] - self.row_clues[rand_row_index][rand_word] - 1

            if next_allowed_index == previous_allowed_index:
                continue
            
            new_index = np.random.randint(previous_allowed_index, next_allowed_index)

        positions[rand_row_index].remove(initial_index)
        positions[rand_row_index].append(new_index)
        positions[rand_row_index].sort()
        individual[rand_row_index] = []
            
        l = [0 for _ in range (self.columns)]
        for i in range(n):
            start_position = positions[rand_row_index][i]
            for j in range (row_clues[rand_row_index][i]):
                l[start_position + j] = 1
        individual[rand_row_index] = l
        return individual


    def evolve(self, generations):
        ## Individual - list of lists with 0s and 1s:
        ## [[1, 0, 1], [1, 1, 1], [0, 1, 0]]

        ## Population - list of individuals

        ## self.row/column_clues - list: [[3], [3, 4], [6], [4, 4], [2, 2], [2, 2], [6, 2], [3, 5], [2, 3], [5]]


        population = [self.generate_individual() for _ in range(self.population_size)]
        for _ in range(generations):
            # Evaluate fitness for each individual
            fitness_scores = [(individual, self.fitness(individual)) for individual in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=False) # descending order

            # Select top individuals for reproduction (elitism)
            elite = [individual for individual, _ in fitness_scores[:self.population_size // 5]]

            # Generate offspring through crossover and mutation
            offspring = []
            while len(offspring) < self.population_size - len(elite):
                parent1, parent2 = random.choices(population, k=2)
                child = self.crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                offspring.append(child)

            population = elite + offspring

        # Return the best solution found
        best_fit = 100000
        best_ind = None
        for i in range (self.population_size):
            if self.fitness(population[i]) <= best_fit:
                best_ind = population[i]
                best_fit = self.fitness(population[i])

        print('BEST FITNES', best_fit)
        # return max(population, key=lambda x: self.fitness(x))
        return best_ind




# Example usage:
rows = 15
columns = 15
generations = 600
population_size = 200
# col_clues = [[3], [3, 4], [6], [4, 4], [2, 2], [2, 2], [6, 2], [3, 5], [2, 3], [5]]
# row_clues = [[2, 2], [1, 2, 3], [3, 2], [3, 1, 1], [1, 2, 1], [1, 1, 5], [10], [5, 3], [1, 1, 3], [1, 1, 2]]


# col_clues = [[6,2], [1,3,1], [4,2], [3,3], [6], [2], [1], [1], [7], [1]]
# row_clues = [[2, 1], [1,1,1], [1,2,1], [4,1], [5,1], [2,6], [2,1], [5], [1,3], [2]]

col_clues = [[12], [14],[4,7], [3,6], [5, 9],
             [10,4], [9,3],[3,2,2],[1,4,2],[2,1],
             [4,1],[6,2],[4,3],[11],[6]]
row_clues = [[5],[7],[9],[3,4,1],[2,3,1,2],
             [2,2,1,5],[2,11],[2,11],[3,3,1,5],[6,1,2],
             [5,1],[6,1],[7,2],[9,3],[12]]

# col_clues = [[9],[12],[3,6],[2,2,5],[1,5,4],
#              [1,8,4],[13],[7,5],[7,5],[1,7,4],
#              [1,3,2,1],[2,2,3,1],[3,3,1],[8,2],[4,3]]
# row_clues = [[3,3],[2,2],[1,4,1],[2,6,2],[1,7,1],
#              [1,8,1],[2,9,2],[2,6,1,2],[2,5,2],[3,2,1,3],
#              [4,3,3],[13],[12,1],[10,2],[15]]

# col_clues = [[3,11],[15],[3,11],[9,1],[4],
#              [3,3],[2,5],[1,3,9],[15],[3,6],
#              [2,2,2,5],[3,11],[4,3,4],[3,5,1],[11]]
# row_clues = [[4,2,4],[4,1,4],[4,2,3],[1,1,2,1],[4,5,2],
#              [5,4,2],[6,3,4],[9,5],[9,5],[3,3,1,1],
#              [3,6,1],[3,8,1],[3,8,1],[3,8,1],[4,9]]



solver = NonogramSolverGA(rows, columns, row_clues, col_clues, population_size)
solution = solver.evolve(generations)

for i in solution:
    for j in range (len(i)):
        if i[j] == 1:
            print('x', end=' ')
        else:
            print(' ', end=' ')
    print()
