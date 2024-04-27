import random
from typing import List


class NonogramSolverGA:
    def __init__(
            self,
            rows: int,
            columns: int,
            row_clues: List[List[int]],
            col_clues: List[List[int]],
            population_size: int = 500,
            mutation_rate: float = 0.2
    ) -> None:
        """
        This class represents a Genetic Algorithm that solves nonograms
        (if fitness reaches 0, then nonogram solved correctly)

        Assumptions during solving process is that algorithm guaranties rows correspondence over row_clues.
        The need is to validly place the pieces to satisfy the col_clues.

        :param rows: number of rows in nonogram
        :param columns: number of columns in nonogram

        :param row_clues: given row clues (order matters), example list: [[3], [3, 4], [6], [4, 4], [2, 2], [2, 2], [6, 2], [3, 5], [2, 3], [5]]

        :param col_clues: given col clues (order matters), example list: [[3], [3, 4], [6], [4, 4], [2, 2], [2, 2], [6, 2], [3, 5], [2, 3], [5]]

        :param population_size: Number of individuals in generation
        :param mutation_rate: Chance of mutation
        """

        self.rows = rows
        self.columns = columns
        self.row_clues = row_clues
        self.col_clues = col_clues
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.positions = [[]]

    def __generate_individual(
            self
    ) -> List[List[int]]:
        """
        Assumption: Guarantees the validity over row_clues

        :return: Returns the list of lists
        """
        grid = []
        for i in range(self.rows):
            row = [0 for _ in range(self.columns)]
            i_position = []
            cursor = 0

            common_length = sum(self.row_clues[i])
            amount = len(self.row_clues[i])  # amount of strings
            n = amount
            for j in range(amount):
                index = random.randint(cursor, (self.columns - common_length - n + 1))
                i_position.append(index)
                for k in range(self.row_clues[i][j]):
                    row[k + index] = 1
                common_length -= self.row_clues[i][j]
                cursor = (index + self.row_clues[i][j] + 1)
                n -= 1
            self.positions.append(i_position)
            grid.append(row)
        return grid

    def __get_column_indexes(
            self,
            individual: List[List[int]]
    ) -> List[List[int]]:
        """

        :param individual:
        :return:
        """

        return [[individual[j][i] for j in range(self.rows)] for i in range(self.columns)]

    def __fitness_im(
            self,
            individual: List[List[int]]
    ) -> int:
        """
        Calculating fitness of individual. Best possible fitness is 0.

        :param individual:
        :return:
        """
        grid = self.__get_column_indexes(individual)
        score = 0

        for i, clue in enumerate(self.col_clues):
            column = grid[i]
            score += abs(sum(column) - sum(clue))

            str_column = [non_empty for non_empty in "".join(str(ch) for ch in column).rsplit("0") if non_empty]
            n_column = len(str_column)

            score += abs(len(clue) - n_column)

            for j in range(min(n_column, len(clue))):
                score += abs(len(str_column[j]) - clue[j])

        return score

    @staticmethod
    def __crossover(
            parent1: List[List[int]],
            parent2: List[List[int]]
    ) -> List[List[int]]:
        """
        Performs single-point crossover operation, that produces only valid 'child' individual.

        :param parent1:
        :param parent2:
        :return:
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def __mutate(
            self,
            individual: List[List[int]]
    ) -> List[List[int]]:
        """
        Performs mutation operation by changing the position of pieces randomly in the randomly chosen row.

        :param individual:
        :return:
        """
        positions = []
        for row in individual:
            position_i = []
            if row[0] == 1:
                position_i.append(0)
            for j in range(len(row) - 1):
                if row[j] == 0 and row[j + 1] == 1:
                    position_i.append(j + 1)
            positions.append(position_i)

        initial_index = new_index = -1
        previous_allowed_index = next_allowed_index = -1

        while initial_index == new_index or previous_allowed_index == next_allowed_index:
            rand_row_index = random.randint(0, self.rows - 1)  # random row

            if sum(self.row_clues[rand_row_index]) + len(self.row_clues[rand_row_index]) - 1 == self.columns:
                continue

            n = len(self.row_clues[rand_row_index])
            rand_word = random.randint(0, n - 1)  # random word in the row
            initial_index = positions[rand_row_index][rand_word]

            if rand_word == 0:  # first word in the row
                previous_allowed_index = 0
            else:
                previous_allowed_index = positions[rand_row_index][rand_word - 1] + self.row_clues[rand_row_index][
                    rand_word - 1] + 1

            if rand_word == n - 1:  # last word in the row
                next_allowed_index = self.columns - self.row_clues[rand_row_index][rand_word]
            else:
                next_allowed_index = positions[rand_row_index][rand_word + 1] - self.row_clues[rand_row_index][
                    rand_word] - 1

            if next_allowed_index == previous_allowed_index:
                continue

            new_index = random.randint(previous_allowed_index, next_allowed_index - 1)

        positions[rand_row_index].remove(initial_index)
        positions[rand_row_index].append(new_index)
        positions[rand_row_index].sort()
        individual[rand_row_index] = []

        line = [0 for _ in range(self.columns)]
        for i in range(n):
            start_position = positions[rand_row_index][i]
            for j in range(self.row_clues[rand_row_index][i]):
                line[start_position + j] = 1
        individual[rand_row_index] = line
        return individual

    def evolve(
            self,
            generations: int = 1000,
            elitism_parameter: float = 0.2,
            not_improving_fitness_early_stopping: int = -1,
            fitness_return: bool = False
    ) -> tuple[list[list[int]], int, list[list[int]]] | tuple[list[list[int]], int]:
        """
        Starts run of GA

        :param generations: Number of generations to perform
        :param elitism_parameter:
        :param not_improving_fitness_early_stopping: natural number, after reaching this number of non-improving fitness
         algorithm will finish it's work
        :param fitness_return:
        :return: best_ind - best individual (list of lists of 0s and 1s),
         best_fit - best fitness (number),
          (optionally) fitness_change - change of fitness
          over generations (only the best individual in each generations considered)
        """

        fitness_change: List = []
        not_improving_counter: int = 0

        population = [self.__generate_individual() for _ in range(self.population_size)]

        # Evaluate fitness for each individual
        fitness_scores = [(individual, self.__fitness_im(individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=False)  # descending order
        fitness_change.append([fitness_scores[0][1]])

        last_fitness = fitness_scores[-1][1]

        for _ in range(generations):
            # Stopping if the solution found
            if fitness_scores[0][1] == 0:
                break

            if not_improving_counter == not_improving_fitness_early_stopping:
                print("Early stopping")
                break

            # Select top individuals for reproduction (elitism)
            elite = [individual for individual, _ in fitness_scores[:int(self.population_size * elitism_parameter)]]

            # Generate offspring through crossover and mutation
            offspring = []
            while len(offspring) < self.population_size - len(elite):
                parent1, parent2 = random.choices(population, k=2)
                child = self.__crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.__mutate(child)
                offspring.append(child)

            population = elite + offspring

            # Evaluate fitness for each individual
            fitness_scores = [(individual, self.__fitness_im(individual)) for individual in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=False)  # descending order

            # update fitness change
            fitness_change.append([fitness_scores[0][1]])

            # update counter if we are on plato
            not_improving_counter = not_improving_counter + 1 if last_fitness == fitness_change[-1] else 0

            # update fitness from "last" iteration
            last_fitness = fitness_change[-1]

        # Return the best solution found
        best_fit = fitness_scores[0][1]
        best_ind = fitness_scores[0][0]
        for i in range(self.population_size):
            if self.__fitness_im(population[i]) <= best_fit:
                best_ind = population[i]
                best_fit = self.__fitness_im(population[i])

        if fitness_return:
            return best_ind, best_fit, fitness_change
        return best_ind, best_fit
