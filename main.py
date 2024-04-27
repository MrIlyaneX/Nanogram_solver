# Use "pypy3 -mpip install -r requirements.txt" to run with modified PyPy (for plots)

def main():
    """

    :return:
    """
    from time import time
    from nonograms.utils import NonogramLoader
    from nonograms.NonogramSolverGA import NonogramSolverGA

    # best ones for now
    mutation_rate: float = 0.9
    population_size: int = 1000
    generations: int = 1500

    elitism_parameter: float = 0.05
    not_improving_fitness_early_stopping: int = 100
    fitness_return: bool = True

    plotting: bool = True  # turn False to use with standard PyPY

    path: str = "db/col_row"
    loader = NonogramLoader(path)

    for item in loader:
        start: float = time()

        solver = NonogramSolverGA(rows=item["rows"],
                                  columns=item["cols"],
                                  row_clues=item["row_clues"],
                                  col_clues=item["col_clues"],
                                  population_size=population_size,
                                  mutation_rate=mutation_rate)

        solution, fitness, fitness_change = solver.evolve(
            generations=generations,
            elitism_parameter=elitism_parameter,
            fitness_return=fitness_return,
            not_improving_fitness_early_stopping=not_improving_fitness_early_stopping
        )

        print(f"best: {fitness}, {(time() - start):2f}")
        print("-------------------------")

        if plotting:
            from nonograms.utils import plot_nonogram, plot_fitness
            plot_nonogram(solution,
                          name=f"fitness:{fitness}\nname: {item['name']}\nsize: {item['rows']} x {item['cols']}")

            plot_fitness(fitness_change,
                         title=f"fitness:{fitness}, p_size: {population_size}, m_rate: {mutation_rate}  ")
        else:
            for i in solution:
                for j in range(len(i)):
                    if i[j] == 1:
                        print('x', end=' ')
                    else:
                        print(' ', end=' ')
                print()
            print("-------------------------")
        break


if __name__ == '__main__':
    main()
