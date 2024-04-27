from typing import List, Tuple, Dict
from copy import deepcopy
import os


def parser(path: str) -> Tuple[int, int, List[List[int]], List[List[int]]]:
    """

    Parses a file.
    Similar file structure needed:

    (example)
    /file beginning
        width 3
        height 3

        columns
        4
        6
        2,2

        rows
        14
        17
        21

    /file end

    :param path: path to file
    :return: (width, height, row_clues, col_clues) of given input file
    """

    def find(from_: str, to_: str) -> str:
        start = text.find(from_)
        end = text.find(to_, start)
        return text[start:end]

    with open(path) as data:
        text = data.read()

    width = int(''.join(i for i in find('width', '\n') if i.isdigit()))
    height = int(''.join(i for i in find('height', '\n') if i.isdigit()))
    # print(width)
    # print(height)

    col_clues_str: List[str] = find('columns', 'rows').split('\n')[1:-2]
    col_clues: List[List[int]] = [list(map(int, i.split(','))) for i in col_clues_str]
    # print(col_clues)

    row_clues_str: List[str] = find('rows', 'goal').split('\n')[1:-1]
    row_clues: List[List[int]] = [list(map(int, i.split(','))) for i in row_clues_str]
    # print(row_clues)

    return width, height, row_clues, col_clues


class NonogramLoader:
    def __init__(self, folder_path: str) -> None:
        """

        :param folder_path: path to folder with .non files
        """

        self.nonograms_data: List = []

        if not os.path.exists(folder_path):
            raise Exception(f"Folder with path {folder_path} does not exists!\n")

        files: List[str] = [f for f in os.listdir(folder_path) if f.endswith(".non")]

        folder_path_ = folder_path
        if folder_path[-1] != "/":
            folder_path_ += "/"

        for file in files:
            file_path = folder_path_ + file
            cols, rows, row_clues, col_clues = parser(file_path)

            self.nonograms_data.append({
                "rows": rows,
                "cols": cols,
                "row_clues": row_clues,
                "col_clues": col_clues,
                "name": file
            })

    def __len__(self) -> int:
        return len(self.nonograms_data)

    def __getitem__(self, item) -> Dict:
        return self.nonograms_data[item]


def plot_nonogram(nonogram_sol: List[List[int]],
                  fig_size: Tuple[int, int] = (4, 4),
                  cmap: str = "bone",
                  name: str = "Nonogram"
                  ) -> None:
    """
    Creates an image of nonogram using matplotlib.plt.pcolor function.

    :param nonogram_sol: nonogram to plot
    :param fig_size: plt parameter
    :param cmap: plt parameter
    :param name: name to show on image
    :return:
    """

    import matplotlib.pyplot as plt

    print(name)

    to_show = deepcopy(nonogram_sol)
    to_show.reverse()

    plt.figure(figsize=fig_size)
    plt.axis("off")

    plt.pcolor(to_show, cmap=cmap)
    plt.title(name)
    plt.show()


def plot_fitness(fitness_change: List[float],
                 plot_size: Tuple[float, float] = (16, 9),
                 title: str = "Fitness change of best Gen over generations"
                 ) -> None:
    """
    Plot fitness change over generations

    :param fitness_change:
    :param plot_size:
    :param title:
    :return:
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=plot_size)
    plt.plot(fitness_change, label="Fitness change")
    plt.xlabel("Generation number")
    plt.ylabel("Fitness score")
    plt.title(title)
    plt.legend(loc="best")
    plt.show()
