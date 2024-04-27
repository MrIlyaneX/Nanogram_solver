from typing import List, Tuple


def parser(path: str) -> Tuple[int, int, List[List[int]], List[List[int]]]:
    def find(from_, to_):
        start = text.find(from_)
        end = text.find(to_, start)
        return text[start:end]

    with open(path) as data:
        text = data.read()

    width = int(''.join(i for i in find('width', '\n') if i.isdigit()))
    height = int(''.join(i for i in find('height', '\n') if i.isdigit()))
    # print(width)
    # print(height)

    col_clues = find('columns', 'rows').split('\n')[1:-2]
    col_clues = [list(map(int, i.split(','))) for i in col_clues]
    # print(col_clues)

    row_clues = find('rows', 'goal').split('\n')[1:-1]
    row_clues = [list(map(int, i.split(','))) for i in row_clues]
    # print(row_clues)

    return (width, height, row_clues, col_clues)


# to read all nonograms from folder
def get_nonograms(path: str):
    pass


if __name__ == '__main__':
    path_ = "100.non"
    print(parser(path_))
