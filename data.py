import chess.pgn
import numpy as np

from collections import Counter



CHESS_DICT = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1,0],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0,1],
}



def load_games(path):
    pgn = open(path)
    games, sides = [], []
    length = 1000
    for i in range(length):
        try:
            if chess.pgn.read_game(pgn).mainline_moves():
                games.append(chess.pgn.read_game(pgn).mainline_moves())
                sides.append(chess.pgn.read_game(pgn).headers['White'])
        except:
            if chess.pgn.read_game(pgn) is None:
                games, sides = games[:len(sides)], sides[:len(games)] # remove extra
                break
            else:
                pass #print(i, chess.pgn.read_game(pgn))

    return games, sides


def process_games(games, sides, player_name=None):
    if player_name is None:
        player_name = Counter(sides).most_common(1)

    X, y = [], []
    counter_2 = 0
    for game in games:
        board = chess.Board()
        white = sides[counter_2]
        if white == player_name:
            remainder = 0
        else:
            remainder = 1
        counter = 0
        for move in game:
            if counter % 2 == remainder:
                X.append(board.copy())
            board.push(move)
            if counter % 2 == remainder:
                y.append(board.copy())
            counter += 1
        counter_2 += 1

    return X, y


def make_matrix(board): 
    pgn = board.epd()
    foo = []  
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []  
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append('.')
            else:
                foo2.append(thing)
        foo.append(foo2)
    return foo


def translate(matrix):
    rows = []
    for row in matrix:
        terms = []
        for term in row:
            terms.append(CHESS_DICT[term])
        rows.append(terms)
    return rows


def get_data(path, player_name=None):
    games, sides = load_games(path)
    X, y = process_games(games, sides, player_name=player_name)

    for i in range(len(X)):
        X[i] = translate(make_matrix(X[i]))
    for i in range(len(y)):
        y[i] = translate(make_matrix(y[i]))
    
    X, y = np.array(X), np.array(y)
    return X, y


