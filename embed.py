import numpy as np
import os
import torch
import tqdm
import data

from cmmd import conditional_operator, RFF



### Constants

DATA_DIR = './players/'



### Player Operators

def get_states_and_actions(path):
    """ Format game data into state and action matrices for each player """
    X, y = data.get_data(path)
    X, y = X.reshape(X.shape[0], -1), y.reshape(y.shape[0], -1)
    print(X.shape, y.shape)
    return X, y


def get_player_operators():
    player_operators = []
    states_rff, actions_rff = None, None

    for name in tqdm.tqdm(sorted(os.listdir(DATA_DIR))):
        path = os.path.join(DATA_DIR, name)
        if not path.endswith('.pgn'):
            continue

        try:
            print(path)
            game_states, actions = get_states_and_actions(path)
        except Exception as e:
            print('Error:', path, e)
            continue

        if states_rff is None:
            states_rff = RFF(game_states.shape[-1])
        if actions_rff is None:
            actions_rff = RFF(actions.shape[-1])

        C = conditional_operator(game_states, actions, states_rff, actions_rff, alpha=1)
        player_operators.append(C)


    player_operators = np.array(player_operators)
    np.save('embeddings.npy', player_operators)

    return player_operators



get_player_operators()


