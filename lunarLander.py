"""
File name: lunarLander.py
    Agent for landing successfully the 'Lunar Lander' which is implemented in
    OpenAI gym (reference [1]).

Usage: python lunarLander.py -h

usage: lunarLander.py [-h] [-v {0,1,2}] -e {train,test} [-a A]

Lunar Lander with DQN

optional arguments:
  -h, --help       show this help message and exit
  -v {0,1,2}       verbose level (0: None, 1: INFO, 2: DEBUG)
  -e {train,test}  execute (train, test)
  -a A             trained agent file

Note: Default convergence criteria met when for 150 consecutive episodes the average reward is > 200.

usage example: Execute step 1.1 with rendering enabled and verbose level set to INFO.
               python project2.py -e 1.1 -r -v 1

Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 02.12.2019

References:
    [1] arXiv:1312.5602 [cs.LG]

Python Version: 3.6
"""

import argparse
import os
import time

# Other classes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import deepNeuralNetwork
import deepQNetwork
# My classes
import emulator as em
import sarfa_saliency

plt.switch_backend('agg')

'''
Constants
'''
C_VERBOSE_NONE = 0  # Printing is disabled
C_VERBOSE_INFO = 1  # Only information printouts (constructor)
C_VERBOSE_DEBUG = 2  # Debugging printing level (all printouts)

NUM_STATE = 8
NUM_ACTION = 4
NUM_SALIENCY_TESTS = 100
SALIENCY_PERTURBATION = 0.1
STATE_LABELS = [
    'x_pos', 'y_pos', 'x_vel', 'y_vel',
    'angle', 'ang_vel', 'left_leg', 'right_leg'
]


def executionTimeToString(execution_time, digits_precision=3):
    """
    Summary:
        Creates a printable string regarding the execution time

    Args:
        execution_time: float
            Execution time in seconds

        digits_precision: integer
            Defines the precision in the decimal digits

    Raises:
        -

    Returns:
        time_string: string
            Printable time string
            i.e. 1 hour 2 minutes 12 seconds 113 milliseconds (3732.113 seconds)

    notes:
        -
    """

    # Three decimal digits accuracy
    execution_time = round(execution_time, digits_precision)

    hours = int(execution_time / 3600)
    minutes = int((execution_time - hours * 3600) / 60)
    seconds = int(execution_time - hours * 3600 - minutes * 60)
    milliseconds = int(round(execution_time - int(execution_time), 3) * 1000)

    time_string = ''

    if hours > 1:
        time_string += str(hours) + ' hours '
    elif hours == 1:
        time_string += str(hours) + ' hour '

    if minutes > 1:
        time_string += str(minutes) + ' minutes '
    elif minutes == 1:
        time_string += str(minutes) + ' minute '

    if seconds > 1:
        time_string += str(seconds) + ' seconds '
    elif seconds == 1:
        time_string += str(seconds) + ' second '

    if milliseconds > 0:
        time_string += str(milliseconds) + ' milliseconds '

    time_string += '(' + str(execution_time) + ' seconds)'

    return time_string


def applySeed(seed, verbose):
    """
    Summary:
        Applies the given seed to the numpy and core python random functions.

    Args:
        seed: int
            Seed value.

        verbose: int
            Verbose level (0: None, 1: INFO, 2: DEBUG)

    Raises:
        -

    Returns:
        -

    notes:
        -
    """

    if verbose != C_VERBOSE_NONE:
        print('Apply Random Seed to the execution environment (seed = ', seed, ')', sep='')

    # Numpy random function
    import numpy
    numpy.random.seed(seed)

    # Python random function
    import random
    random.seed(seed)


def trial(model_file_name, scenario, number_of_trials, rendering=False, graphs_suffix='', verbose=C_VERBOSE_NONE,
          store_history=False, compute_saliency=False, history_save_path='./output/history_test.pkl'):
    """
    Summary:
        Evaluate the trained DQN for a number of trials (number_of_trials).

    Args:
        model_file_name: string
            The saved trained DQN (Keras DNN h5 file).

        scenario: string
            The OpenAI gym scenario to be loaded by the Emulator.

        number_of_trials: int
            How many trials to execute.

        rendering: boolean
            If True, OpenAI gym environment rendering is enabled.

        graphs_suffix: string
            A suffix added in the graphs file names. To be used in case of multiple trials.

        verbose: int
            Verbose level (0: None, 1: INFO, 2: DEBUG)

        store_history: bool
            Store history data or not.

        compute_saliency: bool
            Computes saliency or not.

        history_save_path: str
            Where to store the history file.

    Raises:
        -

    Returns:
        trials_average_reward: float
            The average reward for the trial-episode (100 episodes)

    notes:
        -
    """

    if verbose > C_VERBOSE_NONE:
        print('\nEvaluate the trained DQN in ', str(number_of_trials), ' trials (episodes).', sep='')
        print('- model_file_name = ', model_file_name, ', scenario = ', scenario, ', number_of_trials = ',
              number_of_trials,
              ', rendering = ', rendering, ', graphs_suffix = ', graphs_suffix, sep='')

        # Create a Emulator object instance (without a seed)
    emulator = em.Emulator(scenario=scenario, average_reward_episodes=number_of_trials, statistics=True,
                           rendering=rendering, seed=42, verbose=verbose)

    # Create a Deep Neural Network object instance and load the trained model (model_file_name)
    dnn = deepNeuralNetwork.DeepNeuralNetwork(file_name=model_file_name, verbose=verbose)

    # Start measuring Trials time
    start_time = time.time()

    history = {
        'trial': [],
        'state': [],
        'action': [],
        'reward': [],
        'next_state': [],
        'done': [],
        'q_values': []
    }
    if compute_saliency:
        history['saliency'] = []

    # Trials
    # used as baseline for perturbation
    # for each feature, apply a random noise of 0.2 * (max(feature) - min(feature))
    state_min = np.array([-0.354871, -0.10391249, -0.468456, -0.89336216, -0.15218297, -0.4017307, 0, 0])
    state_max = np.array([-0.00462484, 1.4088593, 0.12988918, 0.05392841, 0.5564749, 0.8584606, 1, 1])
    for i in range(number_of_trials):

        current_state = emulator.start()

        while emulator.emulator_started:
            q_values = dnn.predict(current_state)
            action = np.argmax(q_values)

            if compute_saliency:
                # compute saliency
                saliency = np.zeros(NUM_STATE)
                for _ in range(NUM_SALIENCY_TESTS):
                    for j in range(NUM_STATE):
                        # perturb state
                        perturbed_state = np.array(current_state)
                        if j < 6:  # numerical states
                            perturbed_state[j] = SALIENCY_PERTURBATION * np.random.rand() \
                                                 * (state_max[j] - state_min[j]) + state_min[j]
                        else:  # boolean states
                            perturbed_state = current_state.copy()
                            perturbed_state[j] = 1 - perturbed_state[j]
                        q_values_preturbed = dnn.predict(perturbed_state)

                        max_q = np.max(q_values)
                        q_values /= max_q
                        q_values_preturbed /= max_q

                        q_value_dict = {a: q_values[0, a].astype(np.float64) for a in range(4)}
                        q_value_preturbed_dict = {a: q_values_preturbed[0, a].astype(np.float64) for a in range(4)}
                        saliency[j] = sarfa_saliency.computeSaliencyUsingSarfa(action,
                                                                               q_value_dict,
                                                                               q_value_preturbed_dict)[0]
                saliency /= NUM_SALIENCY_TESTS

            # Experience [s, a, r, s']
            experience = emulator.applyAction(action)

            # save data
            if store_history:
                history['trial'].append(i)
                history['state'].append(current_state)
                history['action'].append(action)
                history['reward'].append(experience[2])
                if experience[3] is not None:
                    history['next_state'].append(experience[3])
                    history['done'].append(False)
                else:
                    history['next_state'].append(current_state)
                    history['done'].append(True)
                history['q_values'].append(q_values)
                if compute_saliency:
                    history['saliency'].append(saliency)

            current_state = experience[3]

    if store_history:
        for k in history.keys():
            history[k] = np.array(history[k])
        history_save_dir = os.path.split(history_save_path)[0]
        if not os.path.exists(history_save_dir):
            os.makedirs(history_save_dir)
        pd.to_pickle(history, history_save_path)

    if verbose > C_VERBOSE_NONE:
        print('\nDQN ', str(number_of_trials), ' trials average = ', emulator.execution_statistics.values[-1, 3],
              ', in ',
              executionTimeToString(time.time() - start_time), sep='')

    return emulator.execution_statistics.values[-1, 3]


def train(scenario, average_reward_episodes, rendering, hidden_layers, hidden_layers_size, memory_size, minibatch_size,
          optimizer_learning_rate, gamma, epsilon_decay_factor, maximum_episodes, model_file_name,
          converge_criteria=None, graphs_suffix='', seed=None, verbose=C_VERBOSE_NONE, store_history=False,
          history_save_path='./output/history_train.pkl'):
    """
    Summary:
        Trains a DQN model for solving the given OpenAI gym scenario.

    Args:
        scenario: string
            The OpenAI gym scenario to be solved.

        average_reward_episodes: int
            On how many concecutive episodes the averaged reward should be calculated.

        rendering: boolean
            If True, OpenAI gym environment rendering is enabled.

        hidden_layers: int
            The number of hidden layers of the Deep Neural Network. Not including the first
            and last layer.

        hidden_layers_size: int
            The size of each hidden layer of the Neural Network.

        memory_size: int
            The size of the replay memory feature which will be used by the DQN.

        minibatch_size: int
            The minibatch size which will be retrieved randomly from the memory in each
            iteration in the DQN.

        optimizer_learning_rate: float
            The Adam optimizer learning rate used in the DNN.

        gamma: float
                The discount factor to be used in the equation (3) of [1].

        epsilon_decay_factor: float
            The decay factor of epsilon parameter, for each iteration step.

        maximum_episodes: int
            The maximum number of episodes to be executed. If DQN converges earlier the training stops.

        model_file_name: string
            The file in which the DQN trained model (DNN Keras) should be saved.

        converge_criteria: int or None
            The DQN converge criteria (when for converge_criteria concecutive episodes average reward
            is > 200, the DQN assumed that has been converged).
            If None, the training continues till the maximum_episodes is reached.

        graphs_suffix: string
            A suffix added in the graphs file names. To be used in case of multiple trains.

        seed: int
            Optional Seed to be used with the OpenAI gym environment, for results reproducability.

        verbose: int
            Verbose level (0: None, 1: INFO, 2: DEBUG)

        store_history: bool
            Store history or not.

        history_save_path: str
            Where to store the history file.

    Raises:
        -

    Returns:
        convergence_episode: int
            In which episode the DQN convergences

        convergence_time: string (time)
            On how much time the DQN convergences

        Rturns None if converge_criteria is None

    notes:
        -
    """

    if verbose > C_VERBOSE_NONE:
        print('\nDQN Training Starts (scenario = ', scenario, ', average_reward_episodes = ', average_reward_episodes,
              ', rendering = ', rendering,
              ', hidden_layers = ', hidden_layers, ', hidden_layers_size = ', hidden_layers_size, ', memory_size = ',
              memory_size,
              ', minibatch_size = ', minibatch_size, ', optimizer_learning_rate = ', optimizer_learning_rate,
              ', gamma = ', gamma,
              ', epsilon_decay_factor = ', epsilon_decay_factor, ', maximum_episodes = ', maximum_episodes,
              ', model_file_name = ', model_file_name,
              ', converge_criteria = ', converge_criteria, ', graphs_suffix = ', graphs_suffix, ', seed = ', seed, ')',
              sep='')

    # If seed is given the apply it
    if seed is not None:
        applySeed(seed, verbose)

    # Create a Emulator object instance
    emulator = em.Emulator(scenario, average_reward_episodes, statistics=True, rendering=rendering, seed=seed,
                           verbose=verbose)

    # Create a Deep Neural Network object instance (Keras with Tensor Flow backend)
    dnn = deepNeuralNetwork.DeepNeuralNetwork(inputs=emulator.state_size, outputs=emulator.actions_number,
                                              hidden_layers=hidden_layers,
                                              hidden_layers_size=hidden_layers_size,
                                              optimizer_learning_rate=optimizer_learning_rate, seed=seed,
                                              verbose=verbose)

    # Create a DQN object instance (we start always from epsilon = 1.0, we control each value with the
    # epsilon_decay_factor
    dqn = deepQNetwork.DeepQNetwork(emulator=emulator, dnn=dnn, states_size=emulator.state_size,
                                    actions_number=emulator.actions_number,
                                    memory_size=memory_size, minibatch_size=minibatch_size, gamma=gamma, epsilon=1.0,
                                    epsilon_decay_factor=epsilon_decay_factor,
                                    seed=seed, verbose=verbose)

    # Start measuring training time
    start_time = time.time()

    history = {
        'trial': [],
        'state': [],
        'action': [],
        'reward': [],
        'next_state': [],
        'done': [],
        'q_values': []
    }

    if converge_criteria is not None:
        # Holds how many concecutive episodes average reward is > 200
        convergence_counter = 0
        episodes_convergence_counter = []  # Holds the convergence_counter for all episodes
        convergence_episode = 0

    # Training starts here
    for i in range(maximum_episodes):
        current_state = emulator.start()

        # See Algorithm 1 in [1]
        while emulator.emulator_started:
            q_values = dnn.predict(current_state)
            action = np.argmax(q_values)

            # Experience [s, a, r, s']
            experience = emulator.applyAction(action)

            # save data
            if store_history:
                history['trial'].append(i)
                history['state'].append(current_state)
                history['action'].append(action)
                history['reward'].append(experience[2])
                if experience[3] is not None:
                    history['next_state'].append(experience[3])
                    history['done'].append(False)
                else:
                    history['next_state'].append(current_state)
                    history['done'].append(True)
                history['q_values'].append(q_values)

            dqn.storeTransition(experience)
            dqn.sampleRandomMinibatch()

            # s = s' at the end of the step, before starting the new step
            current_state = experience[3]

        if converge_criteria is not None:
            # Check if convergence counter should be increased or to be reset
            if emulator.average_reward > 200:
                convergence_counter += 1
            else:
                convergence_counter = 0

            episodes_convergence_counter.append(convergence_counter)

            if verbose > C_VERBOSE_NONE:
                print('Convergence Counter: ', convergence_counter, sep='')

            # DQN model assumed that it has been converged
            if convergence_counter >= converge_criteria:
                convergence_episode = i
                break

    if store_history:
        for k in history.keys():
            history[k] = np.array(history[k])
        history_save_dir = os.path.split(history_save_path)[0]
        if not os.path.exists(history_save_dir):
            os.makedirs(history_save_dir)
        pd.to_pickle(history, history_save_path)

    if converge_criteria is not None:
        convergence_time = time.time() - start_time

    if verbose > C_VERBOSE_NONE and converge_criteria is not None:
        print('\nDQN converged after ', convergence_episode, ' episodes in ', executionTimeToString(convergence_time),
              sep='')
    elif verbose > C_VERBOSE_NONE and converge_criteria is None:
        print('\nDQN trained for ', maximum_episodes, ' episodes in ', executionTimeToString(time.time() - start_time),
              sep='')

    # Create Graphs
    # 1. Steps per Episode
    plt.plot(emulator.execution_statistics.values[:, 0], emulator.execution_statistics.values[:, 1], color='coral',
             linestyle='-')
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps', fontsize=12)
    plt.title('Steps per Episode', fontsize=12)
    plt.savefig('Steps_Per_Episode' + graphs_suffix + '.png')
    plt.clf()

    # 2. Total Reward per Training Episode
    plt.plot(emulator.execution_statistics.values[:, 0], emulator.execution_statistics.values[:, 2], color='coral',
             linestyle='-',
             label='Total Reward')
    plt.plot(emulator.execution_statistics.values[:, 0], emulator.execution_statistics.values[:, 3],
             color='midnightblue', linestyle='--',
             label='Episodes Reward Average')
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Total Reward per Training Episode', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig('Total_Reward_Per_Training_Episode' + graphs_suffix + '.png')
    plt.clf()

    # Save the trained model
    dnn.saveModel(model_file_name)

    if converge_criteria is not None:
        return convergence_episode


def main():
    # Parse input arguments
    description_message = 'Lunar Lander with DQN'

    args_parser = argparse.ArgumentParser(description=description_message,
                                          formatter_class=argparse.RawTextHelpFormatter)

    args_parser.add_argument('-v', action='store', help='verbose level (0: None, 1: INFO, 2: DEBUG)',
                             choices=('0', '1', '2'))
    args_parser.add_argument('-e', action='store', required=True, help='execute (train, test)',
                             choices=('train', 'test'))
    args_parser.add_argument('-a', action='store', required=False, help='trained agent file')
    args_parser.add_argument('-n', action='store', required=False, default=1, help='number of trials during testing')
    args_parser.add_argument('--rendering', action='store_true', required=False, default=False,
                             help='rendering during testing')
    args_parser.add_argument('--store_history', action='store_true', required=False, default=False,
                             help='store history during testing')
    args_parser.add_argument('--compute_saliency', action='store_true', required=False, default=False,
                             help='compute saliency during testing')

    args = args_parser.parse_args()

    if args.e == 'test' and args.a is None:
        args_parser.error('When executing \'test\', the trained agent file (-a) is required.')

    # Verbose level (0: None, 1: INFO, 2: DEBUG)
    verbose = C_VERBOSE_NONE if args.v is None else int(args.v)

    num_trials = int(args.n)
    rendering = args.rendering
    store_history = args.store_history
    compute_saliency = args.compute_saliency

    # Trigger the requested execution type
    if args.e == 'train':
        if verbose:
            print('\nTrain a DQN using seed = 1 and default convergence criteria.')

        # TRAIN WITH SEED = 1, AND converge_criteria = 150

        seed = 1
        train(scenario='LunarLander-v2', average_reward_episodes=100, rendering=False, hidden_layers=1,
              hidden_layers_size=64, memory_size=None, minibatch_size=64, optimizer_learning_rate=0.001, gamma=0.99,
              epsilon_decay_factor=0.99995, maximum_episodes=10000, model_file_name='DQN_Trained.h5',
              converge_criteria=150, graphs_suffix='_Conv_150', seed=seed, verbose=verbose, store_history=store_history)

    else:
        if verbose:
            print('\nTest once the trained DQN agent.')

        trial(model_file_name=args.a, scenario='LunarLander-v2', number_of_trials=num_trials, rendering=rendering,
              graphs_suffix='', verbose=verbose, store_history=store_history, compute_saliency=compute_saliency)


if __name__ == '__main__':
    main()
