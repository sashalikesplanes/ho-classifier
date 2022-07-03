import numpy as np
from scipy import io
import random
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# TO ANY FUTURE READER:
#   This file was made extremely quick and dirty right before the deadline
#   I severly apoligize for any pain you may have experienced while reading this code
#   Please do not judge me too harshly
#   - sasha
def get_var_prev_data(file = Path('data', 'expdata_var_prev.mat'),
                     desired_conditions = ['PRL', 'PRM', 'PRH'],
                     run_indices = [0],
                     subj_indices = [0, 1, 2, 3, 4],
                     time_steps_between_samples=75,
                     start_time=0,
                     end_time=5000,
                     ):
    
    data_dict = io.loadmat(file, simplify_cells=True)['ed']

    n_subjects = len(subj_indices)
    max_n_subjects = 5
    n_runs = len(run_indices)
    n_conditions = len(desired_conditions)
    variables=['e', 'u', 'x', 'dedt', 'dudt', 'dxdt']
    n_variables = len(variables)
    time_window=150
    # For sampling the data
    total_time_steps=end_time - start_time
    sample_freq=50
    
    total_time_steps_data = 12000 # For loading the data
    
    data_freq=100
    

    all_vars_per_cond = []
    for cond in desired_conditions:
        all_vars = np.empty(
            (len(variables), total_time_steps_data, n_runs, max_n_subjects))
        for i, variable in enumerate(variables):
            try:
                all_vars[i] = np.array(data_dict[cond][variable]).reshape((total_time_steps_data, n_runs, max_n_subjects))
            # Will fail if the variable is one of the derivatives as those are not in raw data
            # so we calculate derivative manually
            except KeyError:
                if variable == 'dedt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['e'].reshape((total_time_steps_data, n_runs, max_n_subjects)), 1 / data_freq, axis=0)
                if variable == 'dudt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['u'].reshape((total_time_steps_data, n_runs, max_n_subjects)), 1 / data_freq, axis=0)
                if variable == 'dxdt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['x'].reshape((total_time_steps_data, n_runs, max_n_subjects)), 1 / data_freq, axis=0)
            var_for_scaling = all_vars[i].reshape(
                (total_time_steps_data, n_runs * max_n_subjects))
            var_scaled = StandardScaler(
                copy=False).fit_transform(var_for_scaling)
            all_vars[i] = var_scaled.reshape(
                (total_time_steps_data, n_runs, max_n_subjects))
        all_vars_per_cond.append(all_vars)
    # condition x variable x time x run x subject
    data_array = np.stack(all_vars_per_cond)
    del all_vars_per_cond
    
    n_samples = round((total_time_steps - time_window) / time_steps_between_samples *
                            n_conditions * n_subjects * n_runs)

    X = np.empty((n_samples, n_variables, time_window))
    
    sample_index = 0
    for cond_index in range(n_conditions):
        for subj_index in subj_indices:
            for run_index in run_indices:
                for t in range(start_time, end_time - time_window, time_steps_between_samples):
                    X[sample_index] = data_array[cond_index, :,
                                                 t:t + time_window, run_index, subj_index]
                    sample_index += 1

    del data_array
    # Remove samples to reduce the final sample rate
    return X[:, :, ::int(data_freq / sample_freq)]

def get_fofu_data(valid_pct=0.2,
                  random_labels=False,
                  variables=['e', 'u', 'x', 'dedt', 'dudt', 'dxdt'],
                  data_file='expdata_SI.mat',
                  time_window=150,
                  total_time_steps=12000,
                  sample_freq=50,
                  time_steps_between_samples=75,
                  data_freq=100,
                  subj_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                  run_indices=[0, 1, 2, 3, 4],
                  desired_conditions=['CL', 'CM', 'CH',
                                      'PL', 'PM', 'PH', 'PRL', 'PRM', 'PRH'],
                  condition_labels=[0, 0, 0, 1, 1, 1, 2, 2, 2],
                  display_types=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  valid_subject=None):
    
    print(f"Loading data from: {data_file}")
    print(f"Selecting variables: {variables}")
    print(f"Selecting conditons: {desired_conditions}")
    print(f"With labels: {condition_labels}")
    n_subjects = len(subj_indices)
    n_conditions = len(desired_conditions)
    n_runs = len(run_indices)
    # number of variables selected for the run (e, u, x)
    n_variables = len(variables)
    n_samples_train = round((total_time_steps - time_window) / time_steps_between_samples *
                            n_conditions * n_subjects * n_runs * (1 - valid_pct))
    n_samples_valid = round((total_time_steps - time_window) /
                            time_steps_between_samples * n_conditions * n_subjects * n_runs * valid_pct)

    # Data is loaded as a dictionary of all the conditions
    # In each key there is a another dictionary representing all the recorded variables
    # Each variable is a Rank 3 matrix with axis 0 representing time steps
    # axis 1 representing the run number per subject
    # and axis 2 representing the subject number
    data_dict = io.loadmat("./data/" + data_file, simplify_cells=True)['ed']

    # Convert the data from the dictionaries into a 5 dimensional array
    all_vars_per_cond = []
    for cond in desired_conditions:
        all_vars = np.empty(
            (len(variables), total_time_steps, (n_runs), n_subjects))
        for i, variable in enumerate(variables):
            try:
                all_vars[i] = data_dict[cond][variable]
            # Will fail if the variable is one of the derivatives as those are not in raw data
            # so we calculate derivative manually
            except KeyError:
                if variable == 'dedt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['e'], 1 / data_freq, axis=0)
                if variable == 'dudt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['u'], 1 / data_freq, axis=0)
                if variable == 'dxdt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['x'], 1 / data_freq, axis=0)
            var_for_scaling = all_vars[i].reshape(
                (total_time_steps, n_runs * n_subjects))
            var_scaled = StandardScaler(
                copy=False).fit_transform(var_for_scaling)
            all_vars[i] = var_scaled.reshape(
                (total_time_steps, n_runs, n_subjects))
        all_vars_per_cond.append(all_vars)
    # condition x variable x time x run x subject
    data_array = np.stack(all_vars_per_cond)
    del all_vars_per_cond

    # Setup blank X and Y arrays, they will store both train and test
    X = np.empty((n_samples_train + n_samples_valid, n_variables, time_window))
    Y = np.empty((n_samples_train + n_samples_valid,))
    # The splits variable records the indices of train and valid
    # first array contains all the indices in X, Y arrays corresponding to train data, second the valid data
    splits = ([], [])
    # Take individual samples from the array and add to the X, Y arrays
    sample_index = 0
    for cond_index in range(n_conditions):
        for subj_index in subj_indices:
            for run_index in run_indices:
                for t in range(0, total_time_steps - time_window, time_steps_between_samples):
                    X[sample_index] = data_array[cond_index, :,
                                                 t:t + time_window, run_index, subj_index]
                    # Add ability to randomize labels
                    if random_labels:
                        Y[sample_index] = np.random.randint(0, 3)
                    else:
                        Y[sample_index] = condition_labels[cond_index]
                    # Record if this run will be validation or test
                    if valid_subject == None:
                        if random.random() < valid_pct:
                            splits[1].append(sample_index)
                        else:
                            splits[0].append(sample_index)
                    else:
                        if subj_index == valid_subject:
                            splits[1].append(sample_index)
                        else:
                            splits[0].append(sample_index)

                    sample_index += 1

    del data_array
    # Remove samples to reduce the final sample rate
    return X[:, :, ::int(data_freq / sample_freq)], Y, splits

if __name__ == "__main__":
    file = Path('expdata_var_prev.mat')
    print(get_var_prev_data(file))
