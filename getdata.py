import numpy as np
from scipy import io
import random
from sklearn.preprocessing import StandardScaler


def get_fofu_data_experimnetal(valid_pct=0.2,
                               variables=['e', 'u', 'x',
                                          'dedt', 'dudt', 'dxdt'],
                               time_window=150,
                               n_time_steps=12000,
                               time_steps_between_samples=75,
                               seconds_per_step=0.01,
                               subj_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                               run_indices=[0, 1, 2, 3, 4],
                               desired_conditions=['CL', 'CM', 'CH',
                                                   'PL', 'PM', 'PH', 'PRL', 'PRM', 'PRH'],
                               display_types=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                               n_displays=3, n_conditions_per_display=3):
    """
    This function exists to load the data from the last VdEl experiment (which contained 9 subjects
    each in 9 conditions performing 5 runs per condition) stored as a matlab struct
    In the form convenient for the usage of classifying which display type was used
    by a machine learning algorithm
    """
    n_subjects = len(subj_indices)
    n_conditions = len(desired_conditions)
    n_runs = len(run_indices)
    # number of variables selected for the run (e, u, x)
    n_variables = len(variables)
    n_samples = round((n_time_steps - time_window) / time_steps_between_samples *
                      n_conditions * n_subjects * n_runs)
    n_runs_total_per_display = n_subjects * n_runs * n_conditions_per_display

    # Iterate over the 3 display types
    # Per type, make an array of shape (n_variables, n_time_steps, n_runs_total_per_display)
    #   also generating the derivative variables
    # Standardize the array per variable
    StandardScaler(copy=False)
    # Transpose the array to (n_runs_total_per_display, n_variables,  n_time_steps)
    # iterate over runs
    #   randomly set run to validation with probability valid_pct
    #   append all samples from the run to X (n_samples, n_variables, time_window)
    #   append the display type to Y (n_samples,)
    #   append the current sample index to either validation or train array for splits
    data_dict = io.loadmat('expdata.mat', simplify_cells=True)['ed']
    for display_label in range(n_displays):
        all_runs_per_display = np.empty(
            n_variables, n_time_steps, n_runs_total_per_display)
        # for condition


def get_fofu_data(valid_pct=0.2,
                  variables=['e', 'u', 'x', 'dedt', 'dudt', 'dxdt'],
                  time_window=150,
                  total_time_steps=12000,
                  time_steps_between_samples=75,
                  seconds_per_step=0.01,
                  subj_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                  run_indices=[0, 1, 2, 3, 4],
                  desired_conditions=['CL', 'CM', 'CH',
                                      'PL', 'PM', 'PH', 'PRL', 'PRM', 'PRH'],
                  condition_labels=[0, 0, 0, 1, 1, 1, 2, 2, 2],
                  display_types=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]):

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
    data_dict = io.loadmat('expdata.mat', simplify_cells=True)['ed']

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
                        data_dict[cond]['e'], seconds_per_step, axis=0)
                if variable == 'dudt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['u'], seconds_per_step, axis=0)
                if variable == 'dxdt':
                    all_vars[i] = np.gradient(
                        data_dict[cond]['x'], seconds_per_step, axis=0)
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
                    Y[sample_index] = condition_labels[cond_index]
                    # Record if this run will be validation or test
                    if random.random() < valid_pct:
                        splits[1].append(sample_index)
                    else:
                        splits[0].append(sample_index)

                    sample_index += 1

    del data_array
    return X, Y, splits
