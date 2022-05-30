import numpy as np
from scipy import io
import random
from sklearn.preprocessing import StandardScaler

def get_fofu_data(variables=['e', 'u', 'x', 'dedt', 'dudt', 'dxdt'],
                  data_file='expdata_SI.mat',
                  time_window=150,
                  total_time_steps=12000,
                  sample_freq=50,
                  time_steps_between_samples=75,
                  data_freq=100,
                  n_subjects=5,
                  n_runs=1):
    
    # number of variables selected for the run (e, u, x)
    n_variables = len(variables)
    n_samples =  # TODO something like n_runs * n_subjects * samples per run

    # Data is loaded as a dictionary of all the conditions
    # In each key there is a another dictionary representing all the recorded variables
    # Each variable is a Rank 3 matrix with axis 0 representing time steps
    # axis 1 representing the run number per subject
    # and axis 2 representing the subject number
    data_dict = io.loadmat("./data/" + data_file, simplify_cells=True)['ed']

    # Convert the data from the dictionaries into a 5 dimensional array
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

    # Setup blank X 
    X = np.empty((n_samples_train + n_samples_valid, n_variables, time_window))
    # Take individual samples from the array and add to the X, Y arrays
    for subj_index in subj_indices:
        for run_index in run_indices:
            for t in range(0, total_time_steps - time_window, time_steps_between_samples):
                X[sample_index] = data_array[cond_index, :,
                                             t:t + time_window, run_index, subj_index]
    # Remove samples to reduce the final sample rate
    return X[:, :, ::int(data_freq / sample_freq)], Y, splits


