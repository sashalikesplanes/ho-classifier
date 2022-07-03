from tsai.all import *
import Path
from get_data_pred import get_var_prev_data

path_to_learner = Path('learners', 'best_si')
learner = Learner.load(path_to_learner)
path_to_file = Path('expdata_var_prev.mat')

X = get_var_prev_data(path_to_file)
# Load data for prediction

# Load model for prediction

# Do the predictions
