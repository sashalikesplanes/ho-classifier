import yaml
from pathlib import Path
from functools import reduce

sweep_folder = 'crossValidFix'


with open(Path('sweepConfigs', sweep_folder, 'example.yaml')) as file:
    all_vars_si = yaml.load(file)
    
print(all_vars_si)

# signal_combos = [
#     ['e', 'u'],
#     ['e', 'x'],
#     ['u', 'x'],
#     ['e', 'u', 'x'],
#     ['e', 'u', 'x', 'dedt'],
#     ['e', 'u', 'x', 'dudt'],
#     ['e', 'u', 'x', 'dxdt'],
# ]

for subject_num in range(9):
    new_config = all_vars_si.copy()
    new_config["name"] = f"Subject {subject_num + 1}"
    new_config["parameters"]["valid_subject"]["value"] = subject_num
    
    name_str = f"subject_{subject_num + 1}.yaml"
    with open(Path('sweepConfigs', sweep_folder, name_str), 'w') as file:
        yaml.dump(new_config, file)