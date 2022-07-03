import yaml
sweep_config = {
    "name": "Sweep on LR, Wd, and Dropout using Standardized data with accuracy as metric",
    "project": "HOClass",
    "description": "A sweep using Bayes search, adjusted range on Wd, decreased lower bound on Conv_dropout factor, added data standardization using Verskeer method, changed bottleneck to always off, switched residual to always off, switched LR to use a range",
    "method": "bayes",
    "metric": {
        "name": "accuracy",
        "goal": "maximize",
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 7500,
        "eta": 2,
    },

    "parameters": {
        "epochs": {
            "value": 50
        },
        "lr": {
            "min": 0.0025,
            "max": 0.01,
        },
        "batch_size": {
            "value": 64
        },
        "nf": {
            "value": 24,
        },
        "ks": {
            "value": 64
        },
        "bottleneck": {
            "value": False
        },
        "bottleneck_size": {
            "value": 0
        },
        "depth": {
            "value":  6
        },
        "residual": {
            "value": False
        },
        "valid_pct": {
            "value": 0.2
        },
        "wd": {
            "distribution": "log_uniform",
            "min": -2,
            "max": -1,
        },
        "conv_dropout": {
            "min": 0.05,
            "max": 0.3,
        },
        "variables": {
            "value": ['e', 'u', 'x', 'dedt', 'dudt', 'dxdt']
        },
        "bn": {
            "values": [True, False]
        }
    }
}

print(yaml.dump(sweep_config))
