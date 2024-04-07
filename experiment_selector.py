import json
import os

import numpy as np

configs_dir = "configs/"
experiment_filenames = [experiment_filename for experiment_filename in os.listdir(configs_dir) if ".json" in experiment_filename and "base" not in experiment_filename]


selected_experiments = []
performances = []

for experiment_filename in experiment_filenames:
    with open(os.path.join(configs_dir, experiment_filename)) as experiment_file:
        config = json.load(experiment_file)
        if config['PSNR'] == 12 and config['dataset'] == "cifar10" and config['model'] == "resnet110": # and config['codebooks'][0]['prune_value'] == 12:
            selected_experiments.append(experiment_filename)
            print(experiment_filename)
            print(json.dumps(config, indent=4))
            with open(os.path.join(config['output_dir'], "measures.json")) as measures_file:
                measures = json.load(measures_file)
                performances.append(measures['codebook_model']['acc']['best'])

print(f"selected experiments: {selected_experiments}")
print(performances)
print(f"best experiment: {selected_experiments[np.argmax(performances)]}")