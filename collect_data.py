import sys
import json
import os
import pathlib
import re
import copy

if __name__ == '__main__':

    current_configs = []
    config_filenames = [config_filename for config_filename in os.listdir("configs/") if 'base' not in config_filename]
    experiment_id = max([int(re.search("\d+", config_filename).group()) for config_filename in config_filenames]) + 1
    for config_filename in config_filenames:
        with open(os.path.join("configs/", config_filename)) as config_file:
            current_configs.append(json.load(config_file))
        config_id = re.search("\d+", config_filename).group()

    json_args_file_path = sys.argv[1]
    with open(json_args_file_path) as infile:
        json_data = json.load(infile)

    for codebook_prune_value in [8, 12]:
        for dataset in ["cifar10", "cifar100"]:
            json_data["dataset"] = dataset
            for PSNR in [20, 4, 12, 8, 16]:
                json_data["PSNR"] = PSNR
                for model, codebook_layer in zip(["resnet56", "resnet20", "resnet110"], [8, 2, 17]):
                    json_data["model"] = model
                    for codebook_size in [16]:
                        for beta in [1e-3, 1e-4, 1e-5]:
                            for codebook in json_data["codebooks"]:
                                codebook["prune_value"] = codebook_prune_value
                                codebook["codebook_size"] = codebook_size
                                codebook["layer"] = f"layer1[{codebook_layer}]"
                                codebook["beta"] = f"LinearCoefficient({beta / 10}, {beta})"
                            output_dir = f"output/experiment_{experiment_id}"
                            json_data["output_dir"] = output_dir

                            # Checking for matches in already collected data
                            match_found = False
                            for current_config in current_configs:
                                json_data_for_comparison = copy.deepcopy(json_data)
                                current_config_for_comparison = copy.deepcopy(current_config)
                                del current_config_for_comparison['output_dir']
                                del json_data_for_comparison['output_dir']
                                if current_config_for_comparison == json_data_for_comparison:
                                    print(f'PSNR: {PSNR}, Dataset: {dataset}, Model: {model}, Codebook Prune Value: {codebook_prune_value}, Codebook Size: {codebook_size}, beta: {beta} found. Skipping...')
                                    match_found = True
                            if match_found:
                                continue
                            
                            pathlib.Path(output_dir).mkdir(parents=True)
                            with open(f"configs/experiment_{experiment_id}.json", 'w') as outfile:
                                json.dump(json_data, outfile, indent=4)
                            os.system(f"python PhyDistInf.py configs/experiment_{experiment_id}.json > {output_dir}/report.txt  2>&1")
                            experiment_id += 1
