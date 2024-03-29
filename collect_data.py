import sys
import json
import os
import pathlib

if __name__ == '__main__':

    json_args_file_path = sys.argv[1]
    with open(json_args_file_path) as infile:
        json_data = json.load(infile)
    
    experiment_id = 0
    for codebook_prune_value in [8, 12]:
        for PSNR in [20, 4, 12]:
            json_data["PSNR"] = PSNR
            for model, codebook_layer in zip(["resnet56", "resnet20"], [8, 2]):
                json_data["model"] = model
                json_data["reference_pretrained_weights_path"] = f"resource/cifar_pretrained/{model}.th"
                for codebook_size in [16]:
                    for beta in [1e-3, 1e-4, 1e-5]:
                        for codebook in json_data["codebooks"]:
                            codebook["prune_value"] = codebook_prune_value
                            codebook["codebook_size"] = codebook_size
                            codebook["layer"] = f"layer1[{codebook_layer}]"
                            codebook["beta"] = f"LinearCoefficient({beta / 10}, {beta})"
                        output_folder = f"output/experiment_{experiment_id}"
                        try:
                            pathlib.Path(output_folder).mkdir(parents=True)
                        except FileExistsError:
                            print(f'PSNR: {PSNR}, Model: {model}, Codebook Prune Value: {codebook_prune_value}, Codebook Size: {codebook_size}, beta: {beta} skipped')
                            experiment_id += 1
                            continue
                        json_data["output_folder"] = output_folder
                        with open(f"configs/experiment_{experiment_id}.json", 'w') as outfile:
                            json.dump(json_data, outfile, indent=4)
                        os.system(f"python PhyDistInf.py configs/experiment_{experiment_id}.json > {output_folder}/report.txt  2>&1")
                        experiment_id += 1
