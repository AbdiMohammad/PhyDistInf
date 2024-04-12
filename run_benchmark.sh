#! /bin/bash

python benchmark.py --config configs/base_experiment_resnet20.json --method EC
python benchmark.py --config configs/base_experiment_resnet20.json --method SC 
python benchmark.py --config configs/base_experiment_resnet20.json --method BF-1
python benchmark.py --config configs/base_experiment_resnet20.json --method BF-2

python benchmark.py --config configs/experiment_33.json --method PhyDistInf

python benchmark.py --config configs/base_experiment_resnet56.json --method EC
python benchmark.py --config configs/base_experiment_resnet56.json --method SC
python benchmark.py --config configs/base_experiment_resnet56.json --method BF-1
python benchmark.py --config configs/base_experiment_resnet56.json --method BF-2

python benchmark.py --config configs/experiment_30.json --method PhyDistInf

python benchmark.py --config configs/base_experiment_resnet110.json --method EC
python benchmark.py --config configs/base_experiment_resnet110.json --method SC
python benchmark.py --config configs/base_experiment_resnet110.json --method BF-1
python benchmark.py --config configs/base_experiment_resnet110.json --method BF-2

python benchmark.py --config configs/experiment_42.json --method PhyDistInf
