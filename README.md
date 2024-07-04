# Fides_AsiaCCS
This repository contains the code for our paper "A Generative Framework for Low-Cost Result Validation of Machine Learning-as-a-Service Inference"

```
@inproceedings{10.1145/3634737.3657015,
author = {Kumar, Abhinav and Aguilera, Miguel A. Guirao and Tourani, Reza and Misra, Satyajayant},
title = {A Generative Framework for Low-Cost Result Validation of Machine Learning-as-a-Service Inference},
year = {2024},
isbn = {9798400704826},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3634737.3657015},
doi = {10.1145/3634737.3657015},
booktitle = {Proceedings of the 19th ACM Asia Conference on Computer and Communications Security},
pages = {1246–1260},
numpages = {15},
keywords = {verifiable computing, result verification, trusted execution environment, machine learning as a service, edge computing},
location = {Singapore, Singapore},
series = {ASIA CCS '24}
}


```
## To test Fides -

1) Open the training folder and run the training script to train a service model
2) Open the distillation folder and then run the distillation script to train a distillation model
3) Run
  ```bash
	$   python detector_corrector_training.py
```

4) Run
  ```bash
	$   python attack_testing.py
```


## Disclaimer

**DO NOT USE THIS SOFTWARE TO SECURE ANY 
REAL-WORLD DATA OR COMPUTATION!**

This software is a proof-of-concept, meant for 
testing purposes only.
