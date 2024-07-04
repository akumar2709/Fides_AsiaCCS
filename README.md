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
abstract = {The growing popularity of Machine Learning (ML) has led to its deployment in various sensitive domains, which has resulted in significant research focused on ML security and privacy. However, in some applications, such as Augmented/Virtual Reality, integrity verification of the outsourced ML tasks is more critical-a facet that has not received much attention. Existing solutions, such as multi-party computation and proof-based systems, impose significant computation overhead, which makes them unfit for real-time applications. We propose Fides, a novel framework for real-time integrity validation of ML-as-a-Service (MLaaS) inference. Fides features a novel and efficient distillation technique-Greedy Distillation Transfer Learning-that dynamically distills and fine-tunes a space and compute-efficient verification model for verifying the corresponding service model while running inside a trusted execution environment. Fides features a client-side attack detection model that uses statistical analysis and divergence measurements to identify, with a high likelihood, if the service model is under attack. Fides also offers a re-classification functionality that predicts the original class whenever an attack is identified. We devised a generative adversarial network framework for training the attack detection and re-classification models. The evaluation shows that Fides achieves an accuracy of up to 98\% for attack detection and 94\% for re-classification.},
booktitle = {Proceedings of the 19th ACM Asia Conference on Computer and Communications Security},
pages = {1246–1260},
numpages = {15},
keywords = {verifiable computing, result verification, trusted execution environment, machine learning as a service, edge computing},
location = {Singapore, Singapore},
series = {ASIA CCS '24}
}


```
To test Fides -

1) Open the training folder and run the training script to train a service model
2) Open the distillation folder and then run the distillation script to train a distillation model
3) Run detector_corrector_training.py to train the detection and correction model
4) Run attack_testing.py to test the detection and correction model


## Disclaimer

**DO NOT USE THIS SOFTWARE TO SECURE ANY 
REAL-WORLD DATA OR COMPUTATION!**

This software is a proof-of-concept, meant for 
testing purposes only.
