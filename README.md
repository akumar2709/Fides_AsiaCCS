# Fides_AsiaCCS
This repository contains the code for our paper "A Generative Framework for Low-Cost Result Validation of Machine Learning-as-a-Service Inference"

```
@inproceedings{KumA2024,
  title={A Generative Framework for Low-Cost Result Validation of Machine Learning-as-a-Service Inference},
  author={Kumar, Abhinav and Aguilera, Miguel A Guirao and Tourani, Reza and Misra, Satyajayant},
  booktitle={Proceedings of the 2024 ACM Asia Conference on Computer and Communications Security},
  year={2024}
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
