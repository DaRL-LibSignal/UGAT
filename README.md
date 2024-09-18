
# Uncertainty-Aware Grounded Action Transformation for Sim2Real_TSC


## 🚀 🚀 🚀
## We have created a docker image for your convenience to use this code base!

This docker code base contains three projects, first pull from docker hub: 

`docker pull danielda1/ugat:latest`

`docker run -it --name ugat_case danielda1/ugat:latest`

For this repo's paper:  CDC23: Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control

`cd /DaRL/UGAT_Docker/`

`python sim2real.py`

## At the same time of using this Docker Image, you have the the readily prepared LibSignal
This is a multi-simulator supported framework, provide easy-to-configure settings for sim-to-sim simulated sim-to-real training and testing.
For details, please visit: https://darl-libsignal.github.io/


For LibSignal - Then go to the terminal: 

`cd /DaRL/LibSignal`

`python run.py`


## We have also included another sim-to-real for RL - TSC tasks:  

> AAAI24: Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning (https://github.com/DaRL-LibSignal/PromptGAT)

Stay in the same docker environment, go to command line:

`cd /DaRL/PromptGAT`

`python sim2real.py`


## Instructions:

1. For prepartion: 
Please make sure you install the LibSignal required environment (especially: CityFlow, SUMO)
Check the doc for help: https://darl-libsignal.github.io/

2. For running and starting:
After successfully installed all the requirement, consider starting the training by simply running sim2real.py

> Please note, if you want to change the configs settings in `real`, please open the file at location:
`/DaRL/UGAT_Docker/data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.rou.xml`

And then you should be able to see the following settings, please in-annotate the one that you want to experiment with:

```
<!-- v1:lighter -->
<!-- <vType accel="1.0" decel="2.5" emergencyDecel='6.0' startupDelay='0.5' id="pkw" length="5.0" maxSpeed="11.111" minGap="2.5" width="2.0"/> -->
<!-- v2:heavier -->
<!-- <vType accel="1.0" decel="2.5" emergencyDecel='6.0' startupDelay='0.75' id="pkw" length="5.0" maxSpeed="11.111" minGap="2.5" width="2.0"/> -->
<!-- v3:rain -->
<vType accel="0.75" decel="3.5" emergencyDecel='6.0' startupDelay='0.25'  id="pkw" length="5.0" maxSpeed="11.111" minGap="2.5" width="2.0"/>
<!-- v4:snow -->
<!-- <vType accel="0.5" decel="1.5" emergencyDecel='2.0' startupDelay='0.5' id="pkw" length="5.0" maxSpeed="11.111" minGap="2.5" width="2.0"/> -->

```


3. For debugging, please make sure all the files are correctly imported
Please focus on the sim2real_trainer.py line 207-220. This is the outline of the GAT model.

4. You may arbitrarily change any uncertainty model 
Currently is using an uncertainty estimation within the model layers, you can apply any kind of uncertainty you wish.

5. Suggestion:
Make sure the sim2real.py runnable, then transplant the bare GAT model without uncertainty to your application scenario.
After successfully doing that, you may then decide to evaluate the model/action uncertainty and propose some solutions.

6. A tip in this project:
Probably you will not need to check the logs in this repo if for learning the GAT part soley, however, if you want, you can always find a log file after
one time of execution under the folder of: `Sim2Real_TSC/data/output_data/sim2real`. But by defauly, if you execute a second time, the last data will be overwrite.
To visualize the learning process: please consider using the script vis.ipynb by providing the path to a file ends with BRF.log: log_file='xxxBRF.log'


> Thanks Awesome Romir Sharma for helping creating the docker and make it easy to use!



## Cite:
If you find this paper helpful, please cite us:
```
@inproceedings{da2023uncertainty,
  title={Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control},
  author={Da, Longchao and Mei, Hao and Sharma, Romir and Wei, Hua},
  booktitle={2023 62nd IEEE Conference on Decision and Control (CDC)},
  pages={1124--1129},
  year={2023},
  organization={IEEE}
}
```

