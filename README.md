
# Uncertainty-Aware Grounded Action Transformation for Sim2Real_TSC


## Instructions:

1. For prepartion: 
Please make sure you install the LibSignal required environment (especially: CityFlow, SUMO)
Check the doc for help: https://darl-libsignal.github.io/

2. For running and starting:
After successfully installed all the requirement, consider starting the training by simply running sim2real.py

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

