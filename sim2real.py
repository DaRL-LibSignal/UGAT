import os.path
import task
import trainer
import agent
import dataset
from common import interface
from common.registry import Registry
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse

import sys

import torch
import faulthandler


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=8, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="0", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='report', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo', 'traci'],
                    help="interface type")  # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx', 'real'],
                    help="method of calculating delay")  # apx(approximate) or real

parser.add_argument('-t', '--task', type=str, default="sim2real", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
parser.add_argument('-n', '--network', type=str, default="cityflow1x3", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')
parser.add_argument('--calculate_transfer_metrics', type=bool, default=False, help='enable transfer metric calculation')
parser.add_argument('--decentralized_GAT', type=str, default="both", choices=['forward', 'inverse', 'both', 'centralized'], help='enable decentralized GAT by training forward, inverse, or both in a decentralized manner')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

print(f"Parser args: \n{args}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available. Running on CPU.")

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        """

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)  # register logger path

        if self.config['model'].get('graphic', False):

            self.config['command']['world'] = 'sumo'
            interface.World_param_Interface(self.config)

            self.config['command']['world'] = 'cityflow'
            interface.World_param_Interface(self.config)

        else:
            raise ValueError

            # interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config)
        # make output dir if not exist
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)
        self.trainer = Registry.mapping['trainer_mapping'] \
            [Registry.mapping['command_mapping']['setting'].param['task']](logger, self.config)
        self.task = Registry.mapping['task_mapping'] \
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)

    # Enable fault handler
    faulthandler.enable()

    # Start timer
    start_time = time.time()

    # try:
    #     test.run()
    # except Exception as e:
    #     # Print the error message if an exception occurs
    #     print(f"An error occurred: {e}")
    # finally:
    #     # Calculate and print the elapsed time, regardless of success or failure
    #     elapsed_time = time.time() - start_time
    #     print(f"elapsed_time: {elapsed_time}")

    test.run()
    elapsed_time = time.time() - start_time
    print(f"elapsed_time: {elapsed_time}")
