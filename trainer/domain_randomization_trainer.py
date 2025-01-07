import datetime
import os
import sys
import time
import pickle as pkl
import numpy as np
import random
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
from trainer.sim2real_trainer import SIM2REALTrainer

from common import interface
from common.stat_utils import NN_predictor, UNCERTAINTY_predictor
from agent.utils import idx2onehot
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torch.nn.functional as F
from  torch import optim, nn
from torch.nn.utils import clip_grad_norm_

# The function here will be mostly the same as sim2real trainer, so we inherit 
@Registry.register_trainer("domain_randomization")
class DomainRandomizationTrainer(SIM2REALTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''

    def __init__(
            self,
            logger,
            config,
            gpu=0,
            cpu=False,
            name="domain_randomization"
    ):
        super().__init__(
            logger,
            config,
            gpu=0,
            cpu=False,
            name=name
        )
        self._count = 1
    
    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world_sim = Registry.mapping['world_mapping']['cityflow'](
            self.path, 
            Registry.mapping['command_mapping']['setting'].param['thread_num'], 
            domain_randomization=True)

        self.world_real = Registry.mapping['world_mapping']['sumo'](
            self.path.replace('cityflow', 'sumo_gaus'),
            interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def recreate_world(self):
        '''
        This function will recreate the randomize world, for cityflow only
        '''
        # traffic setting is in the world mapping
        self.world_sim = Registry.mapping['world_mapping']['cityflow'](
            self.path, 
            Registry.mapping['command_mapping']['setting'].param['thread_num'],
            domain_randomization=True, count=self._count)
        self._count += 1

    def run(self):
        global e
        self.create_world()
        self.create_agents()
        self.create_metrics()
        self.create_env()
        self.root_path = Registry.mapping['logger_mapping']['path'].path
        self.model_path = os.path.join(self.root_path, Registry.mapping['logger_mapping']['setting'].param['model_dir'])
        self.data_path = os.path.join(self.root_path, Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        self.debug_path = os.path.join(self.root_path, 'debug')

        # Set all agent networks to be the same if centralized
        # if self.parameter_sharing:
        #     path = self.model_path + "-1_0.pt"
        #     self.load_pretrained(path)
        
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)

        #'pretrained'/'restart'
        if self.experiment_mode == 'pretrained': 
            path = self.model_path

        elif self.experiment_mode =='restart':
            # path = self.model_path  # temporarily disable for checking domain_randomization
            path = self.pretrain(episodes=self.pretrain_n) # previously has +5

        # collect trajectories for transfer metric
        if self.transfer_metric == True:
            start_time = time.time()  # Start timer
            self.precollect_trajectories(100, "precollected.pkl")
            end_time = time.time()  # End timer

            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Time taken for precollecting trajectories: {elapsed_time:.2f} seconds")

        # pretrain model and load pretrained ones
        best_e = -1
        V = []
        self.load_shared(path) # load both on the sim and real
        for e in range(self.episodes):
            
            self.recreate_world()
            self.create_env()

            # If decentralized handle real rollout differently?
            if self.decentralized_GAT == "both":
                self.load_shared(path)
                R = self.real_rollout(e - 1, True)
                self.sim_rollout(e - 1, True)
                V.append(R)
            else:
                R = self.real_rollout(e - 1)
                self.sim_rollout(e - 1)
                V.append(R)

            # Handle single agent and multi-sgent cases differently. Now also assume num agents > 1
            assert len(self.agents_real) > 1, "This code support multi-agent only"

            # Train one forward and one inverse model for all agents for centralized GAT
            if self.decentralized_GAT == "centralized":
                
                # sim2real training
                sim_rollout_time = time.time()
                path = self.sim_tain(episode=20, e=e, writer=self.writer)
                end_time = time.time()
                print(f"Sim rollout time: {end_time - sim_rollout_time}, avg time per rollout: {end_time / 20}")

            # Train forward and inverse models separately for each agent for decentralized GAT
            elif self.decentralized_GAT == "both":
                    
                # sim2real training
                sim_rollout_time = time.time()
                path = self.noGAT_sim_train(episode=20, e=e, writer=self.writer) # TODO THIS IS SETUP FOR NO GAT CURRENTLY, CHANGE LATER
                end_time = time.time()
                print(f"Sim rollout time: {end_time - sim_rollout_time}, avg time per rollout: {(end_time - sim_rollout_time) / 20}")
            

        self.load_shared(path)
        R = self.real_eval(e)
        V.append(R)

    print('-' * 10 + 'finished' + '-' * 10)

    # file to save target policy
    # file to save real rollout (states, action -> states'), forward model
    # file to save sim rollout (states, states' -> action), inverse model
    # value v() in real
    # optimize process log
