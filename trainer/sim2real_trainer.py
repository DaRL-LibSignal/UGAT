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

@Registry.register_trainer("sim2real")
class SIM2REALTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''

    def __init__(
            self,
            logger,
            config,
            gpu=0,
            cpu=False,
            name="sim2real"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.config = config

        self.HISTORY_T = Registry.mapping['trainer_mapping']['setting'].param['history']
        self.pretrain_n = Registry.mapping['trainer_mapping']['setting'].param['pretrain_n']
        self.INVERSE = Registry.mapping['trainer_mapping']['setting'].param['inverse_type']
        self.experiment_mode = Registry.mapping['trainer_mapping']['setting'].param['experiment_mode']


        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']

        self.transfer_metric = Registry.mapping['trainer_mapping']['setting'].param['transfer_metric']
        self.decentralized_GAT = Registry.mapping['trainer_mapping']['setting'].param['decentralized_GAT']
        self.parameter_sharing = Registry.mapping['trainer_mapping']['setting'].param['parameter_sharing']

        self.learning_rate = 0.0001 # set for before sim2real manually

        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later

        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][
            Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        time_date = str(datetime.now())[:-7].replace(":", "_").replace(" ", "_").replace("-", "_")
        self.writer_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'writer/' + time_date)
        if not os.path.exists(self.writer_path):
            os.makedirs(self.writer_path)
        
        #self.writer = SummaryWriter(log_dir=self.writer_path)
        self.writer = None

        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
                                         '_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world_sim = Registry.mapping['world_mapping']['cityflow'](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'])

        self.world_real = Registry.mapping['world_mapping']['sumo'](
            self.path.replace('cityflow', 'sumo_gaus'),
            interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric_sim = Metrics(lane_metrics, world_metrics, self.world_sim, self.agents_sim)
        self.metric_real = Metrics(lane_metrics, world_metrics, self.world_real, self.agents_real)

    def verify_agent_networks(self, agents):
        """
        Verify whether all agents have the same network.

        :param agents: List of agents whose networks are to be compared.
        :return: Boolean indicating if all agents have the same network and details of differences if any.
        """
        if not agents:
            print("No agents to compare.")
            return False

        # Extract the network (architecture and weights) of the first agent as a reference
        reference_network = agents[0].model  # Assuming `network` holds the model object

        for idx, agent in enumerate(agents[1:], start=1):
            current_network = agent.model  # Assuming `network` holds the model object

            # Check if the architectures are the same
            if str(reference_network) != str(current_network):
                print(f"Agent {idx} has a different architecture.")
                return False

            # Check if the weights are the same
            for ref_param, curr_param in zip(reference_network.parameters(), current_network.parameters()):
                if not torch.equal(ref_param.data, curr_param.data):
                    print(f"Agent {idx} has different weights.")
                    return False

        print("All agents have the same network")
        return True
    
    def check_agent_networks(self):
        print("\nVerifying simulation agents...")
        same_sim = self.verify_agent_networks(self.agents_sim)

        test = []

        for ag in self.agents_sim:
            test.append(ag)

        for ag in self.agents_real:
            test.append(ag)
            
        print("\nVerifying real agents...")
        same_real = self.verify_agent_networks(self.agents_real)

        print("\nTest case all nets shared...")
        testing = self.verify_agent_networks(test)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents_sim = []
        self.agents_real = []

        # Copy same network for all agents if parameter sharing
        if self.parameter_sharing:
            print(f"Creating agents ")

            agent_sim = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
            self.world_sim, 0)

            self.agents_sim.append(agent_sim)

            num_agent = int(len(self.world_sim.intersections) / agent_sim.sub_agents)
            print(f"Total number of agents: {num_agent}, Total number of sub agents: {agent_sim.sub_agents}")

            # Copy agent n times, one for each desired agent
            for i in range(1, num_agent):
                self.agents_sim.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                self.world_sim, i, self.agents_sim[0]))
            
            for i in range(0, num_agent):
                self.agents_real.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                self.world_real, i, self.agents_sim[0]))

        else:

            agent_sim = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                self.world_sim, 0)
            
            num_agent = int(len(self.world_sim.intersections) / agent_sim.sub_agents)
            
            print(f"Total number of agents: {num_agent}, Total number of sub agents: {agent_sim.sub_agents}")
            self.agents_sim.append(agent_sim)  # initialized N agents for traffic light control

            for i in range(1, num_agent):
                self.agents_sim.append(
                    Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                        self.world_sim, i))
                
            agent_real = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                self.world_real, 0)

            num_agent = int(len(self.world_real.intersections) / agent_real.sub_agents)
            self.agents_real.append(agent_real)  # initialized N agents for traffic light control
            for i in range(1, num_agent):
                self.agents_real.append(
                    Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                        self.world_real, i))
            
        # Added functionality to check whether agents share parameters
        #self.check_agent_networks()


    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env_sim = TSCEnv(self.world_sim, self.agents_sim, self.metric_sim)
        self.env_real = TSCEnv(self.world_real, self.agents_real, self.metric_real)
    

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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Set all agent networks to be the same if centralized
        # if self.parameter_sharing:
        #     path = self.model_path + "-1_0.pt"
        #     self.load_pretrained(path)
        

        self.forward_models = []
        self.inverse_models = []

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)

        # TODO: support multiple intersections
        if self.INVERSE == 'NN':
            # TODO: support multiple intersections
            self.forward_model = NN_predictor(self.logger,
                                              (self.agents_real[0].ob_generator.ob_length + self.agents_real[0].action_space.n) * self.HISTORY_T,
                                              self.agents_real[0].ob_generator.ob_length, device, self.model_path,
                                              self.data_path + 'real.pkl', history=self.HISTORY_T)
            self.inverse_model = NN_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * 2,
                                              self.agents_real[0].action_space.n, device, self.model_path,
                                              self.data_path + 'sim.pkl', backward=True)
            # pretrain model and load pretrained ones

        # This supports multiple agents now
        elif self.INVERSE == 'UNCERTAINTY':

            # Considers number of agents in both input and output dimension calculations to allow for multi-agent setting
            num_agents = len(self.agents_real)

            if num_agents > 1 and self.decentralized_GAT == "centralized":

                print(f"Centralized GAT")

                self.forward_model = NN_predictor(self.logger,
                                                (self.agents_real[0].ob_generator.ob_length * num_agents + self.agents_real[0].action_space.n * num_agents) * self.HISTORY_T,
                                                self.agents_real[0].ob_generator.ob_length * num_agents, device, self.model_path,
                                                self.data_path + 'real.pkl', history=self.HISTORY_T)
                self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * num_agents * 2,
                                                self.agents_real[0].action_space.n * num_agents, device, self.model_path,
                                                self.data_path + 'sim.pkl', backward=True)
                
            # Decentralized GAT (forward and inverse models for each agent)
            elif num_agents > 1 and self.decentralized_GAT == "both":

                print(f"Decentralized GAT")

                # Create a forward and inverse model for every agent
                for i in range(num_agents):

                    # Include agent index in the file name
                    real_file_path = f"{self.data_path}real_{i+1}.pkl"
                    sim_file_path = f"{self.data_path}sim_{i+1}.pkl"
                

                    self.forward_model = NN_predictor(self.logger,
                                                    (self.agents_real[0].ob_generator.ob_length + self.agents_real[0].action_space.n) * self.HISTORY_T,
                                                    self.agents_real[0].ob_generator.ob_length, device, self.model_path,
                                                    self.data_path + 'real.pkl', history=self.HISTORY_T)
                    self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * 2,
                                                    self.agents_real[0].action_space.n, device, self.model_path,
                                                    self.data_path + 'sim.pkl', backward=True)
                    
                    self.forward_models.append(self.forward_model)
                    self.inverse_models.append(self.inverse_model)

            elif num_agents > 1 and self.decentralized_GAT == "forward":

                print(f"Decentralized forward only")
            
            elif num_agents > 1 and self.decentralized_GAT == "inverse":

                print(f"Decentralized inverse only")

            # Single agent case 
            else:

                self.forward_model = NN_predictor(self.logger,
                                                (self.agents_real[0].ob_generator.ob_length * num_agents + self.agents_real[0].action_space.n * num_agents) * self.HISTORY_T,
                                                self.agents_real[0].ob_generator.ob_length * num_agents, device, self.model_path,
                                                self.data_path + 'real.pkl', history=self.HISTORY_T)
                self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * num_agents * 2,
                                                self.agents_real[0].action_space.n * num_agents, device, self.model_path,
                                                self.data_path + 'sim.pkl', backward=True)


        #'pretrained'/'restart'
        if self.experiment_mode == 'pretrained': 
            path = self.model_path

        elif self.experiment_mode =='restart':
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

            # Handle single agent and multi-sgent cases differently
            if num_agents > 1:

                # Train one forward and one inverse model for all agents for centralized GAT
                if self.decentralized_GAT == "centralized":

                    # Calculate transfer metric if flag is present
                    if self.config['command'].get('calculate_transfer_metrics', False):
                        # train forward model
                        self.forward_train(writer=self.writer, transfer_calc=True)

                    else:
                        # train forward model
                        self.forward_train(writer=self.writer)

                    # train inverse model
                    _ = self.inverse_train(writer=self.writer)
                    
                    # sim2real training
                    sim_rollout_time = time.time()
                    path = self.sim_tain(episode=20, e=e, writer=self.writer)
                    end_time = time.time()
                    print(f"Sim rollout time: {end_time - sim_rollout_time}, avg time per rollout: {end_time / 20}")

                # Train forward and inverse models separately for each agent for decentralized GAT
                elif self.decentralized_GAT == "both":
                        
                    # train forward model
                    #self.forward_train(writer=self.writer)

                    # train inverse model
                    #_ = self.inverse_train(writer=self.writer)
                    
                    # sim2real training
                    sim_rollout_time = time.time()
                    path = self.noGAT_sim_train(episode=20, e=e, writer=self.writer) # TODO THIS IS SETUP FOR NO GAT CURRENTLY, CHANGE LATER
                    end_time = time.time()
                    print(f"Sim rollout time: {end_time - sim_rollout_time}, avg time per rollout: {(end_time - sim_rollout_time) / 20}")
            
            else:
                # Calculate transfer metric if flag is present
                if self.config['command'].get('calculate_transfer_metrics', False):
                    # train forward model
                    self.forward_train(writer=self.writer, transfer_calc=True)

                else:
                    # train forward model
                    self.forward_train(writer=self.writer)

                # train inverse model
                _ = self.inverse_train(writer=self.writer)
                
                # sim2real training
                path = self.sim_tain(episode=20, e=e, writer=self.writer)

        self.load_shared(path)
        R = self.real_eval(e)
        V.append(R)

    print('-' * 10 + 'finished' + '-' * 10)

    # file to save target policy
    # file to save real rollout (states, action -> states'), forward model
    # file to save sim rollout (states, states' -> action), inverse model
    # value v() in real
    # optimize process log

    def pretrain(self, episodes):
        """
        pretrain a model for in the simulator(cityflows)
        """
        total_decision_num = 0
        pretrained_episode = episodes
        flush = 0
        states_record = []
        action_record = []

        self.centralized_pretrain_replay_buffer = deque(maxlen=self.buffer_size*len(self.agents_sim))
        
        for e in range(pretrained_episode):
            # TODO: check this reset agent
            self.metric_sim.clear()
            last_obs = self.env_sim.reset()  # agent * [sub_agent, feature]
            epo_action_record = []
            epo_states_record = []

            epo_states_record.append(last_obs[0])
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])  # [agent, intersections]

                    # if total_decision_num > self.learning_start:
                    #     actions = []
                    #     for idx, ag in enumerate(self.agents_sim):
                    #         actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                    #     actions = np.stack(actions)  # [agent, intersections]
                    # else:
                    #     actions = np.stack([ag.sample() for ag in self.agents_sim])

                    actions = np.stack([ag.sample() for ag in self.agents_sim])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents_sim):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric_sim.update(rewards)

                    # debug
                    epo_states_record.append(obs[0])
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                    for idx, ag in enumerate(self.agents_sim):
                        self.centralized_pretrain_replay_buffer.append((f'{e}_{i // self.action_interval}_{ag.id}', 
                                                               (last_obs[idx], last_phase[idx], actions[idx], rewards[idx], obs[idx], cur_phase[idx])))
                        # ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                        #             obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0

                    total_decision_num += 1
                    last_obs = obs
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    q_loss = self.train_shared_net(self.centralized_pretrain_replay_buffer)
                    #cur_loss_q = np.stack([ag.train() for ag in self.agents_sim])  # TODO: training

                    episode_loss.append(q_loss)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    self.agents_sim[0].update_target_network()
                    #[ag.update_target_network() for ag in self.agents_sim]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            self.logger.info("episode:{}/{}, pretrain sim avg travel time:{}".format(e, pretrained_episode,
                                                                             self.metric_sim.real_average_travel_time()))

            action_record.append(epo_action_record)
            states_record.append(epo_states_record)
            self.writeLog("TRAIN", e, self.metric_sim.real_average_travel_time(), \
                          mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                          self.metric_sim.throughput())

        model_path = self.agents_sim[0].save_model(e=-1)

        return model_path
    
    # TODO Add logging info to measure time of forward + inverse
    def decentralized_sim_train(self, episode, e, writer):
        action_diff = []
        action_distribution = []
        action_uncertainty = []
        total_decision_num = 0
        [ag.replay_buffer.clear() for ag in self.agents_sim]

        self.centralized_replay_buffer = deque(maxlen=self.buffer_size*len(self.agents_sim))

        # Sets threshold for inverse model
        if self.INVERSE == 'UNCERTAINTY':
            self.alpha = np.array([0.0])

        for ep in range(episode):
            epoch_diff = []
            epoch_distribution = []
            epoch_uncertainty = []

            self.metric_sim.clear()
            last_obs = self.env_sim.reset()
            episode_loss = []
            i = 0

            # Changed input size to accommodate decentralized multi-agent scenario
            input_size = (last_obs[0].shape[1] + 8) * self.HISTORY_T

            # Initialize sequence for each agent
            seq = [np.zeros((1, input_size)) for _ in range(len(self.agents_sim))]

            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])  # [agent, intersections]

                    actions_cold = []
                    actions = []
                    states = []
                    joint_action = []
                    actions_prob = []
                    joint_s_prime = []
                    grounding_actions = []

                    # Loop through agents gathering actions, action probs, and states
                    for idx, ag in enumerate(self.agents_sim):
                        # Get actions as one hot encoded over 8 possible actions
                        actions_cold.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        action_onehot = idx2onehot(np.array(ag.get_action(last_obs[idx], last_phase[idx], test=False)), 8)
                        actions.append(action_onehot)

                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                        states.append(last_obs[idx])

                        # Create forward input specific to this agent
                        # Construct the state-action pair for each agent
                        agent_state = np.concatenate([states[idx].flatten(), actions[idx].flatten()]).flatten()  # Flatten actions

                        seq[idx] = np.concatenate((seq[idx].flatten(), agent_state))  # Update sequence for this agent

                    # Joint action for environment step (centralized)
                    joint_action = np.concatenate(actions).flatten()

                    # Use the forward model for each agent
                    joint_s_prime = []
                    for idx in range(len(self.agents_sim)):
                        agent_input = torch.from_numpy(seq[idx][-input_size:]).float()
                        agent_input = agent_input.unsqueeze(0)
                        agent_s_prime = self.forward_models[idx].predict(agent_input)[0]
                        joint_s_prime.append(agent_s_prime)

                    # For each agent, calculate inverse model
                    for idx, ag in enumerate(self.agents_sim):

                        inverse_input = torch.cat((torch.tensor(states[idx]).to(self.device).float(), joint_s_prime[idx]), dim=1).float()

                        if self.INVERSE == 'NN':
                            logit_distribution, uncertainty = self.inverse_models[idx].predict(inverse_input)
                            distribution = F.softmax(logit_distribution)

                            action_max = np.array(np.argmax(logit_distribution.to(self.device).numpy()))
                            grounding_actions.append([np.array(action_max)])
                            grounding_actions = np.stack(grounding_actions)
                            epoch_diff.append([actions[idx][0], grounding_actions[idx][0]])
                            epoch_distribution.append(distribution.to(self.device).numpy())

                        # TODO Implement uncertainty for decentralized MARL
                        elif self.INVERSE == 'UNCERTAINTY':

                            if self.decentralized_GAT == "both":
                                uncertainty = torch.tensor(1.0).to(self.device)
                                actions_cold = np.array([actions_cold])
                                actions_cold = np.concatenate(actions_cold).flatten()
                            
                            else:

                                logit_distribution, uncertainty = self.inverse_models[idx].predict(inverse_input)

                                # Reshape to (batch_size, num_agents, 8)
                                logit_distribution = logit_distribution.view(-1, 1, 8)

                                # Assuming logit_distribution is your logits tensor
                                distribution = F.softmax(logit_distribution, dim=2)

                                # Get the action with the maximum probability for each agent in the batch
                                action_max = torch.argmax(distribution, dim=2).to(self.device)

                                # Clone max actions and remove grounded actions from computation graph
                                grounding_actions.append(action_max.clone().detach())  # Remove the list brackets

                                grounding_actions = torch.stack(grounding_actions).flatten().cpu().numpy()

                                actions_cold = np.concatenate(actions_cold).flatten()

                                epoch_distribution.append(distribution)
                                epoch_uncertainty.append(uncertainty)

                    if uncertainty.detach().item() < self.alpha:
                        print(f"Should not be called right now")
                        rewards_list = []
                        for _ in range(self.action_interval):
                            obs, rewards, dones, _ = self.env_sim.step(joint_action)  # Environment step
                            i += 1
                            rewards_list.append(np.stack(rewards))

                        rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]

                        self.metric_sim.update(rewards)
                        cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                        for idx, ag in enumerate(self.agents_sim):
                            ag.remember(last_obs[idx], last_phase[idx], grounding_actions[idx], actions_prob[idx],
                                        rewards[idx], obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')

                    else:
                        epoch_diff.append(joint_action)
                        rewards_list = []

                        for _ in range(self.action_interval):
                            obs, rewards, dones, _ = self.env_sim.step(actions_cold)  # Environment step
                            i += 1
                            rewards_list.append(np.stack(rewards))
                        rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]

                        self.metric_sim.update(rewards)
                        cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                        for idx, ag in enumerate(self.agents_sim):

                            self.centralized_replay_buffer.append((f'{e}_{i // self.action_interval}_{ag.id}', (last_obs[idx], last_phase[idx], actions[idx], rewards[idx], obs[idx], cur_phase[idx])))

                            # ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx],
                            #             rewards[idx], obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')

                    total_decision_num += 1
                    last_obs = obs

                # Modified to have all agent experiences train and update the main network
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    
                    #print(f"Agent: {idx}, step: {i}, total_decision_num: {total_decision_num}, self.update_model_rate: {self.update_model_rate}, replay buffer: {len(ag.replay_buffer)}, first: {ag.replay_buffer[0]}")
                    q_loss = self.train_shared_net(self.centralized_replay_buffer)

                    #cur_loss_q = np.stack([ag.train() for ag in self.agents_sim])
                    episode_loss.append(q_loss)

                # Modified to have all agent experiences train and update the target network
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    
                    self.agents_sim[0].update_target_network() # Update shared target network

                    #[ag.update_target_network() for ag in self.agents_sim]
            

                if all(dones):
                    break
            
            
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            action_diff.append(epoch_diff)
            action_distribution.append(epoch_distribution)

            if self.INVERSE == 'UNCERTAINTY':
                
                # TODO Implement Uncertanity for Decentralized MARL
                if self.decentralized_GAT == "both":
                    pass
                
                else:

                    temp = torch.vstack(epoch_uncertainty).flatten().mean().item()

                    action_uncertainty.append(temp)

            elif self.INVERSE == 'NN':
                temp = 0
                action_uncertainty.append(temp)

            self.logger.info("episode:{}/{}, sim avg travel time:{}".format(ep, episode,
                                                                            self.metric_sim.real_average_travel_time()))
            self.writeLog("TRAIN", ep, self.metric_sim.real_average_travel_time(),
                        mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                        self.metric_sim.throughput())
            

        #self.test(e=e)
        #writer.flush()
        act_dif = np.concatenate(action_diff)

        if self.INVERSE == 'UNCERTAINTY':
            # TODO Implement Uncertanity for Decentralized MARL
            if self.decentralized_GAT == "both":
                pass
            else:
                act_uncertainty = np.vstack(action_uncertainty)
                np.save(os.path.join(self.debug_path, f'act_uncertainty_{e}.npy'), act_uncertainty)
                self.alpha = np.mean(act_uncertainty)

        np.save(os.path.join(self.debug_path, f'act_diff_{e}.npy'), act_dif)
        model_path = self.agents_sim[0].save_model(e=e)
        return model_path
    

    # TODO Add logging info to measure time of forward + inverse
    def noGAT_sim_train(self, episode, e, writer):
        action_diff = []
        action_distribution = []
        total_decision_num = 0
        [ag.replay_buffer.clear() for ag in self.agents_sim]

        self.centralized_replay_buffer = deque(maxlen=self.buffer_size*len(self.agents_sim))

        # Sets threshold for inverse model
        if self.INVERSE == 'UNCERTAINTY':
            self.alpha = np.array([0.0])

        for ep in range(episode):
            epoch_diff = []
            epoch_distribution = []
            epoch_uncertainty = []

            self.metric_sim.clear()
            last_obs = self.env_sim.reset()
            episode_loss = []
            i = 0

            # Changed input size to accommodate decentralized multi-agent scenario
            input_size = (last_obs[0].shape[1] + 8) * self.HISTORY_T

            # Initialize sequence for each agent
            seq = [np.zeros((1, input_size)) for _ in range(len(self.agents_sim))]

            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])  # [agent, intersections]

                    actions_cold = []
                    actions = []
                    states = []
                    joint_action = []
                    actions_prob = []

                    # Loop through agents gathering actions, action probs, and states
                    for idx, ag in enumerate(self.agents_sim):
                        # Get actions as one hot encoded over 8 possible actions
                        actions_cold.append(ag.get_action_sharednet(self.agents_sim[0], last_obs[idx], last_phase[idx], test=False))
                        action_onehot = idx2onehot(np.array(ag.get_action_sharednet(self.agents_sim[0], last_obs[idx], last_phase[idx], test=False)), 8)
                        actions.append(action_onehot)

                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                        states.append(last_obs[idx])

                        # Create forward input specific to this agent
                        # Construct the state-action pair for each agent
                        agent_state = np.concatenate([states[idx].flatten(), actions[idx].flatten()]).flatten()  # Flatten actions

                        seq[idx] = np.concatenate((seq[idx].flatten(), agent_state))  # Update sequence for this agent

                    # Joint action for environment step (centralized)
                    joint_action = np.concatenate(actions).flatten()

                    # Same actions but not one hot for environment step (centralized)
                    actions_cold = np.concatenate(actions_cold).flatten()

                    epoch_diff.append(joint_action)
                    rewards_list = []

                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env_sim.step(actions_cold)  # Environment step
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]

                    self.metric_sim.update(rewards)
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                    for idx, ag in enumerate(self.agents_sim):

                        self.centralized_replay_buffer.append((f'{e}_{i // self.action_interval}_{ag.id}', (last_obs[idx], last_phase[idx], actions[idx], rewards[idx], obs[idx], cur_phase[idx])))

                    total_decision_num += 1
                    last_obs = obs

                # Modified to have all agent experiences train and update the main network
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    
                    q_loss = self.train_shared_net(self.centralized_replay_buffer)

                    episode_loss.append(q_loss)

                # Modified to have all agent experiences train and update the target network
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    
                    self.agents_sim[0].update_target_network() # Update shared target network
                
                if all(dones):
                    break
            
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            for idx, ag in enumerate(self.agents_sim):
                if idx == 0:
                    print(f"Agent: {idx}, Epsilon: {ag.epsilon}, Epsilon decay: {ag.epsilon_decay}, Epsilon min: {ag.epsilon_min}")

            action_diff.append(epoch_diff)
            action_distribution.append(epoch_distribution)

            self.logger.info("episode:{}/{}, sim avg travel time:{}".format(ep, episode,
                                                                            self.metric_sim.real_average_travel_time()))
            self.writeLog("TRAIN", ep, self.metric_sim.real_average_travel_time(),
                        mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                        self.metric_sim.throughput())
            

        act_dif = np.concatenate(action_diff)

        np.save(os.path.join(self.debug_path, f'act_diff_{e}.npy'), act_dif)
        model_path = self.agents_sim[0].save_model_no_print(e=e)
        return model_path
    

    # Train the shared network
    def train_shared_net(self, replay_buffer):

        active_agent = self.agents_sim[0]

        # Randomly sample from shared experience replay
        samples = random.sample(replay_buffer, 64)
        b_t, b_tp, rewards, actions = active_agent._batchwise(samples)

        # Use agent 1 train functionality to train shared network for convenience in code
        out = active_agent.target_model(b_tp, train=False)
        target = rewards + active_agent.gamma * torch.max(out, dim=1)[0]
        target_f = active_agent.model(b_t, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = active_agent.criterion(active_agent.model(b_t, train=True), target_f)
        active_agent.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(active_agent.model.parameters(), active_agent.grad_clip)
        active_agent.optimizer.step()

        # Modify the decay rate for all agents at the same time for the multi-agent case
        if active_agent.epsilon > active_agent.epsilon_min:
            active_agent.epsilon *= active_agent.epsilon_decay

        return loss.clone().detach().numpy()

    
    # Loading is set up for parameter sharing, all agents share a network
    def load_shared(self, model_path):

        base_path = sys.path[0] + "/"
        real_path = base_path + model_path
        [ag.load_shared_model(self.agents_sim[0].learning_rate, e="", customized_path=real_path) for ag in self.agents_sim]
        [ag.load_shared_model(self.agents_sim[0].learning_rate, e="", customized_path=real_path) for ag in self.agents_real]
        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("timestamp:{} successfully loaded a model with param {}".format(time_stamp, model_path))

    # Now supports multi-agent setting
    def sim_rollout(self, e, decentralized=False):
        self.metric_sim.clear()
        obs = self.env_sim.reset()

        # Separate records for each agent in decentralized mode
        action_records = [[] for _ in self.agents_sim]  # Separate actions for each agent
        states_records = [[] for _ in self.agents_sim]  # Separate states for each agent
        joint_action_record = []  # Joint action record for centralized case
        joint_states_record = []  # Joint states record for centralized case

        for i in range(self.steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_sim])
                actions = []
                for idx, ag in enumerate(self.agents_sim):
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    actions.append(action)

                joint_action = np.concatenate(actions)  # Concatenate actions for joint case
                joint_action_record.append(joint_action)  # For centralized storage

                for idx, ag in enumerate(self.agents_sim):
                    states_records[idx].append(obs[idx])  # Individual agent states
                    action_records[idx].append(actions[idx])  # Individual agent actions

                # Store joint state only for centralized case
                if not decentralized:
                    joint_state = np.concatenate(obs)
                    joint_states_record.append(joint_state)

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_sim.step(joint_action)  # Use joint action
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_sim.update(rewards)

            if all(dones):
                break

        if decentralized:
            # Save individual datasets for each agent
            for idx, (state_rec, action_rec) in enumerate(zip(states_records, action_records)):
                file_path = os.path.join(self.data_path, f'sim_{idx+1}.pkl')
                with open(file_path, 'wb') as f:
                    pkl.dump([[state_rec], [action_rec]], f)
        else:
            # Save joint dataset
            with open(os.path.join(self.data_path, 'sim.pkl'), 'wb') as f:
                pkl.dump([[joint_states_record], [joint_action_record]], f)

        self.logger.info(
            "Epoch %d Sim Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
                e,
                self.metric_sim.real_average_travel_time(),
                self.metric_sim.rewards(),
                self.metric_sim.queue(),
                self.metric_sim.delay(),
                self.metric_sim.throughput()))
        return

    # Now supports multi-agent setting
    def real_rollout(self, e, decentralized=False):
        discount = 0.95
        self.metric_real.clear()
        R = np.array([0 for _ in self.agents_real], dtype=np.float64)
        obs = self.env_real.reset()
        action_records = [[] for _ in self.agents_real]  # Separate records for each agent
        states_records = [[] for _ in self.agents_real]  # Separate records for each agent
        joint_action_record = []  # Joint action record for centralized case
        joint_states_record = []  # Joint states record for centralized case

        for a in self.agents_real:
            a.reset()

        for i in range(self.steps):
            discount *= discount
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = []
                for idx, ag in enumerate(self.agents_real):
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    actions.append(action)

                joint_action = np.concatenate(actions)  # Concatenate actions for joint case
                joint_action_record.append(joint_action)  # For centralized storage

                for idx, ag in enumerate(self.agents_real):
                    states_records[idx].append(obs[idx])  # Individual agent states
                    action_records[idx].append(actions[idx])  # Individual agent actions

                # Store joint state only for centralized case
                if not decentralized:
                    joint_state = np.concatenate(obs)
                    joint_states_record.append(joint_state)

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(joint_action)  # Use joint action
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                R += discount * rewards
                self.metric_real.update(rewards)

            if all(dones):
                break

        if decentralized:
            # Save individual datasets for each agent
            for idx, (state_rec, action_rec) in enumerate(zip(states_records, action_records)):
                file_path = os.path.join(self.data_path, f'real_{idx+1}.pkl')
                with open(file_path, 'wb') as f:
                    pkl.dump([[state_rec], [action_rec]], f)
        else:
            # Save joint dataset
            with open(os.path.join(self.data_path, 'real.pkl'), 'wb') as f:
                pkl.dump([[joint_states_record], [joint_action_record]], f)

        self.logger.info(
            "Epoch %d Real Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
                e,
                self.metric_real.real_average_travel_time(),
                self.metric_real.rewards(),
                self.metric_real.queue(),
                self.metric_real.delay(),
                self.metric_real.throughput()))
        return R
    

    def precollect_trajectories(self, num_episodes, save_path="precollected.pkl"):
        """
        Collect trajectories from a specified number of episodes and save them as a list of state-action pairs.

        :param num_episodes: Number of episodes to collect trajectories from.
        :param save_path: Path to save the collected trajectories as a pickle file.
        """
        trajectories = []  # To store all trajectories

        for e in range(num_episodes):
            self.metric_real.clear()
            obs = self.env_real.reset()
            for a in self.agents_real:
                a.reset()

            episode_trajectory = []  # To store state-action pairs for the current episode
            for step in range(self.steps):
                if step % self.action_interval == 0:
                    # Get actions from all agents
                    phases = np.stack([ag.get_phase() for ag in self.agents_real])
                    actions = [
                        ag.get_action(obs[idx], phases[idx], test=True)
                        for idx, ag in enumerate(self.agents_real)
                    ]
                    
                    # Concatenate actions into a joint action vector
                    joint_action = np.concatenate(actions)

                    # Store state-action pair
                    joint_state = np.concatenate(obs)  # Combine all agent observations into one state
                    episode_trajectory.append((joint_state.flatten(), joint_action.flatten()))  # Save state-action pair

                    # Step the environment
                    for _ in range(self.action_interval):
                        obs, _, done, _ = self.env_real.step(joint_action)


                    if all(done):
                        break

            # Add the episode trajectory to the list of all trajectories
            trajectories.append(episode_trajectory)

        # Save all collected trajectories to a pickle file
        with open(save_path, 'wb') as f:
            pkl.dump(trajectories, f)

        self.logger.info(f"Precollected {num_episodes} episodes and saved to {save_path}")

        return



    def forward_train(self, writer, transfer_calc=False):
        print(f"SHOULD NOT BE HERE")
        if self.decentralized_GAT == "both":
            # Iterate over each agent's forward model
            for i, forward_model in enumerate(self.forward_models):
                # Generate the dataset path for this agent
                dataset_path = f"{self.data_path}real_{i+1}.pkl"
                
                # Load the dataset for the agent's forward model
                forward_model.load_dataset(dataset_path)
                
                # Optionally calculate transfer loss if enabled
                if transfer_calc:
                    forward_model.calculate_transfer_loss("precollected.pkl", 1, len(self.agents_real))
                
                # Train the forward model
                forward_model.train(100, writer=writer, sign=f'forward', agent_num=i+1)
        else:
            # Centralized training logic remains unchanged
            self.forward_model.load_dataset()
            if transfer_calc:
                self.forward_model.calculate_transfer_loss("precollected.pkl", 1, len(self.agents_real))
            self.forward_model.train(100, writer=writer, sign='forward')


    def inverse_train(self, writer):
        if self.decentralized_GAT == "both":
            uncertainty_lists = []
            
            # Iterate over each agent's inverse model
            for i, inverse_model in enumerate(self.inverse_models):
                # Generate the dataset path for this agent
                dataset_path = f"{self.data_path}sim_{i+1}.pkl"
                
                # Load the dataset for the agent's inverse model
                inverse_model.load_dataset(dataset_path)
                
                # Train the inverse model and get uncertainty feedback
                uncertainty_list = inverse_model.train(100, writer=writer, sign=f'inverse', agent_num=i+1)
                
                # Collect the uncertainty feedback for all agents
                uncertainty_lists.append(uncertainty_list)
            
            return uncertainty_lists
        else:
            # Centralized training logic remains unchanged
            self.inverse_model.load_dataset()
            # Get the uncertainty feedback from inverse training
            uncertainty_list = self.inverse_model.train(100, writer=writer, sign='inverse')
            return uncertainty_list


    def sim_tain(self, episode, e, writer):
        action_diff = []
        action_distribution = []
        action_uncertainty = []

        # TODO: add diff of a and a_grounding
        # TODO: transfer a prob to a_hat
        total_decision_num = 0
        # [ag.load_model(e='', customized_path=path) for ag in self.agents_sim]
        [ag.replay_buffer.clear() for ag in self.agents_sim]

        # Sets threshold for inverse model
        if self.INVERSE == 'UNCERTAINTY':
            #self.alpha = np.zeros(1)
            self.alpha = np.array([0.0])

        for ep in range(episode):

            #print(f"self.alpha: {self.alpha}")

            epoch_diff = []
            epoch_distribution = []
            epoch_uncertainty = []
           
            self.metric_sim.clear()
            last_obs = self.env_sim.reset()
            episode_loss = []
            i = 0

            # Added number of agents to modify input size for multi-agent setting
            num_agents = len(self.agents_sim)

            # Changed input size to accomodate multi-agent scenario
            input_size = (last_obs[0].shape[1] * num_agents + 8 * num_agents) * self.HISTORY_T

            seq = np.zeros((1, input_size))
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])  # [agent, intersections]

                    actions = []
                    actions_cold = []
                    states = []
                    joint_action = []
                    actions_prob = []
                    joint_s_prime = []
                    grounding_actions = []

                    # Loop through agents gathering actions, action probs, and states
                    # Now supports multi-agent setting and only using a single for loop
                    for idx, ag in enumerate(self.agents_sim):

                        # Get actions as one hot encoded over 8 possible actions
                        actions_cold.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        action_onehot = idx2onehot(np.array(ag.get_action(last_obs[idx], last_phase[idx], test=False)), 8)
                        actions.append(action_onehot)

                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                        states.append(last_obs[0])

                    # Concatenate joint state for multi-agent setting
                    joint_state = np.concatenate(states).flatten()

                    # Added joint action for multi-agent setting
                    joint_action = np.concatenate(actions).flatten()

                    # Forward input is now joint state and joint action to support multi-agent scenario
                    forward_input = np.concatenate((joint_state, joint_action)).reshape(1, -1)

                    seq = np.concatenate((seq, forward_input), axis=1)

                    joint_s_prime.append(self.forward_model.predict(torch.from_numpy(seq[:, -input_size :]).float())[0])

                    # Move tensors to GPU
                    s_prime_tensor = joint_s_prime[0].to(self.device)
                    last_obs_tensor = torch.from_numpy(joint_state).to(self.device).float()

                    # Resize last obs tensor to match s_prime_tensor (1, num agents * num observations per agent)
                    last_obs_tensor = last_obs_tensor.unsqueeze(0)

                    inverse_input = torch.cat((last_obs_tensor, s_prime_tensor), dim=1).float()

                    # Moved this block outside of the for loop to compute once for all agents instead of once per agent
                    if self.INVERSE == 'NN':
                        #print("NN")
                        logit_distribution, uncertainty = self.inverse_model.predict(inverse_input) #uncertainty is just holding a place here, not actually working
                        distribution = F.softmax(logit_distribution)

                        action_max = np.array(np.argmax(logit_distribution.to(self.device).numpy()))
                        grounding_actions.append([np.array(action_max)])
                        grounding_actions = np.stack(grounding_actions)
                        epoch_diff.append([actions[0][0], grounding_actions[0][0]])
                        epoch_distribution.append(distribution.to(self.device).numpy())

                    # Inverse model generates a single uncertainity value for estep in every epoch
                    elif self.INVERSE =='UNCERTAINTY':

                        # TODO This assumes uncertainty is calculated as a singular value for the joint grounded action (grounded actions for all agents)
                        logit_distribution, uncertainty = self.inverse_model.predict(inverse_input)

                        # Reshape to (batch_size, num_agents, num_actions)
                        logit_distribution = logit_distribution.view(-1, num_agents, 8)

                        # Assuming logit_distribution is your logits tensor
                        distribution = F.softmax(logit_distribution, dim=2)

                        # Get the action with the maximum probability for each agent in the batch
                        action_max = torch.argmax(distribution, dim=2).to(self.device)

                        # Clone max actions and remove grounded actions from computation graph
                        grounding_actions.append(action_max.clone().detach())  # Remove the list brackets

                        grounding_actions = torch.stack(grounding_actions).flatten().cpu().numpy()

                        actions_cold = np.concatenate(actions_cold).flatten()

                        # Now uses joint action for epoch diff in the multi-agent setting
                        #epoch_diff.append([actions_cold, grounding_actions])

                        epoch_distribution.append(distribution)
                        epoch_uncertainty.append(uncertainty)

                    # if uncertainty.detach().numpy()[0] < np.array(1): #self.alpha
                    #if uncertainty.detach().numpy()[0] < self.alpha: #self.alpha

                    #print(f"self.alpha: {self.alpha}, uncertainty.detach().item(): {uncertainty.detach().item()}")

                    if uncertainty.detach().item() < self.alpha:
                        # take grounding action

                        rewards_list = []
                        for _ in range(self.action_interval):
                            obs, rewards, dones, _ = self.env_sim.step(grounding_actions.flatten())
                            i += 1
                            rewards_list.append(np.stack(rewards))

                        # Commented out because it is not being used
                        #seq[:, -8:] = idx2onehot(grounding_actions[0], 8)


                        rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]

                        self.metric_sim.update(rewards)
                        cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                        for idx, ag in enumerate(self.agents_sim):
                            ag.remember(last_obs[idx], last_phase[idx], grounding_actions[idx], actions_prob[idx],
                                        rewards[idx],
                                        obs[idx], cur_phase[idx], dones[idx],
                                        f'{e}_{i // self.action_interval}_{ag.id}')

                    else:
                        # take original action
                        epoch_diff.append(joint_action)
                        rewards_list = []
                        
                        # Commenting out not sure why here
                        #forward_input = np.concatenate((last_obs[0], idx2onehot(actions[0], 8)), axis=1)
                        #seq = np.concatenate((seq, forward_input), axis=1)

                        # Changed actions to joint action to support multi-agent setting
                        for _ in range(self.action_interval):
                            obs, rewards, dones, _ = self.env_sim.step(actions_cold)
                            i += 1
                            rewards_list.append(np.stack(rewards))
                        rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]

                        self.metric_sim.update(rewards)
                        cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                        for idx, ag in enumerate(self.agents_sim):
                            ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx],
                                        rewards[idx],
                                        obs[idx], cur_phase[idx], dones[idx],
                                        f'{e}_{i // self.action_interval}_{ag.id}')

                    total_decision_num += 1
                    last_obs = obs
                
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.train() for ag in self.agents_sim])  # TODO: training
                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents_sim]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            action_diff.append(epoch_diff)
            action_distribution.append(epoch_distribution)

            if self.INVERSE == 'UNCERTAINTY':
                #temp = np.vstack(epoch_uncertainty).flatten().mean()
                temp = torch.vstack(epoch_uncertainty).flatten().mean().item()

                action_uncertainty.append(temp)

            elif self.INVERSE == 'NN':
                temp = 0
                action_uncertainty.append(temp)
                
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(ep, episode,
                                                                             self.metric_sim.real_average_travel_time()))
            self.writeLog("TRAIN", ep, self.metric_sim.real_average_travel_time(), \
                          mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                          self.metric_sim.throughput())

            # writer.add_scalar("real_a_travel time/simu_train", self.metric_sim.real_average_travel_time(), e)
            # writer.add_scalar("real_throughput/simu_train", self.metric_sim.throughput(), e)
            # writer.add_scalar("rewards/simu_train", self.metric_sim.rewards(), e)
            # writer.add_scalar("queue/simu_train", self.metric_sim.queue(), e)
            # writer.add_scalar("delay/simu_train", self.metric_sim.queue(), e)
            # writer.add_scalar("mean_loss/simu_train", mean_loss, e)
            # writer.add_scalar("uncertanty/act_uncertainty_epo", temp, e)

        self.test(e=e)
        writer.flush()
        act_dif = np.concatenate(action_diff)
        # act_distribution = np.vstack(action_distribution)
        if self.INVERSE == 'UNCERTAINTY':
            act_uncertainty = np.vstack(action_uncertainty)
            np.save(os.path.join(self.debug_path, f'act_uncertainty_{e}.npy'), act_uncertainty)
            self.alpha = np.mean(act_uncertainty)

        np.save(os.path.join(self.debug_path, f'act_diff_{e}.npy'), act_dif)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        model_path = self.agents_sim[-1].save_model(e=e)
        return model_path
    
    


    def real_eval(self, e):
        # TODO: already supports multiple intersections, 
        discount = 0.95
        self.metric_real.clear()
        R = np.array([0 for _ in self.agents_real])
        obs = self.env_real.reset()
        for a in self.agents_real:
            a.reset()
        for i in range(self.test_steps):
            discount *= discount
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = []
                for idx, ag in enumerate(self.agents_real):
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    actions.append(action)

                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                R += discount * rewards
                self.metric_real.update(rewards)
            if all(dones):
                break

        self.logger.info(
            "Final Real Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
                self.metric_real.real_average_travel_time(), \
                self.metric_real.rewards(), self.metric_real.queue(), self.metric_real.delay(),
                self.metric_real.throughput()))
        return R

    def test(self, e):
        self.metric_sim.clear()
        obs = self.env_sim.reset()
        for i in range(self.steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_sim])
                actions = []
                for idx, ag in enumerate(self.agents_sim):
                    action = ag.get_action(obs[idx], phases[idx], test=True)
                    actions.append(action)

                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_sim.update(rewards)
            if all(dones):
                break

        self.logger.info(
            "Eval sim Travel Time Epoch: %d is %.4f, throughput: %d" % (e, self.metric_sim.real_average_travel_time(), self.metric_sim.throughput()))

        return

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" + \
              "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()
