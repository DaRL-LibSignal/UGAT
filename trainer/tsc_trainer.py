import os
import sys
import pickle as pkl
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from agent.utils import idx2onehot
from trainer.base_trainer import BaseTrainer
import datetime
from common.stat_utils import log_passing_lane_actinon, write_action_record
import torch
import torch.optim as optim
import random


print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.device_count()) # Number of GPUs detected

from common.gat_utils import load_and_split_forward_data, load_and_split_inverse_data, NN_predictor, UNCERTAINTY_predictor, PKLDataset


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''

    def __init__(
            self,
            logger,
            gpu=0,
            cpu=False,
            name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.training_iterations = Registry.mapping['trainer_mapping']['setting'].param['training_iterations']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']

        self.gat = Registry.mapping['trainer_mapping']['setting'].param['gat']
        self.gattype = Registry.mapping['trainer_mapping']['setting'].param['gattype']
        self.uncertainty_setting = Registry.mapping['trainer_mapping']['setting'].param['uncertainty']
        self.delayedgat = Registry.mapping['trainer_mapping']['setting'].param['delayedgat']
        self.grounding_pattern = Registry.mapping['trainer_mapping']['setting'].param['grounding_pattern']
        self.ground_original = Registry.mapping['trainer_mapping']['setting'].param['ground_original']
        self.last_n_uncertainties = Registry.mapping['trainer_mapping']['setting'].param['last_n_uncertainties']
        self.prob_grounding = Registry.mapping['trainer_mapping']['setting'].param['prob_grounding']
        self.network_version = Registry.mapping['trainer_mapping']['setting'].param['network_version']

        self.net = Registry.mapping['trainer_mapping']['setting'].param['network']
        self.load_pretrained = Registry.mapping['trainer_mapping']['setting'].param['load_pretrained']
        
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later

        # TODO: support other dataset in the future
        self.create()
        self.dataset = Registry.mapping['dataset_mapping'][
            Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
                                         '_BRF.log').rstrip(
                                         '_ACT.log') + '_DTL.log'
                                     )

        self.action_log_file = os.path.join(
                Registry.mapping['logger_mapping']['path'].path,
                Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log').rstrip('_DTL.log') + '_ACT.log'
            )


        # Path to the folder
        path = 'collected'
        
        # Check if the folder exists
        if os.path.exists(path):
            # Iterate through all files in the folder
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                # Remove each file if it exists and is a file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"{file_path} has been deleted")
            print("All files in the 'collected' folder have been deleted.")
        else:
            print(f"The folder '{path}' does not exist.")

        self.total_decision_num = 0
        
        # Initialize GAT models
        if self.gat == True:

            self.forward_models = []
            self.inverse_models = []

            self.total_decision_num = 0
            self.mean_uncertainty = 0
            # Dictionary to store the last two uncertainties for each agent
            self.last_two_uncertainties = {idx: [] for idx in range(len(self.agents_sim))}
            self.avg_agent_uncertainties = [0 for i in range(len(self.agents_sim))]

            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # Considers number of agents in both input and output dimension calculations to allow for multi-agent setting
            num_agents = len(self.agents_real)
            
            # Initialize centralized GAT models
            if self.gattype == "centralized":
                self.last_two_central_uncertainties = []
                print(f"\n------- INITIALIZING GAT MODELS CENTRALIZED -------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
                self.forward_model = NN_predictor(self.logger,
                                                (num_agents, self.agents_real[0].ob_generator.ob_length), (num_agents, self.agents_real[0].action_space.n),
                                                self.agents_real[0].ob_generator.ob_length, self.device, gat_path, 'collected/ereal_train_full.pkl', False, 1, 'central')
                self.inverse_model = UNCERTAINTY_predictor(self.logger, (num_agents, self.agents_real[0].ob_generator.ob_length), 0, 0, (num_agents, self.agents_real[0].ob_generator.ob_length), self.agents_real[0].action_space.n, self.device, gat_path,
                                                'collected/esim_train_full.pkl', backward=True, history=1, mode='central')
            # Initialize decentralized GAT models
            elif self.gattype == "decentralized":
                print(f"\n------- INITIALIZING GAT MODELS DECENTRALIZED -------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')

                # Single state and action for forward model and single states for inverse model input, identical to vanilla GAT
                for i in range(num_agents):
                    self.forward_model = NN_predictor(self.logger,
                                                    (1, self.agents_real[0].ob_generator.ob_length), (1, self.agents_real[0].action_space.n),
                                                    self.agents_real[0].ob_generator.ob_length, self.device, gat_path, 'collected/ereal_train_full.pkl')
                    self.inverse_model = UNCERTAINTY_predictor(self.logger, (1, self.agents_real[0].ob_generator.ob_length), 0, 0, (1, self.agents_real[0].ob_generator.ob_length), self.agents_real[0].action_space.n, self.device, gat_path, 'collected/esim_train_full.pkl', backward=True, history=1, mode='dec')
                    
                    self.forward_models.append(self.forward_model)
                    self.inverse_models.append(self.inverse_model)

            # Initialize JL-GAT models
            elif self.gattype == "jlgat":
                print(f"\n------- INITIALIZING JL-GAT MODELS -------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
                
                # Hardcoded values for # of neighbors for each agent until I fix later
                if self.net == "cityflow1x3":
                    self.agents_real[0].neighbors = 2
                    self.agents_real[1].neighbors = 3
                    self.agents_real[2].neighbors = 2

                # Hardcoded values for # of neighbors for each agent until I fix later
                elif self.net == "cityflow4x4":
                
                    # Bottom row
                    self.agents_real[0].neighbors = 3
                    self.agents_real[1].neighbors = 4
                    self.agents_real[2].neighbors = 4
                    self.agents_real[3].neighbors = 3

                    # 2nd from bottom row
                    self.agents_real[4].neighbors = 4
                    self.agents_real[5].neighbors = 5
                    self.agents_real[6].neighbors = 5
                    self.agents_real[7].neighbors = 4

                    # 3rd from bottom row
                    self.agents_real[8].neighbors = 4
                    self.agents_real[9].neighbors = 5
                    self.agents_real[10].neighbors = 5
                    self.agents_real[11].neighbors = 4

                    # Top row
                    self.agents_real[12].neighbors = 3
                    self.agents_real[13].neighbors = 4
                    self.agents_real[14].neighbors = 4
                    self.agents_real[15].neighbors = 3
                
                for idx, ag in enumerate(self.agents_real):

                    # Forward model outputs a single predicted next state based on joint local information
                    # Initialized with the following dimensions: Joint local State: (Agent + Neighbors, State size), Joint local action: (Agent + Neighbors, Action size)
                    if self.network_version == 4:
                        self.forward_model = NN_predictor(self.logger,
                                                (ag.neighbors, self.agents_real[0].ob_generator.ob_length), (1, self.agents_real[0].action_space.n),
                                                self.agents_real[0].ob_generator.ob_length, self.device, gat_path, 'collected/ereal_train_full.pkl', backward=False, history=1, mode='jlgat4')
                    elif self.network_version == 5:
                        self.forward_model = NN_predictor(self.logger,
                                                (1, self.agents_real[0].ob_generator.ob_length), (ag.neighbors, self.agents_real[0].action_space.n),
                                                self.agents_real[0].ob_generator.ob_length, self.device, gat_path, 'collected/ereal_train_full.pkl', backward=False, history=1, mode='jlgat5')
                    else:
                        self.forward_model = NN_predictor(self.logger,
                                                (ag.neighbors, self.agents_real[0].ob_generator.ob_length), (ag.neighbors, self.agents_real[0].action_space.n),
                                                self.agents_real[0].ob_generator.ob_length, self.device, gat_path, 'collected/ereal_train_full.pkl')

                    # Inverse model outputs a single predicted action based on joint local information (also added actions of neighbors to inverse model, assuming they're fixed)
                    if self.network_version == 2:
                        self.inverse_model = UNCERTAINTY_predictor(self.logger, (ag.neighbors, self.agents_real[0].ob_generator.ob_length), 0, 0, (1, self.agents_real[0].ob_generator.ob_length), self.agents_real[0].action_space.n, self.device, gat_path, 'collected/esim_train_full.pkl', backward=True, history=1, mode='wo_action')
                    elif self.network_version == 3:
                        self.inverse_model = UNCERTAINTY_predictor(self.logger, (1, self.agents_real[0].ob_generator.ob_length), 0, (ag.neighbors - 1, self.agents_real[0].action_space.n), (1, self.agents_real[0].ob_generator.ob_length), self.agents_real[0].action_space.n, self.device, gat_path, 'collected/esim_train_full.pkl', backward=True, history=1, mode='wo_state')
                    else:
                        self.inverse_model = UNCERTAINTY_predictor(self.logger, (1, self.agents_real[0].ob_generator.ob_length), (ag.neighbors - 1, self.agents_real[0].ob_generator.ob_length), (ag.neighbors - 1, self.agents_real[0].action_space.n), (1, self.agents_real[0].ob_generator.ob_length),
                                                        self.agents_real[0].action_space.n, self.device, gat_path,
                                                        'collected/esim_train_full.pkl', backward=True)
                    
                    self.forward_models.append(self.forward_model)
                    self.inverse_models.append(self.inverse_model)
                    

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

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''

        self.agents_sim = []
        self.agents_real = []

        
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

    def train_flow(self):
        '''
        Main training flow
        '''
        if self.delayedgat == True:
            # Run for a set number of episodes
            for e in range(self.episodes):
        
                # Sim rollout + collect data
                self.sim_rollout(e, self.gattype)
        
                # Real rollout + collect data
                self.train_test(e, self.gattype)
        
                # Update GAT models
                self.gat_training(e)

                # Delay GAT training until episode 150
                if e < 200:
                    # Run regular policy training for some number of iterations
                    self.gat = False
                    self.policy_training(e)
                    self.gat = True
                else:
                    # Run GAT policy training for some number of iterations
                    self.gat = True
                    self.policy_training(e)

        else:

            if self.load_pretrained:
                for ag in self.agents_sim:
                    ag.load_model(0, True, self.net)
                    ag.optimizer = optim.RMSprop(ag.model.parameters(),
                                       lr=ag.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)
            
            # Run for a set number of episodes
            for e in range(self.episodes):
        
                # Sim rollout + collect data
                self.sim_rollout(e, self.gattype)
        
                # Real rollout + collect data
                self.train_test(e, self.gattype)
        
                # Update GAT models
                self.gat_training(e)
    
                # Run policy training for some number of iterations
                self.policy_training(e)


    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''

        total_decision_num = 0
        flush = 0

        for e in range(self.episodes):
            # TODO: check this reset agent
            self.metric_sim.clear()
            last_obs = self.env_sim.reset()  # agent * [sub_agent, feature]
            state_action_next_state = []
            for a in self.agents_sim:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env_sim.eng.set_save_replay(True)
                    self.env_sim.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env_sim.eng.set_save_replay(False)
            
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents_sim):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
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

                    state_action_next_state.append((last_obs, actions, obs))
                    
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                    for idx, ag in enumerate(self.agents_sim):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                                    obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
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

            self.writeLog("TRAIN", e, self.metric_sim.real_average_travel_time(), \
                          mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                          self.metric_sim.throughput())
            self.logger.info(
                "step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps, \
                                                                                              mean_loss,
                                                                                              self.metric_sim.rewards(),
                                                                                              self.metric_sim.queue(),
                                                                                              self.metric_sim.delay(),
                                                                                              int(self.metric_sim.throughput())))
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes,
                                                                             self.metric_sim.real_average_travel_time()))
            for j in range(len(self.world_sim.intersections)):
                self.logger.debug(
                    "intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric_sim.lane_rewards()[j], \
                                                                                    self.metric_sim.lane_queue()[j]))

            # Test policy in test environment and record ATT
            if self.test_when_train:
                self.train_test(e)

            # Save the data for the current episode
            file_path = 'collected/esim_train.pkl'
            
            # Save new data directly to the file
            with open(file_path, 'wb') as f:
                pkl.dump(state_action_next_state, f)

        # Save models after training
        [ag.save_model(e=self.episodes) for ag in self.agents_sim]


    def policy_training(self, episode):
        """
        Train the agent(s) without saving data or collecting it, and log the iteration number.
    
        :param episode: int, the current episode number for logging
        :return: None
        """
        flush = 0
    
        for e in range(self.training_iterations):
            uncertainty_sum = 0
            agent_uncertainty_sums = [0 for i in range(len(self.agents_sim))]
            ga_by_agent = [0 for i in range(len(self.agents_sim))]
            grounded_action_count = 0
            self.metric_sim.clear()
            last_obs = self.env_sim.reset()
            for a in self.agents_sim:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                self.env_sim.eng.set_save_replay(False)
    
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
    
                    if self.total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents_sim):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        actions = np.stack(actions)
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents_sim])
    
                    actions_prob = [ag.get_action_prob(last_obs[idx], last_phase[idx]) for idx, ag in enumerate(self.agents_sim)]

                    original_actions = actions.copy()
                    grounded_actions = [9 for i in range(len(self.agents_sim))]
    
                    if self.gat:
                        if self.gattype == "centralized":
                            one_hot_actions = np.concatenate([
                                idx2onehot(np.array([action]), 8) for action in actions
                            ], axis=0)
                            
                            state_tensor = torch.tensor(np.array(last_obs)).squeeze(1).unsqueeze(0).float().to(self.device)
                            action_tensor = torch.tensor(one_hot_actions).unsqueeze(0).float().to(self.device)

                            pred_next_state = self.forward_model.model(state_tensor, action_tensor)
    
                            result = self.inverse_model.model(state_tensor, pred_next_state)
                            grounded_action, uncertainty = result[0], result[1]

                            if self.uncertainty_setting == True:
                                uncertainty_sum += uncertainty.item()
                                if uncertainty < self.mean_uncertainty:
                                    grounded_action_reshaped = grounded_action.view(len(self.agents_sim), 8)
                                    actions = torch.argmax(grounded_action_reshaped, dim=1).cpu().numpy()
                                    grounded_actions = actions
                                    grounded_action_count += len(self.agents_sim)
                            else:
                                grounded_action_reshaped = grounded_action.view(len(self.agents_sim), 8)
                                
                                actions = torch.argmax(grounded_action_reshaped, dim=1).cpu().numpy()
                                
                                grounded_actions = actions
                                grounded_action_count += len(self.agents_sim)
                                
                                for j in range(len(ga_by_agent)):
                                    ga_by_agent[j] += 1


                        elif self.gattype == "decentralized":
                            for idx, ag in enumerate(self.agents_sim):
                                individual_state = last_obs[idx]
                                individual_action = idx2onehot(np.array([actions[idx]]), 8)

                                
                                individual_state = torch.from_numpy(individual_state).float().to(self.device).unsqueeze(0)
                                individual_action = torch.from_numpy(individual_action).float().to(self.device).unsqueeze(0)
                                
                                pred_next_state = self.forward_models[idx].model(individual_state, individual_action).unsqueeze(0)
    
                                result = self.inverse_models[idx].model(individual_state, pred_next_state)
                                
                                grounded_action, uncertainty = result[0], result[1]

                                if self.uncertainty_setting == True:
                                    agent_uncertainty_sums[idx] += uncertainty.item()
                                    if uncertainty < self.avg_agent_uncertainties[idx]:
                                        actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                        
                                        grounded_actions[idx] = actions[idx]
                                        grounded_action_count += 1
                                        ga_by_agent[idx] += 1
                                else:
                                    actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                    
                                    grounded_actions[idx] = actions[idx]
                                    grounded_action_count += 1
                                    ga_by_agent[idx] += 1

                        # Currently setup for 1x3 only
                        elif self.gattype == "jlgat":
                            
                            if self.net == "cityflow1x3":
                                for idx, ag in enumerate(self.agents_sim):
                                            
                                    # Ground based upon neighbor static actions (avoid cascade)
                                    
                                    if idx == 0:  # Agent 0: Uses its own state + agent 1's state + its own & agent 1's actions
                                        relevant_states = np.concatenate([last_obs[0], last_obs[1]])
                                        relevant_actions = np.concatenate([
                                        idx2onehot(np.array([actions[0]]), 8),
                                        idx2onehot(np.array([actions[1]]), 8)
                                        ])

                                        ind_action = idx2onehot(np.array([actions[0]]), 8)

                                        ind_state = last_obs[0]
                                        neighbor_states = last_obs[1]
                                        
                                        neighbor_actions = idx2onehot(np.array([actions[1]]), 8)
                                        
                                    elif idx == 1:  # Agent 1: Uses all agent states and actions
                                        relevant_states = np.concatenate([last_obs[0], last_obs[1], last_obs[2]])
                                        relevant_actions = np.concatenate([
                                        idx2onehot(np.array([actions[0].cpu().numpy()]) if isinstance(actions[0], torch.Tensor) else np.array([actions[0]]), 8),
                                        idx2onehot(np.array([actions[1].cpu().numpy()]) if isinstance(actions[1], torch.Tensor) else np.array([actions[1]]), 8),
                                        idx2onehot(np.array([actions[2].cpu().numpy()]) if isinstance(actions[2], torch.Tensor) else np.array([actions[2]]), 8)
                                    ])

                                        ind_state = last_obs[1]
                                        neighbor_states = np.concatenate([last_obs[0], last_obs[2]])

                                        ind_action = idx2onehot(np.array([actions[1]]), 8)

                                        neighbor_actions = np.concatenate([
                                        idx2onehot(np.array([actions[0].cpu().numpy()]) if isinstance(actions[0], torch.Tensor) else np.array([actions[0]]), 8),
                                        idx2onehot(np.array([actions[2].cpu().numpy()]) if isinstance(actions[2], torch.Tensor) else np.array([actions[2]]), 8)
                                        ])
                                            
                                    elif idx == 2:  # Agent 2: Uses its own state + agent 1's state + its own & agent 1's actions
                                        relevant_states = np.concatenate([last_obs[1], last_obs[2]])
                                        relevant_actions = np.concatenate([
                                        idx2onehot(np.array([actions[1].cpu().numpy()]) if isinstance(actions[1], torch.Tensor) else np.array([actions[1]]), 8),
                                        idx2onehot(np.array([actions[2].cpu().numpy()]) if isinstance(actions[2], torch.Tensor) else np.array([actions[2]]), 8)
                                        ])

                                        ind_state = last_obs[2]
                                        neighbor_states = last_obs[1]

                                        ind_action = idx2onehot(np.array([actions[2]]), 8)
                                        
                                        neighbor_actions = idx2onehot(np.array([actions[1]]), 8)

                                    
                                    # Create tensors for use in input
                                    relevant_states = torch.from_numpy(relevant_states).float().to(self.device).unsqueeze(0)
                                    ind_state = torch.from_numpy(ind_state).float().to(self.device).unsqueeze(0)
                                    ind_action = torch.from_numpy(ind_action).float().to(self.device).unsqueeze(0)
                                    neighbor_states = torch.from_numpy(neighbor_states).float().to(self.device).unsqueeze(0)
                                    actions_ = torch.from_numpy(relevant_actions).float().to(self.device).unsqueeze(0)
                                    neighbor_actions_tensor = torch.from_numpy(neighbor_actions).float().to(self.device).unsqueeze(0)

                                    # Predict next state
                                    if self.network_version == 4:
                                        pred_next_state = self.forward_models[idx].model(relevant_states, ind_action)
                                    elif self.network_version == 5:
                                        pred_next_state = self.forward_models[idx].model(ind_state, actions_)
                                    else:
                                        pred_next_state = self.forward_models[idx].model(relevant_states, actions_).unsqueeze(0)
                            
                                    # Compute inverse model results
                                    if self.network_version == 2:
                                        result = self.inverse_models[idx].model(relevant_states, pred_next_state)
                                    elif self.network_version == 3:
                                        result = self.inverse_models[idx].model(ind_state, neighbor_actions_tensor, pred_next_state)
                                    else:
                                        result = self.inverse_models[idx].model(ind_state, neighbor_states, neighbor_actions_tensor, pred_next_state)
                                    
                                    grounded_action, uncertainty = result[0], result[1]

                                    # Use uncertainty
                                    if self.uncertainty_setting:

                                        agent_uncertainty_sums[idx] += uncertainty.item()

                                        if self.grounding_pattern:

                                            if episode % 2 == 0:
                                                ground_pattern = 0
                                            else:
                                                ground_pattern = 1
                                            
                                            # If none of the above settings, run the traditional UGAT approach
                                            if uncertainty < self.avg_agent_uncertainties[idx]:

                                                if ground_pattern == 0 and (idx == 0 or idx == 2):
                                                    
                                                    if self.network_version == 2:
                                                        actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                                    else:
                                                        actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                    
                                                    grounded_actions[idx] = actions[idx]
                                                    grounded_action_count += 1
            
                                                    ga_by_agent[idx] += 1

                                                elif ground_pattern == 1 and idx == 1:
                            
                                                    if self.network_version == 2:
                                                        actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                                    else:
                                                        actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                        
                                                    grounded_actions[idx] = actions[idx]
                                                    grounded_action_count += 1
        
                                                    ga_by_agent[idx] += 1
                                                    
                                        else:
                                                            
                                            # If none of the above settings, run the traditional UGAT approach
                                            if uncertainty < self.avg_agent_uncertainties[idx]:
                                
                                                if self.network_version == 2:
                                                    actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                                else:
                                                    actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                        
                                                grounded_actions[idx] = actions[idx]
                                                grounded_action_count += 1
        
                                                ga_by_agent[idx] += 1


                                    # Use grounding pattern without uncertainty
                                    elif self.grounding_pattern:

                                        if episode % 2 == 0:
                                            ground_pattern = 0
                                        else:
                                            ground_pattern = 1

                                        if ground_pattern == 0 and (idx == 0 or idx == 2):
                                                    
                                            if self.network_version == 2:
                                                actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                            else:
                                                actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                    
                                            grounded_actions[idx] = actions[idx]
                                            grounded_action_count += 1
            
                                            ga_by_agent[idx] += 1

                                        elif ground_pattern == 1 and idx == 1:
                            
                                            if self.network_version == 2:
                                                actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                            else:
                                                actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                        
                                            grounded_actions[idx] = actions[idx]
                                            grounded_action_count += 1
        
                                            ga_by_agent[idx] += 1

                                    # If probabilistic grounding flag, determine whether to ground based on that flag setting
                                    elif self.prob_grounding != 0:

                                        if random.random() < self.prob_grounding:

                                            if self.network_version == 2:
                                                actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                            else:
                                                actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                    
                                            grounded_actions[idx] = actions[idx]
                                            grounded_action_count += 1
        
                                            ga_by_agent[idx] += 1
                                                
                                    # If no flags always ground every action
                                    else:
                                        if self.network_version == 2 or self.network_version == 3:
                                            actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                        else:
                                            actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                
                                        grounded_actions[idx] = actions[idx]
                                        grounded_action_count += 1

                                        ga_by_agent[idx] += 1
                                        
                            elif self.net == "cityflow4x4":

                                agent_info_map = {
                                    0: [0, 1, 4],  # Agent 0 gets info from itself, agent 1, and agent 4
                                    1: [0, 1, 2, 5],
                                    2: [1, 2, 3, 6],
                                    3: [2, 3, 7],
                                    4: [0, 4, 5, 8],
                                    5: [1, 4, 5, 6, 9],
                                    6: [2, 5, 6, 7, 10],
                                    7: [3, 6, 7, 11],
                                    8: [4, 8, 9, 12],
                                    9: [5, 8, 9, 10, 13],
                                    10: [6, 9, 10, 11, 14],
                                    11: [7, 10, 11, 15],
                                    12: [8, 12, 13],
                                    13: [9, 12, 13, 14],
                                    14: [10, 13, 14, 15],
                                    15: [11, 14, 15]
                                }
                                
                                for idx, ag in enumerate(self.agents_sim):
                                    relevant_indices = agent_info_map.get(idx, [])
                                    
                                    # Collect relevant states
                                    relevant_states = np.concatenate([last_obs[i] for i in relevant_indices])

                                    # Collect neighbor relevant states
                                    relevant_n_states = np.concatenate([last_obs[i] for i in relevant_indices if i != idx])
                                    
                                    # Collect relevant actions
                                    relevant_actions = np.concatenate([
                                        idx2onehot(np.array([actions[i]]), 8) for i in relevant_indices])

                                    # Update neighbor_actions by excluding the current agent's own action
                                    neighbor_actions = np.concatenate([
                                        idx2onehot(np.array([actions[i]]), 8) for i in relevant_indices if i != idx])

                                    ind_state = last_obs[idx]
                            
                                    # Create tensors for use in input
                                    relevant_states = torch.from_numpy(relevant_states).float().to(self.device).unsqueeze(0)
                                    relevant_n_states = torch.from_numpy(relevant_n_states).float().to(self.device).unsqueeze(0)
                                    actions_ = torch.from_numpy(relevant_actions).float().to(self.device).unsqueeze(0)
                                    neighbor_actions_tensor = torch.from_numpy(neighbor_actions).float().to(self.device).unsqueeze(0)
                                    ind_state = torch.from_numpy(ind_state).float().to(self.device).unsqueeze(0)

                                    # Predict next state
                                    pred_next_state = self.forward_models[idx].model(relevant_states, actions_).unsqueeze(0)
                            
                                    # Compute inverse model results
                                    # Compute inverse model results
                                    if self.network_version == 2:
                                        result = self.inverse_models[idx].model(relevant_states, pred_next_state)
                                    else:
                                        result = self.inverse_models[idx].model(ind_state, relevant_n_states, neighbor_actions_tensor, pred_next_state)
                                    
                                    grounded_action, uncertainty = result[0], result[1]

                                    # Use uncertainty
                                    if self.uncertainty_setting:
                                                    
                                        # If none of the above settings, run the traditional UGAT approach
                                        agent_uncertainty_sums[idx] += uncertainty.item()
                                        if uncertainty < self.avg_agent_uncertainties[idx]:
                        
                                            if self.network_version == 2:
                                                actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                            else:
                                                actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                
                                            grounded_actions[idx] = actions[idx]
                                            grounded_action_count += 1

                                            ga_by_agent[idx] += 1

                                    
                                    # Use grounding pattern without uncertainty
                                    elif self.grounding_pattern:

                                        if episode % 2 == 0:
                                            ground_pattern = 0
                                        else:
                                            ground_pattern = 1

                                        if ground_pattern == 0 and idx in {0, 2, 5, 7, 8, 10, 13, 15}:
                                            
                                            actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                    
                                            grounded_actions[idx] = actions[idx]
                                            grounded_action_count += 1
            
                                            ga_by_agent[idx] += 1

                                        elif ground_pattern == 1 and idx in {1, 3, 4, 6, 9, 11, 12, 14}:
                            
                                            actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                        
                                            grounded_actions[idx] = actions[idx]
                                            grounded_action_count += 1
        
                                            ga_by_agent[idx] += 1

                                    # If probabilistic grounding flag, determine whether to ground based on that flag setting
                                    elif self.prob_grounding != 0:

                                        if random.random() < self.prob_grounding:
                                            
                                            actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()
                                                    
                                            grounded_actions[idx] = actions[idx]
                                            grounded_action_count += 1
        
                                            ga_by_agent[idx] += 1
                                                
                                    # If no flags always ground every action
                                    else:
                                        if self.network_version == 2:
                                            actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                        else:
                                            actions[idx] = torch.argmax(grounded_action, dim=1).cpu().item()

                                        grounded_actions[idx] = actions[idx]
                                        grounded_action_count += 1

                                        ga_by_agent[idx] += 1
                            
    
                    actions = actions.flatten()
                    
                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env_sim.step(actions)
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)
                    self.metric_sim.update(rewards)
    
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
                    for idx, ag in enumerate(self.agents_sim):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                                    obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
    
                    self.total_decision_num += 1
                    last_obs = obs
    
                    if self.total_decision_num > self.learning_start and \
                            self.total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                        cur_loss_q = np.stack([ag.train() for ag in self.agents_sim])
                        episode_loss.append(cur_loss_q)
                    if self.total_decision_num > self.learning_start and \
                            self.total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                        [ag.update_target_network() for ag in self.agents_sim]

                    # Ensure actions are in the correct format (list of numbers)
                    # original_actions = original_actions.flatten().tolist()
                    # grounded_actions_fixed = [item.item() if isinstance(item, np.ndarray) else item for item in grounded_actions]
                    # actions = actions.tolist()
                    
                    # self.writeActionLog(episode, i, 3600, original_actions, grounded_actions_fixed, actions)
    
                    if all(dones):
                        break
    
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            if self.gattype == "decentralized" or self.gattype == "jlgat" or self.gattype == "central_fwd_dec_inv":
                for idx, ag in enumerate(self.agents_sim):

                    # Update last_two_uncertainties
                    self.last_two_uncertainties[idx].append(agent_uncertainty_sums[idx] / 360)
                    if len(self.last_two_uncertainties[idx]) > self.last_n_uncertainties:
                        self.last_two_uncertainties[idx].pop(0)

                    # Update mean uncertainity for next episode
                    self.avg_agent_uncertainties[idx] = np.mean(self.last_two_uncertainties[idx])

                self.logger.info(
                "Policy training episode: {}, grounded actions taken: {}, last two uncertainties: {}, avg agent uncertainties: {}, grounded actions by agent: {}".format(episode, grounded_action_count, self.last_two_uncertainties, self.avg_agent_uncertainties, ga_by_agent))

            elif self.gattype == "centralized" or self.gattype == "central_inv_dec_fwd":
                
                # Update last_two_uncertainties
                self.last_two_central_uncertainties.append(uncertainty_sum / 360)
                if len(self.last_two_central_uncertainties) > self.last_n_uncertainties:
                    self.last_two_central_uncertainties.pop(0)

                # Update mean uncertainity for next episode
                self.mean_uncertainty = np.mean(self.last_two_central_uncertainties)

                self.logger.info(
                "Policy training episode: {}, grounded actions taken: {}, last uncertainties: {}, avg uncertainty: {}, grounded actions by agent: {}".format(episode, grounded_action_count, self.last_two_central_uncertainties, self.mean_uncertainty, ga_by_agent))


            self.writeLog("TRAIN", e, self.metric_sim.real_average_travel_time(), \
                          mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                          self.metric_sim.throughput())
            self.logger.info(
                "Policy training episode: {}, iteration {}/{}, policy training avg travel time:{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(episode, e, self.training_iterations, self.metric_sim.real_average_travel_time(), \
                                                                                                            mean_loss,
                                                                                                            self.metric_sim.rewards(),
                                                                                                            self.metric_sim.queue(),
                                                                                                            self.metric_sim.delay(),
                                                                                                            int(self.metric_sim.throughput())))

            # if e % self.save_rate == 0:
            #     [ag.save_model(e=e) for ag in self.agents_sim]




    def gat_training(self, e):
        
        # If GAT training desired, handle after both sim and real datasets updated above
            if self.gat == True:
                if self.gattype == "centralized":
                    # Load and split the real and sim data to prepare for forward / inverse model training

                    # Forward data split using real data
                    load_and_split_forward_data("collected/ereal_train.pkl", "collected/ereal_train_full.pkl", "collected/ereal_test_full.pkl",
                                       8, 0.2, 42, "centralized", len(self.agents_real))

                    # Inverse data split using sim data
                    load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full.pkl", "collected/esim_test_full.pkl",
                                       8, 0.2, 42, "centralized", len(self.agents_sim))

                    # Train the centralized forward model
                    self.forward_model.train(100, 'forward', len(self.agents_real), 5000 * len(self.agents_real))

                    # Train the centralized inverse model
                    self.inverse_model.train(100, 'inverse', len(self.agents_sim), 5000 * len(self.agents_real))
                    
                elif self.gattype == "decentralized":
                    # Load and split the real and sim data to prepare for forward / inverse model training

                    # Forward data split using real data
                    load_and_split_forward_data("collected/ereal_train.pkl", "collected/ereal_train_full", "collected/ereal_test_full",
                                       8, 0.2, 42, "decentralized", len(self.agents_real))

                    # Inverse data split using sim data
                    load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full", "collected/esim_test_full",
                                       8, 0.2, 42, "decentralized", len(self.agents_sim))

                    for idx, ag in enumerate(self.agents_sim):
                        
                        # Train the decentralized forward model
                        self.forward_models[idx].train(100, 'forward', idx, 5000, "decentralized")
    
                        # Train the decentralized inverse model
                        self.inverse_models[idx].train(100, 'inverse', idx, 5000, "decentralized")
                        

                elif self.gattype == "jlgat":
                    # Load and split the real and sim data to prepare for forward / inverse model training

                    # Forward data split using real data
                    load_and_split_forward_data("collected/ereal_train.pkl", "collected/ereal_train_full", "collected/ereal_test_full",
                                    8, 0.2, 42, "jlgat", len(self.agents_real))

                    # Inverse data split using sim data
                    if self.network_version == 2:
                        load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full", "collected/esim_test_full",
                                           8, 0.2, 42, "decentralized", len(self.agents_sim))
                    elif self.network_version == 3:
                        load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full", "collected/esim_test_full",
                                           8, 0.2, 42, "jlgat3", len(self.agents_sim))
                    else:
                        load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full", "collected/esim_test_full",
                                           8, 0.2, 42, "jlgat", len(self.agents_sim))

                    for idx, ag in enumerate(self.agents_sim):
                        
                        # Train the JLGAT forward model
                        self.forward_models[idx].train(100, 'forward', idx, 5000, "jlgat", self.network_version)
    
                        # Train the JLGAT inverse model
                        self.inverse_models[idx].train(100, 'inverse', idx, 5000, "jlgat", self.network_version)
                    
    
    def sim_rollout(self, e, mode="centralized"):
        '''
        single_rollout
        Perform a single rollout in the simulated environment and save data.
    
        :param: None
        :return: None
        '''
    
        path = 'collected'
        output_file = 'esim_train.pkl'
        file_path = os.path.join(path, output_file)

        # Initialize metrics and reset the environment
        self.metric_sim.clear()
        last_obs = self.env_sim.reset()
        state_action_next_state = []
    
        # Reset agents
        for a in self.agents_sim:
            a.reset()
    
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env_sim.eng.set_save_replay(True)
                self.env_sim.eng.set_replay_file(os.path.join(self.replay_file_dir, "single_rollout_replay.txt"))
            else:
                self.env_sim.eng.set_save_replay(False)
    
        i = 0
        while i < self.steps:
            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in self.agents_sim])
    
                # Get agent actions
                actions = []
                for idx, ag in enumerate(self.agents_sim):
                    actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=True))
                actions = np.stack(actions)
    
                # Perform actions for the specified interval and collect data
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_sim.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
    
                rewards = np.mean(rewards_list, axis=0)
                self.metric_sim.update(rewards)

                if mode == "decentralized" or mode == "central_fwd_dec_inv":
                    # Store the transition (agent_index, state, action, next_state) for each agent
                    for idx, (state, action, next_state) in enumerate(zip(last_obs, actions, obs)):
                        state_action_next_state.append((idx, state, action, next_state))

                # Data format is (Individual state, Joint-local state for neighbors, actions taken by neighbors, next individual state, individual action to cause transition)
                elif mode == "jlgat" and self.net == "cityflow1x3":

                    if self.network_version == 2:
                        # Joint local information storage for cityflow1x3 network
                        state_action_next_state.append((0, np.concatenate([last_obs[0], last_obs[1]], axis=0), actions[0].reshape(-1, 1), obs[0], actions[0].reshape(-1, 1)))
                        state_action_next_state.append((1, np.concatenate([last_obs[0], last_obs[1], last_obs[2]], axis=0), actions[1].reshape(-1, 1), obs[1], actions[1].reshape(-1, 1)))
                        state_action_next_state.append((2, np.concatenate([last_obs[1], last_obs[2]], axis=0), actions[2].reshape(-1, 1), obs[2], actions[2].reshape(-1, 1)))
                    elif self.network_version == 3:
                        # Joint local information storage for cityflow1x3 network
                        state_action_next_state.append((0, last_obs[0], actions[1].reshape(-1, 1), obs[0], actions[0].reshape(-1, 1)))
                        state_action_next_state.append((1, last_obs[2], np.concatenate([actions[0], actions[2]], axis=0).reshape(-1, 1), obs[1], actions[1].reshape(-1, 1)))
                        state_action_next_state.append((2, last_obs[2], actions[2].reshape(-1, 1).reshape(-1, 1), obs[2], actions[2].reshape(-1, 1)))
                    else:
                        # Joint local information storage for cityflow1x3 network
                        state_action_next_state.append((0, last_obs[0], last_obs[1], actions[1].reshape(-1, 1), obs[0],  actions[0].reshape(-1, 1)))
                        state_action_next_state.append((1, last_obs[1], np.concatenate([last_obs[0], last_obs[2]], axis=0), np.concatenate([actions[0], actions[2]], axis=0).reshape(-1, 1), obs[1], actions[1].reshape(-1, 1)))
                        state_action_next_state.append((2, last_obs[2], last_obs[1], actions[1].reshape(-1, 1), obs[2],  actions[2].reshape(-1, 1)))

                # Data format is (Joint-local state, actions taken by neighbors, next individual state, individual action to cause transition)
                elif mode == "jlgat" and self.net == "cityflow4x4":
                    agent_info_map = {
                                    0: [0, 1, 4],  # Agent 0 gets info from itself, agent 1, and agent 4
                                    1: [0, 1, 2, 5],
                                    2: [1, 2, 3, 6],
                                    3: [2, 3, 7],
                                    4: [0, 4, 5, 8],
                                    5: [1, 4, 5, 6, 9],
                                    6: [2, 5, 6, 7, 10],
                                    7: [3, 6, 7, 11],
                                    8: [4, 8, 9, 12],
                                    9: [5, 8, 9, 10, 13],
                                    10: [6, 9, 10, 11, 14],
                                    11: [7, 10, 11, 15],
                                    12: [8, 12, 13],
                                    13: [9, 12, 13, 14],
                                    14: [10, 13, 14, 15],
                                    15: [11, 14, 15]
                                }
                    
                    for agent, neighbors in agent_info_map.items():
                        
                        # Exclude the agent's own actions from the list of neighbor actions
                        neighbor_idx = [i for i in neighbors if i != agent]

                        total_idx = [i for i in neighbors]
                        
                        # Collect the joint-local state for the agent and its neighbors
                        joint_local_state = np.concatenate([last_obs[i] for i in neighbor_idx], axis=0)

                        full_local_state = np.concatenate([last_obs[i] for i in total_idx], axis=0)
                        
                        # Collect actions taken by the neighbors (excluding the agent itself)
                        actions_taken_by_neighbors = np.concatenate([actions[i] for i in neighbor_idx], axis=0).reshape(-1, 1)

                        # Individual state
                        individual_state = last_obs[agent]
                        
                        # Collect the next state for the individual agent
                        next_state = obs[agent]

                        # Collect the action for the individual agent
                        individual_action = actions[agent]
                        
                        # Append the tuple to the list
                        if self.network_version == 2:
                            state_action_next_state.append((agent, full_local_state, individual_action, next_state))
                        else:
                            state_action_next_state.append((agent, individual_state, joint_local_state, actions_taken_by_neighbors, next_state, individual_action))
                else:
                    state_action_next_state.append((last_obs, actions, obs))
                    
                last_obs = obs
    
            if all(dones):
                break

        if e % self.save_rate == 0:
            [ag.save_model(e=e) for ag in self.agents_sim]
    
        # Save the collected rollout data
        os.makedirs(path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(state_action_next_state, f)

            self.logger.info("Sim Rollout episode:{}/{}, sim avg travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(e, self.episodes, self.metric_sim.real_average_travel_time(), self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(), int(self.metric_sim.throughput())))
        #for j in range(len(self.world_sim.intersections)):
            # self.logger.debug("intersection:{}, individual reward:{}, individual queue:{}, individual delay:{}, individual throughput:{}".format(j, self.metric_sim.lane_rewards()[j], self.metric_sim.queue()[j], self.metric_sim.delay()[j], int(self.metric_sim.throughput()[j])))


    def train_test(self, e, mode="centralized"):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        last_obs = self.env_real.reset()
        self.metric_real.clear()
        state_action_next_state = []
        
        for a in self.agents_real:
            a.load_model(e)
            a.reset()
            
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = []
                for idx, ag in enumerate(self.agents_real):
                    actions.append(ag.get_action(last_obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric_real.update(rewards)

                if mode == "decentralized" or mode == "central_inv_dec_fwd":
                    # Store the transition (agent_index, state, action, next_state) for each agent
                    for idx, (state, action, next_state) in enumerate(zip(last_obs, actions, obs)):
                        state_action_next_state.append((idx, state, action, next_state))

                # Joint Local state, Joint Local action, Individual next state
                elif mode == "jlgat" and self.net == "cityflow1x3":

                    # Ablation without neighbor action info in forward
                    if self.network_version == 4:
                        # Joint local information storage for cityflow1x3 network
                        state_action_next_state.append((0, np.concatenate([last_obs[0], last_obs[1]], axis=0), actions[0].reshape(-1, 1), obs[0]))
                        state_action_next_state.append((1, np.array(last_obs).squeeze(axis=1), actions[1].reshape(-1, 1), obs[1]))
                        state_action_next_state.append((2, np.concatenate([last_obs[1], last_obs[2]], axis=0), actions[2].reshape(-1, 1), obs[2]))
                    # Ablation without neighbor state info in forward
                    elif self.network_version == 5:
                        # Joint local information storage for cityflow1x3 network
                        state_action_next_state.append((0, last_obs[0], np.concatenate([actions[0], actions[1]], axis=0).reshape(-1, 1), obs[0]))
                        state_action_next_state.append((1, last_obs[1], np.concatenate([actions[0], actions[1], actions[2]], axis=0).reshape(-1, 1), obs[1]))
                        state_action_next_state.append((2, last_obs[2], np.concatenate([actions[1], actions[2]], axis=0).reshape(-1, 1), obs[2]))
                    else:
                        # Joint local information storage for cityflow1x3 network
                        state_action_next_state.append((0, np.concatenate([last_obs[0], last_obs[1]], axis=0), np.concatenate([actions[0], actions[1]], axis=0).reshape(-1, 1), obs[0]))
                        state_action_next_state.append((1, np.array(last_obs).squeeze(axis=1), actions.reshape(-1, 1), obs[1]))
                        state_action_next_state.append((2, np.concatenate([last_obs[1], last_obs[2]], axis=0), np.concatenate([actions[1], actions[2]], axis=0).reshape(-1, 1), obs[2]))

                elif mode == "jlgat" and self.net == "cityflow4x4":
                    # Joint local information storage for cityflow4x4 network
                    state_action_next_state.append((0, np.concatenate([last_obs[i] for i in [0, 1, 4]], axis=0), np.concatenate([actions[i] for i in [0, 1, 4]], axis=0).reshape(-1, 1), obs[0]))
                    state_action_next_state.append((1, np.concatenate([last_obs[i] for i in [0, 1, 2, 5]], axis=0), np.concatenate([actions[i] for i in [0, 1, 2, 5]], axis=0).reshape(-1, 1), obs[1]))
                    state_action_next_state.append((2, np.concatenate([last_obs[i] for i in [1, 2, 3, 6]], axis=0), np.concatenate([actions[i] for i in [1, 2, 3, 6]], axis=0).reshape(-1, 1), obs[2]))
                    state_action_next_state.append((3, np.concatenate([last_obs[i] for i in [2, 3, 7]], axis=0), np.concatenate([actions[i] for i in [2, 3, 7]], axis=0).reshape(-1, 1), obs[3]))
                    state_action_next_state.append((4, np.concatenate([last_obs[i] for i in [0, 4, 5, 8]], axis=0), np.concatenate([actions[i] for i in [0, 4, 5, 8]], axis=0).reshape(-1, 1), obs[4]))
                    state_action_next_state.append((5, np.concatenate([last_obs[i] for i in [1, 4, 5, 6, 9]], axis=0), np.concatenate([actions[i] for i in [1, 4, 5, 6, 9]], axis=0).reshape(-1, 1), obs[5]))
                    state_action_next_state.append((6, np.concatenate([last_obs[i] for i in [2, 5, 6, 7, 10]], axis=0), np.concatenate([actions[i] for i in [2, 5, 6, 7, 10]], axis=0).reshape(-1, 1), obs[6]))
                    state_action_next_state.append((7, np.concatenate([last_obs[i] for i in [3, 6, 7, 11]], axis=0), np.concatenate([actions[i] for i in [3, 6, 7, 11]], axis=0).reshape(-1, 1), obs[7]))
                    state_action_next_state.append((8, np.concatenate([last_obs[i] for i in [4, 8, 9, 12]], axis=0), np.concatenate([actions[i] for i in [4, 8, 9, 12]], axis=0).reshape(-1, 1), obs[8]))
                    state_action_next_state.append((9, np.concatenate([last_obs[i] for i in [5, 8, 9, 10, 13]], axis=0), np.concatenate([actions[i] for i in [5, 8, 9, 10, 13]], axis=0).reshape(-1, 1), obs[9]))
                    state_action_next_state.append((10, np.concatenate([last_obs[i] for i in [6, 9, 10, 11, 14]], axis=0), np.concatenate([actions[i] for i in [6, 9, 10, 11, 14]], axis=0).reshape(-1, 1), obs[10]))
                    state_action_next_state.append((11, np.concatenate([last_obs[i] for i in [7, 10, 11, 15]], axis=0), np.concatenate([actions[i] for i in [7, 10, 11, 15]], axis=0).reshape(-1, 1), obs[11]))
                    state_action_next_state.append((12, np.concatenate([last_obs[i] for i in [8, 12, 13]], axis=0), np.concatenate([actions[i] for i in [8, 12, 13]], axis=0).reshape(-1, 1), obs[12]))
                    state_action_next_state.append((13, np.concatenate([last_obs[i] for i in [9, 12, 13, 14]], axis=0), np.concatenate([actions[i] for i in [9, 12, 13, 14]], axis=0).reshape(-1, 1), obs[13]))
                    state_action_next_state.append((14, np.concatenate([last_obs[i] for i in [10, 13, 14, 15]], axis=0), np.concatenate([actions[i] for i in [10, 13, 14, 15]], axis=0).reshape(-1, 1), obs[14]))
                    state_action_next_state.append((15, np.concatenate([last_obs[i] for i in [11, 14, 15]], axis=0), np.concatenate([actions[i] for i in [11, 14, 15]], axis=0).reshape(-1, 1), obs[15]))
                
                else:
                    state_action_next_state.append((last_obs, actions, obs))

                last_obs = obs
            
            if all(dones):
                break
                
        self.logger.info("Real rollout step:{}/{}, travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format( \
            e, self.episodes, self.metric_real.real_average_travel_time(), self.metric_real.rewards(), \
            self.metric_real.queue(), self.metric_real.delay(), int(self.metric_real.throughput())))
        self.writeLog("Real rollout", e, self.metric_real.real_average_travel_time(), \
                      100, self.metric_real.rewards(), self.metric_real.queue(), self.metric_real.delay(), self.metric_real.throughput())

        # Save the data for the current episode
        file_path = 'collected/ereal_train.pkl'
            
        # Save new data directly to the file
        with open(file_path, 'wb') as f:
            pkl.dump(state_action_next_state, f)
        
        return self.metric_real.real_average_travel_time()


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

    
    def writeActionLog(self, episode_num, step, total_steps, orig_actions, grounded_actions, actions_taken):
        '''
        writeActionLog
        Write log for record and debug, including episode information and actions taken by both the original and grounded agents in the specified format.
    
        :param episode_num: current episode number
        :param total_episodes: total number of episodes
        :param agent_actions: actions taken by the original agent
        :param grounded_actions: actions taken by the grounded agent
        :return: None
        '''
        res = f"Policy training episode:{episode_num}, step:{step}/{total_steps}, original actions:{orig_actions}, grounded actions:{grounded_actions}, actions taken: {actions_taken}"
        
        log_handle = open(self.action_log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

