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
        self.oaat = Registry.mapping['trainer_mapping']['setting'].param['oaat']
        self.local_grounding_only = Registry.mapping['trainer_mapping']['setting'].param['local_grounding_only']
        self.ground_original = Registry.mapping['trainer_mapping']['setting'].param['ground_original']
        self.oaat_num = 0
        
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
                                         '_BRF.log') + '_DTL.log'
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
                                                (self.agents_real[0].ob_generator.ob_length * num_agents + self.agents_real[0].action_space.n * num_agents),
                                                self.agents_real[0].ob_generator.ob_length * num_agents, self.device, gat_path, 'collected/ereal_train_full.pkl')
                self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * num_agents * 2,
                                                self.agents_real[0].action_space.n * num_agents, self.device, gat_path,
                                                'collected/esim_train_full.pkl', backward=True)
            # Initialize decentralized GAT models
            elif self.gattype == "decentralized":
                print(f"\n------- INITIALIZING GAT MODELS DECENTRALIZED -------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')

                
                for i in range(num_agents):
                    self.forward_model = NN_predictor(self.logger,
                                                    (self.agents_real[0].ob_generator.ob_length + self.agents_real[0].action_space.n),
                                                    self.agents_real[0].ob_generator.ob_length, self.device, gat_path, 'collected/ereal_train_full.pkl')
                    self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * 2,
                                                    self.agents_real[0].action_space.n, self.device, gat_path,
                                                    'collected/esim_train_full.pkl', backward=True)
                    
                    self.forward_models.append(self.forward_model)
                    self.inverse_models.append(self.inverse_model)

            elif self.gattype == "central_fwd_dec_inv":
                self.last_two_central_uncertainties = []
                print(f"\n------- INITIALIZING GAT MODELS CENTRALIZED FWD / DECENTRALIZED INV-------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
                self.forward_model = NN_predictor(self.logger,
                                                (self.agents_real[0].ob_generator.ob_length * num_agents + self.agents_real[0].action_space.n * num_agents),
                                                self.agents_real[0].ob_generator.ob_length * num_agents, self.device, gat_path, 'collected/ereal_train_full.pkl')
                for i in range(num_agents):
                    self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * 2,
                                                    self.agents_real[0].action_space.n, self.device, gat_path,
                                                    'collected/esim_train_full.pkl', backward=True)
                    self.inverse_models.append(self.inverse_model)

            elif self.gattype == "central_inv_dec_fwd":
                self.last_two_central_uncertainties = []
                print(f"\n------- INITIALIZING GAT MODELS CENTRALIZED INV / DECENTRALIZED FWD -------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')

                for i in range(num_agents):
                    self.forward_model = NN_predictor(self.logger,
                                                    (self.agents_real[0].ob_generator.ob_length + self.agents_real[0].action_space.n),
                                                    self.agents_real[0].ob_generator.ob_length, self.device, gat_path, 'collected/ereal_train_full.pkl')
                    
                    self.forward_models.append(self.forward_model)

                self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * num_agents * 2,
                                                self.agents_real[0].action_space.n * num_agents, self.device, gat_path,
                                                'collected/esim_train_full.pkl', backward=True)

            # Initialize JL-GAT models
            elif self.gattype == "jlgat":
                print(f"\n------- INITIALIZING JL-GAT MODELS -------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
                
                self.net = Registry.mapping['trainer_mapping']['setting'].param['network']

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

                    self.forward_model = NN_predictor(self.logger,
                                                    (self.agents_real[0].ob_generator.ob_length * ag.neighbors + self.agents_real[0].action_space.n * ag.neighbors),
                                                    self.agents_real[0].ob_generator.ob_length * ag.neighbors, self.device, gat_path, 'collected/ereal_train_full.pkl')
                    self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * ag.neighbors * 2,
                                                    self.agents_real[0].action_space.n * ag.neighbors, self.device, gat_path,
                                                    'collected/esim_train_full.pkl', backward=True)
                    
                    self.forward_models.append(self.forward_model)
                    self.inverse_models.append(self.inverse_model)

                    # print(f"agent: {idx}, intersection: {ag.inter_obj.id}, roads: {ag.inter_obj.roads}")
                    

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

                    original_actions = actions
    
                    if self.gat:
                        if self.gattype == "centralized":
                            combined_state = np.concatenate([state.flatten() for state in last_obs[:len(self.agents_real)]])
                            one_hot_actions = np.concatenate([
                                idx2onehot(np.array([action]), 8).flatten() for action in actions[:len(self.agents_real)]
                            ])
                            joint_state_action = np.concatenate([combined_state, one_hot_actions], axis=0)
                            joint_state_action = torch.from_numpy(joint_state_action).float().to(self.device).unsqueeze(0)
    
                            pred_next_state = self.forward_model.model(joint_state_action)
                            current_state_tensor = torch.from_numpy(combined_state).float().to(self.device)
                            inverse_input = torch.cat([current_state_tensor.unsqueeze(0), pred_next_state], dim=1).to(self.device)
    
                            result = self.inverse_model.model(inverse_input)
                            grounded_action, uncertainty = result[0], result[1]

                            if self.uncertainty_setting == True:
                                uncertainty_sum += uncertainty.item()
                                if uncertainty < self.mean_uncertainty:
                                    grounded_action_reshaped = grounded_action.view(len(self.agents_sim), 8)
                                    actions = torch.argmax(grounded_action_reshaped, dim=1).cpu().numpy()
                                    grounded_action_count += len(self.agents_sim)
                            else:
                                grounded_action_reshaped = grounded_action.view(len(self.agents_sim), 8)
                                actions = torch.argmax(grounded_action_reshaped, dim=1).cpu().numpy()
                                grounded_action_count += len(self.agents_sim)
    
                        elif self.gattype == "decentralized":
                            for idx, ag in enumerate(self.agents_sim):
                                individual_state = last_obs[idx].flatten()
                                individual_action = idx2onehot(np.array([actions[idx]]), 8).flatten()
                                state_action = np.concatenate([individual_state, individual_action], axis=0)
                                state_action = torch.from_numpy(state_action).float().to(self.device).unsqueeze(0)
    
                                pred_next_state = self.forward_models[idx].model(state_action)
                                current_state_tensor = torch.from_numpy(individual_state).float().to(self.device)
                                inverse_input = torch.cat([current_state_tensor.unsqueeze(0), pred_next_state], dim=1).to(self.device)
    
                                result = self.inverse_models[idx].model(inverse_input)
                                grounded_action, uncertainty = result[0], result[1]

                                if self.uncertainty_setting == True:
                                    agent_uncertainty_sums[idx] += uncertainty.item()
                                    if uncertainty < self.avg_agent_uncertainties[idx]:
                                        actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                        grounded_action_count += 1
                                else:
                                    actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                    grounded_action_count += 1

                        elif self.gattype == "central_fwd_dec_inv":
                            # Centralized Forward Model
                            combined_state = np.concatenate([state.flatten() for state in last_obs[:len(self.agents_real)]])
                            one_hot_actions = np.concatenate([
                                idx2onehot(np.array([action]), 8).flatten() for action in actions[:len(self.agents_real)]
                            ])
                            joint_state_action = np.concatenate([combined_state, one_hot_actions], axis=0)
                            joint_state_action = torch.from_numpy(joint_state_action).float().to(self.device).unsqueeze(0)
                        
                            # Predict the next state using the forward model
                            pred_next_state = self.forward_model.model(joint_state_action)
                            
                            # Split the predicted next state into individual agent states
                            pred_next_state_split = pred_next_state.view(len(self.agents_sim), -1)  # Shape: (num_agents, state_length_per_agent)
                            
                            for idx, ag in enumerate(self.agents_sim):
                                individual_pred_next_state = pred_next_state_split[idx].unsqueeze(0)  # Get predicted next state for current agent
                                
                                individual_state = last_obs[idx].flatten()
                        
                                # Prepare input for the inverse model (state + predicted next state)
                                current_state_tensor = torch.from_numpy(individual_state).float().to(self.device)
                                inverse_input = torch.cat([current_state_tensor.unsqueeze(0), individual_pred_next_state], dim=1).to(self.device)
                        
                                # Use inverse model to compute grounded action and uncertainty
                                result = self.inverse_models[idx].model(inverse_input)
                                grounded_action, uncertainty = result[0], result[1]
                        
                                if self.uncertainty_setting == True:
                                    agent_uncertainty_sums[idx] += uncertainty.item()
                                    if uncertainty < self.avg_agent_uncertainties[idx]:
                                        actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                        grounded_action_count += 1
                                else:
                                    actions[idx] = torch.argmax(grounded_action.view(1, 8), dim=1).cpu().item()
                                    grounded_action_count += 1

                        elif self.gattype == "central_inv_dec_fwd":
                            # Step 1: Decentralized Forward Model - Predict each agent's next state independently
                            pred_next_states = []
                            
                            for idx, ag in enumerate(self.agents_sim):
                                individual_state = last_obs[idx].flatten()
                                individual_action = idx2onehot(np.array([actions[idx]]), 8).flatten()
                                
                                state_action = np.concatenate([individual_state, individual_action], axis=0)
                                state_action = torch.from_numpy(state_action).float().to(self.device).unsqueeze(0)
                        
                                # Predict next state using individual agent's forward model
                                pred_next_state = self.forward_models[idx].model(state_action)
                                pred_next_states.append(pred_next_state)
                        
                            # Step 2: Centralized Inverse Model - Combine all predictions and apply inverse model
                            joint_pred_next_state = torch.cat(pred_next_states, dim=1)  # Concatenate predictions across agents
                            combined_state = np.concatenate([state.flatten() for state in last_obs[:len(self.agents_real)]])
                            
                            current_state_tensor = torch.from_numpy(combined_state).float().to(self.device).unsqueeze(0)
                            inverse_input = torch.cat([current_state_tensor, joint_pred_next_state], dim=1).to(self.device)

                            # Get grounded actions and uncertainties
                            result = self.inverse_model.model(inverse_input)
                            grounded_action, uncertainty = result[0], result[1]
                        
                            # Step 3: Update actions based on uncertainty settings
                            if self.uncertainty_setting:
                                uncertainty_sum += uncertainty.item()
                                if uncertainty < self.mean_uncertainty:
                                    grounded_action_reshaped = grounded_action.view(len(self.agents_sim), 8)
                                    actions = torch.argmax(grounded_action_reshaped, dim=1).cpu().numpy()
                                    grounded_action_count += len(self.agents_sim)
                            else:
                                grounded_action_reshaped = grounded_action.view(len(self.agents_sim), 8)
                                actions = torch.argmax(grounded_action_reshaped, dim=1).cpu().numpy()
                                grounded_action_count += len(self.agents_sim)

                        # Currently setup for 1x3 only
                        elif self.gattype == "jlgat":
                            
                            if self.net == "cityflow1x3":
                                for idx, ag in enumerate(self.agents_sim):

                                    # Ground based upon original intended actions
                                    if self.ground_original:
                                    
                                        if idx == 0:  # Agent 0: Uses its own state + agent 1's state + its own & agent 1's actions
                                            relevant_states = np.concatenate([last_obs[0].flatten(), last_obs[1].flatten()])
                                            relevant_actions = np.concatenate([
                                                idx2onehot(np.array([original_actions[0]]), 8).flatten(),
                                                idx2onehot(np.array([original_actions[1]]), 8).flatten()
                                            ])
                                        
                                        elif idx == 1:  # Agent 1: Uses all agent states and actions
                                            relevant_states = np.concatenate([last_obs[0].flatten(), last_obs[1].flatten(), last_obs[2].flatten()])
                                            relevant_actions = np.concatenate([
                                                idx2onehot(np.array([original_actions[0]]), 8).flatten(),
                                                idx2onehot(np.array([original_actions[1]]), 8).flatten(),
                                                idx2onehot(np.array([original_actions[2]]), 8).flatten()
                                            ])
                                            
                                        elif idx == 2:  # Agent 2: Uses its own state + agent 1's state + its own & agent 1's actions
                                            relevant_states = np.concatenate([last_obs[2].flatten(), last_obs[1].flatten()])
                                            relevant_actions = np.concatenate([
                                                idx2onehot(np.array([original_actions[2]]), 8).flatten(),
                                                idx2onehot(np.array([original_actions[1]]), 8).flatten()
                                            ])
                                            
                                    # Ground based upon grounded actions (cascade)
                                    else:

                                        if idx == 0:  # Agent 0: Uses its own state + agent 1's state + its own & agent 1's actions
                                            relevant_states = np.concatenate([last_obs[0].flatten(), last_obs[1].flatten()])
                                            relevant_actions = np.concatenate([
                                                idx2onehot(np.array([actions[0]]), 8).flatten(),
                                                idx2onehot(np.array([actions[1]]), 8).flatten()
                                            ])
                                        
                                        elif idx == 1:  # Agent 1: Uses all agent states and actions
                                            relevant_states = np.concatenate([last_obs[0].flatten(), last_obs[1].flatten(), last_obs[2].flatten()])
                                            relevant_actions = np.concatenate([
                                                idx2onehot(np.array([actions[0]]), 8).flatten(),
                                                idx2onehot(np.array([actions[1]]), 8).flatten(),
                                                idx2onehot(np.array([actions[2]]), 8).flatten()
                                            ])
                                            
                                        elif idx == 2:  # Agent 2: Uses its own state + agent 1's state + its own & agent 1's actions
                                            relevant_states = np.concatenate([last_obs[2].flatten(), last_obs[1].flatten()])
                                            relevant_actions = np.concatenate([
                                                idx2onehot(np.array([actions[2]]), 8).flatten(),
                                                idx2onehot(np.array([actions[1]]), 8).flatten()
                                            ])

                                    
                                    # Create state-action input
                                    state_action = np.concatenate([relevant_states, relevant_actions], axis=0)
                                    state_action = torch.from_numpy(state_action).float().to(self.device).unsqueeze(0)
                            
                                    # Predict next state
                                    pred_next_state = self.forward_models[idx].model(state_action)
                                    current_state_tensor = torch.from_numpy(relevant_states).float().to(self.device)
                                    inverse_input = torch.cat([current_state_tensor.unsqueeze(0), pred_next_state], dim=1).to(self.device)
                            
                                    # Compute inverse model results
                                    result = self.inverse_models[idx].model(inverse_input)
                                    grounded_action, uncertainty = result[0], result[1]

                                    # Use uncertainty
                                    if self.uncertainty_setting:

                                        # If one agent at a time and current agent is the select agent, they may ground their action...
                                        if self.oaat and self.oaat_num == idx:
                                            agent_uncertainty_sums[idx] += uncertainty.item()
                                            if uncertainty < self.avg_agent_uncertainties[idx]:
                                                
                                                batch_size, num_elements = grounded_action.shape
                                                new_first_dim = num_elements // 8
    
                                                # Reshape to (N, 8)
                                                reshaped_tensor = grounded_action.view(new_first_dim, 8)
    
                                                select_idx = idx
    
                                                # If last agent, select the last spot for state and action
                                                if idx == 2:
                                                    select_idx = 1
                                                    
                                                selected_tensor = reshaped_tensor[select_idx]
                        
                                                actions[idx] = torch.argmax(selected_tensor, dim=0).cpu().item()
                                                grounded_action_count += 1

                                                ga_by_agent[idx] += 1

                                        # If one agent at a time and not current agent, no GAT allowed...
                                        elif self.oaat:
                                            agent_uncertainty_sums[idx] += uncertainty.item()

                                        # Only ground based on local observations and alternate which agents can ground
                                        elif self.local_grounding_only:

                                            # Ground agent 1 and 3 every other episode
                                            if episode % 2 == 0:

                                                agent_uncertainty_sums[idx] += uncertainty.item()
                                                if uncertainty < self.avg_agent_uncertainties[idx] and idx in [0, 2]:
                                                    
                                                    batch_size, num_elements = grounded_action.shape
                                                    new_first_dim = num_elements // 8
        
                                                    # Reshape to (N, 8)
                                                    reshaped_tensor = grounded_action.view(new_first_dim, 8)
        
                                                    select_idx = idx
        
                                                    # If last agent, select the last spot for state and action
                                                    if idx == 2:
                                                        select_idx = 1
                                                        
                                                    selected_tensor = reshaped_tensor[select_idx]
                            
                                                    actions[idx] = torch.argmax(selected_tensor, dim=0).cpu().item()
                                                    grounded_action_count += 1
    
                                                    ga_by_agent[idx] += 1
                                                    
                                            # Handle local grounding for agent 1  
                                            else:

                                                agent_uncertainty_sums[idx] += uncertainty.item()
                                                if uncertainty < self.avg_agent_uncertainties[idx] and idx == 1:
                                                    
                                                    batch_size, num_elements = grounded_action.shape
                                                    new_first_dim = num_elements // 8
        
                                                    # Reshape to (N, 8)
                                                    reshaped_tensor = grounded_action.view(new_first_dim, 8)
        
                                                    select_idx = idx
        
                                                    # If last agent, select the last spot for state and action
                                                    if idx == 2:
                                                        select_idx = 1
                                                        
                                                    selected_tensor = reshaped_tensor[select_idx]
                            
                                                    actions[idx] = torch.argmax(selected_tensor, dim=0).cpu().item()
                                                    grounded_action_count += 1
    
                                                    ga_by_agent[idx] += 1
                                                
                                    # If no flags always ground every action
                                    else:
                                        
                                        batch_size, num_elements = grounded_action.shape
                                        new_first_dim = num_elements // 8

                                        # Reshape to (N, 8)
                                        reshaped_tensor = grounded_action.view(new_first_dim, 8)

                                        select_idx = idx

                                        # If last agent, select the last spot for action
                                        if idx == 2:
                                            select_idx = 1
                                                
                                        selected_tensor = reshaped_tensor[select_idx]
                    
                                        actions[idx] = torch.argmax(selected_tensor, dim=0).cpu().item()
                                        grounded_action_count += 1
                                        
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
                                    relevant_states = np.concatenate([last_obs[i].flatten() for i in relevant_indices])
                                    
                                    # Collect relevant actions
                                    relevant_actions = np.concatenate([
                                        idx2onehot(np.array([actions[i]]), 8).flatten() for i in relevant_indices])
                            
                                    # Create state-action input
                                    state_action = np.concatenate([relevant_states, relevant_actions], axis=0)
                                    state_action = torch.from_numpy(state_action).float().to(self.device).unsqueeze(0)
                            
                                    # Predict next state
                                    pred_next_state = self.forward_models[idx].model(state_action)
                                    current_state_tensor = torch.from_numpy(relevant_states).float().to(self.device)
                                    inverse_input = torch.cat([current_state_tensor.unsqueeze(0), pred_next_state], dim=1).to(self.device)
                            
                                    # Compute inverse model results
                                    result = self.inverse_models[idx].model(inverse_input)
                                    grounded_action, uncertainty = result[0], result[1]
                            
                                    if self.uncertainty_setting:
                                        agent_uncertainty_sums[idx] += uncertainty.item()
                                        if uncertainty < self.avg_agent_uncertainties[idx]:

                                            # Determine the position of the active agent in its relevant indices list
                                            active_agent_pos = {idx: agent_info_map[idx].index(idx) for idx in agent_info_map}
                                            
                                            batch_size, num_elements = grounded_action.shape
                                            new_first_dim = num_elements // 8

                                            # Reshape to (N, 8)
                                            reshaped_tensor = grounded_action.view(new_first_dim, 8)

                                            # Find where the agent's index appears in its own subset list
                                            select_idx = active_agent_pos[idx]
                                                
                                            selected_tensor = reshaped_tensor[select_idx]
                    
                                            actions[idx] = torch.argmax(selected_tensor, dim=0).cpu().item()
                                                 
                                            grounded_action_count += 1
                                    else:
                                        
                                        # Determine the position of the active agent in its relevant indices list
                                        active_agent_pos = {idx: agent_info_map[idx].index(idx) for idx in agent_info_map}
                                            
                                        batch_size, num_elements = grounded_action.shape
                                        new_first_dim = num_elements // 8

                                        # Reshape to (N, 8)
                                        reshaped_tensor = grounded_action.view(new_first_dim, 8)

                                        # Find where the agent's index appears in its own subset list
                                        select_idx = active_agent_pos[idx]
                                                
                                        selected_tensor = reshaped_tensor[select_idx]
                    
                                        actions[idx] = torch.argmax(selected_tensor, dim=0).cpu().item()
                                                 
                                        grounded_action_count += 1
                            
    
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
                    if len(self.last_two_uncertainties[idx]) > 2:
                        self.last_two_uncertainties[idx].pop(0)

                    # Update mean uncertainity for next episode
                    self.avg_agent_uncertainties[idx] = np.mean(self.last_two_uncertainties[idx])

                if self.oaat_num < len(self.agents_sim) - 1:
                    self.oaat_num += 1
                else:
                    self.oaat_num = 0

                self.logger.info(
                "Policy training episode: {}, grounded actions taken: {}, last two uncertainties: {}, avg agent uncertainties: {}, grounded actions by agent: {}".format(episode, grounded_action_count, self.last_two_uncertainties, self.avg_agent_uncertainties, ga_by_agent))

            elif self.gattype == "centralized" or self.gattype == "central_inv_dec_fwd":
                
                # Update last_two_uncertainties
                self.last_two_central_uncertainties.append(uncertainty_sum / 360)
                if len(self.last_two_central_uncertainties) > 2:
                    self.last_two_central_uncertainties.pop(0)

                # Update mean uncertainity for next episode
                self.mean_uncertainty = np.mean(self.last_two_central_uncertainties)

                self.logger.info(
                "Policy training episode: {}, grounded actions taken: {}, last two uncertainties: {}, avg uncertainty: {}".format(episode, grounded_action_count, self.last_two_central_uncertainties, self.mean_uncertainty))


            self.writeLog("TRAIN", e, self.metric_sim.real_average_travel_time(), \
                          mean_loss, self.metric_sim.rewards(), self.metric_sim.queue(), self.metric_sim.delay(),
                          self.metric_sim.throughput())
            self.logger.info(
                "Policy training episode: {}, iteration {}/{}, step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(episode, e, self.training_iterations, i, self.steps, \
                                                                                                            mean_loss,
                                                                                                            self.metric_sim.rewards(),
                                                                                                            self.metric_sim.queue(),
                                                                                                            self.metric_sim.delay(),
                                                                                                            int(self.metric_sim.throughput())))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents_sim]
                
            self.logger.info("Policy training episode: {}, iteration {}/{}, real avg travel time:{}".format(episode, e, self.training_iterations, self.metric_sim.real_average_travel_time()))
            for j in range(len(self.world_sim.intersections)):
                self.logger.debug(
                    "Policy training episode: {}, iteration {}/{}, intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(episode, e, self.training_iterations, j, self.metric_sim.lane_rewards()[j], \
                                                                                                  self.metric_sim.lane_queue()[j]))




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

                elif self.gattype == "central_fwd_dec_inv":
                    # Load and split the real and sim data to prepare for forward / inverse model training

                    # Forward data split using real data
                    load_and_split_forward_data("collected/ereal_train.pkl", "collected/ereal_train_full.pkl", "collected/ereal_test_full.pkl",
                                       8, 0.2, 42, "centralized", len(self.agents_real))

                    # Inverse data split using sim data
                    load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full", "collected/esim_test_full",
                                       8, 0.2, 42, "decentralized", len(self.agents_sim))

                    # Train the centralized forward model
                    self.forward_model.train(100, 'forward', len(self.agents_real), 5000 * len(self.agents_real))

                    for idx, ag in enumerate(self.agents_sim):
    
                        # Train the decentralized inverse model
                        self.inverse_models[idx].train(100, 'inverse', idx, 5000, "decentralized")

                elif self.gattype == "central_inv_dec_fwd":
                    # Load and split the real and sim data to prepare for forward / inverse model training

                    # Forward data split using real data
                    load_and_split_forward_data("collected/ereal_train.pkl", "collected/ereal_train_full", "collected/ereal_test_full",
                                       8, 0.2, 42, "decentralized", len(self.agents_real))

                    # Inverse data split using sim data
                    load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full.pkl", "collected/esim_test_full.pkl",
                                       8, 0.2, 42, "centralized", len(self.agents_sim))

                    # Train the centralized inverse model
                    self.inverse_model.train(100, 'inverse', len(self.agents_sim), 5000 * len(self.agents_real))

                    for idx, ag in enumerate(self.agents_sim):
    
                        # Train the decentralized forward model
                        self.forward_models[idx].train(100, 'forward', idx, 5000, "decentralized")
                        

                elif self.gattype == "jlgat":
                    # Load and split the real and sim data to prepare for forward / inverse model training

                    # Forward data split using real data
                    load_and_split_forward_data("collected/ereal_train.pkl", "collected/ereal_train_full", "collected/ereal_test_full",
                                       8, 0.2, 42, "jlgat", len(self.agents_real))

                    # Inverse data split using sim data
                    load_and_split_inverse_data("collected/esim_train.pkl", "collected/esim_train_full", "collected/esim_test_full",
                                       8, 0.2, 42, "jlgat", len(self.agents_sim))

                    for idx, ag in enumerate(self.agents_sim):
                        
                        # Train the decentralized forward model
                        self.forward_models[idx].train(100, 'forward', idx, 5000, "jlgat")
    
                        # Train the decentralized inverse model
                        self.inverse_models[idx].train(100, 'inverse', idx, 5000, "jlgat")
                    
    
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
                    actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
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
                
                elif mode == "jlgat" and self.net == "cityflow1x3":
                    # Joint local information storage for cityflow1x3 network
                    state_action_next_state.append((0, np.concatenate([last_obs[0], last_obs[1]], axis=1), np.concatenate([actions[0], actions[1]], axis=0).reshape(-1, 1), np.concatenate([obs[0], obs[1]], axis=1)))
                    state_action_next_state.append((1, np.array(last_obs).reshape(1, -1), actions.reshape(-1, 1), np.array(obs).reshape(1, -1)))
                    state_action_next_state.append((2, np.concatenate([last_obs[1], last_obs[2]], axis=1), np.concatenate([actions[1], actions[2]], axis=0).reshape(-1, 1), np.concatenate([obs[1], obs[2]], axis=1)))

                elif mode == "jlgat" and self.net == "cityflow4x4":
                    # Joint local information storage for cityflow4x4 network
                    state_action_next_state.append((0, np.concatenate([last_obs[i] for i in [0, 1, 4]], axis=1), np.concatenate([actions[i] for i in [0, 1, 4]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [0, 1, 4]], axis=1)))
                    state_action_next_state.append((1, np.concatenate([last_obs[i] for i in [0, 1, 2, 5]], axis=1), np.concatenate([actions[i] for i in [0, 1, 2, 5]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [0, 1, 2, 5]], axis=1)))
                    state_action_next_state.append((2, np.concatenate([last_obs[i] for i in [1, 2, 3, 6]], axis=1), np.concatenate([actions[i] for i in [1, 2, 3, 6]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [1, 2, 3, 6]], axis=1)))
                    state_action_next_state.append((3, np.concatenate([last_obs[i] for i in [2, 3, 7]], axis=1), np.concatenate([actions[i] for i in [2, 3, 7]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [2, 3, 7]], axis=1)))
                    state_action_next_state.append((4, np.concatenate([last_obs[i] for i in [0, 4, 5, 8]], axis=1), np.concatenate([actions[i] for i in [0, 4, 5, 8]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [0, 4, 5, 8]], axis=1)))
                    state_action_next_state.append((5, np.concatenate([last_obs[i] for i in [1, 4, 5, 6, 9]], axis=1), np.concatenate([actions[i] for i in [1, 4, 5, 6, 9]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [1, 4, 5, 6, 9]], axis=1)))
                    state_action_next_state.append((6, np.concatenate([last_obs[i] for i in [2, 5, 6, 7, 10]], axis=1), np.concatenate([actions[i] for i in [2, 5, 6, 7, 10]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [2, 5, 6, 7, 10]], axis=1)))
                    state_action_next_state.append((7, np.concatenate([last_obs[i] for i in [3, 6, 7, 11]], axis=1), np.concatenate([actions[i] for i in [3, 6, 7, 11]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [3, 6, 7, 11]], axis=1)))
                    state_action_next_state.append((8, np.concatenate([last_obs[i] for i in [4, 8, 9, 12]], axis=1), np.concatenate([actions[i] for i in [4, 8, 9, 12]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [4, 8, 9, 12]], axis=1)))
                    state_action_next_state.append((9, np.concatenate([last_obs[i] for i in [5, 8, 9, 10, 13]], axis=1), np.concatenate([actions[i] for i in [5, 8, 9, 10, 13]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [5, 8, 9, 10, 13]], axis=1)))
                    state_action_next_state.append((10, np.concatenate([last_obs[i] for i in [6, 9, 10, 11, 14]], axis=1), np.concatenate([actions[i] for i in [6, 9, 10, 11, 14]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [6, 9, 10, 11, 14]], axis=1)))
                    state_action_next_state.append((11, np.concatenate([last_obs[i] for i in [7, 10, 11, 15]], axis=1), np.concatenate([actions[i] for i in [7, 10, 11, 15]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [7, 10, 11, 15]], axis=1)))
                    state_action_next_state.append((12, np.concatenate([last_obs[i] for i in [8, 12, 13]], axis=1), np.concatenate([actions[i] for i in [8, 12, 13]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [8, 12, 13]], axis=1)))
                    state_action_next_state.append((13, np.concatenate([last_obs[i] for i in [9, 12, 13, 14]], axis=1), np.concatenate([actions[i] for i in [9, 12, 13, 14]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [9, 12, 13, 14]], axis=1)))
                    state_action_next_state.append((14, np.concatenate([last_obs[i] for i in [10, 13, 14, 15]], axis=1), np.concatenate([actions[i] for i in [10, 13, 14, 15]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [10, 13, 14, 15]], axis=1)))
                    state_action_next_state.append((15, np.concatenate([last_obs[i] for i in [11, 14, 15]], axis=1), np.concatenate([actions[i] for i in [11, 14, 15]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [11, 14, 15]], axis=1)))
                    
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

            self.logger.info("Sim Rollout episode:{}/{}, real avg travel time:{}".format(e, self.episodes, self.metric_sim.real_average_travel_time()))
        for j in range(len(self.world_sim.intersections)):
            self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric_sim.lane_rewards()[j], \
                                                                                    self.metric_sim.lane_queue()[j]))


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
                    # Collect state-action-next_state and agent index for saving
                    for idx, (obs_agent, action_agent) in enumerate(zip(last_obs, actions)):
                        state_action_next_state.append((idx, obs_agent, action_agent, obs[idx]))

                elif mode == "jlgat" and self.net == "cityflow1x3":
                    # Joint local information storage for cityflow1x3 network
                    state_action_next_state.append((0, np.concatenate([last_obs[0], last_obs[1]], axis=1), np.concatenate([actions[0], actions[1]], axis=0).reshape(-1, 1), np.concatenate([obs[0], obs[1]], axis=1)))
                    state_action_next_state.append((1, np.array(last_obs).reshape(1, -1), actions.reshape(-1, 1), np.array(obs).reshape(1, -1)))
                    state_action_next_state.append((2, np.concatenate([last_obs[1], last_obs[2]], axis=1), np.concatenate([actions[1], actions[2]], axis=0).reshape(-1, 1), np.concatenate([obs[1], obs[2]], axis=1)))

                elif mode == "jlgat" and self.net == "cityflow4x4":
                    # Joint local information storage for cityflow4x4 network
                    state_action_next_state.append((0, np.concatenate([last_obs[i] for i in [0, 1, 4]], axis=1), np.concatenate([actions[i] for i in [0, 1, 4]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [0, 1, 4]], axis=1)))
                    state_action_next_state.append((1, np.concatenate([last_obs[i] for i in [0, 1, 2, 5]], axis=1), np.concatenate([actions[i] for i in [0, 1, 2, 5]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [0, 1, 2, 5]], axis=1)))
                    state_action_next_state.append((2, np.concatenate([last_obs[i] for i in [1, 2, 3, 6]], axis=1), np.concatenate([actions[i] for i in [1, 2, 3, 6]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [1, 2, 3, 6]], axis=1)))
                    state_action_next_state.append((3, np.concatenate([last_obs[i] for i in [2, 3, 7]], axis=1), np.concatenate([actions[i] for i in [2, 3, 7]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [2, 3, 7]], axis=1)))
                    state_action_next_state.append((4, np.concatenate([last_obs[i] for i in [0, 4, 5, 8]], axis=1), np.concatenate([actions[i] for i in [0, 4, 5, 8]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [0, 4, 5, 8]], axis=1)))
                    state_action_next_state.append((5, np.concatenate([last_obs[i] for i in [1, 4, 5, 6, 9]], axis=1), np.concatenate([actions[i] for i in [1, 4, 5, 6, 9]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [1, 4, 5, 6, 9]], axis=1)))
                    state_action_next_state.append((6, np.concatenate([last_obs[i] for i in [2, 5, 6, 7, 10]], axis=1), np.concatenate([actions[i] for i in [2, 5, 6, 7, 10]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [2, 5, 6, 7, 10]], axis=1)))
                    state_action_next_state.append((7, np.concatenate([last_obs[i] for i in [3, 6, 7, 11]], axis=1), np.concatenate([actions[i] for i in [3, 6, 7, 11]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [3, 6, 7, 11]], axis=1)))
                    state_action_next_state.append((8, np.concatenate([last_obs[i] for i in [4, 8, 9, 12]], axis=1), np.concatenate([actions[i] for i in [4, 8, 9, 12]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [4, 8, 9, 12]], axis=1)))
                    state_action_next_state.append((9, np.concatenate([last_obs[i] for i in [5, 8, 9, 10, 13]], axis=1), np.concatenate([actions[i] for i in [5, 8, 9, 10, 13]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [5, 8, 9, 10, 13]], axis=1)))
                    state_action_next_state.append((10, np.concatenate([last_obs[i] for i in [6, 9, 10, 11, 14]], axis=1), np.concatenate([actions[i] for i in [6, 9, 10, 11, 14]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [6, 9, 10, 11, 14]], axis=1)))
                    state_action_next_state.append((11, np.concatenate([last_obs[i] for i in [7, 10, 11, 15]], axis=1), np.concatenate([actions[i] for i in [7, 10, 11, 15]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [7, 10, 11, 15]], axis=1)))
                    state_action_next_state.append((12, np.concatenate([last_obs[i] for i in [8, 12, 13]], axis=1), np.concatenate([actions[i] for i in [8, 12, 13]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [8, 12, 13]], axis=1)))
                    state_action_next_state.append((13, np.concatenate([last_obs[i] for i in [9, 12, 13, 14]], axis=1), np.concatenate([actions[i] for i in [9, 12, 13, 14]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [9, 12, 13, 14]], axis=1)))
                    state_action_next_state.append((14, np.concatenate([last_obs[i] for i in [10, 13, 14, 15]], axis=1), np.concatenate([actions[i] for i in [10, 13, 14, 15]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [10, 13, 14, 15]], axis=1)))
                    state_action_next_state.append((15, np.concatenate([last_obs[i] for i in [11, 14, 15]], axis=1), np.concatenate([actions[i] for i in [11, 14, 15]], axis=0).reshape(-1, 1), np.concatenate([last_obs[i] for i in [11, 14, 15]], axis=1)))
                
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

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        cityflow_trained_save = '/home/derekmei233/DaRL/Sim2Real_TSC/data/output_data/tsc/sim2real_paper/compare/cityflow_train_dqn_pickout/cityflow1x1/test/model/'
        cityflow_sub = 'cityflow_dqn/cityflow1x1'
        sumo_sub = 'sumo_dqn/sumohz1x1'

        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            path_sub = cityflow_sub
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        else:
            path_sub = sumo_sub
            
        self.metric.clear()
        # load_path = 'data/output_data/tsc/'+path_sub+'/test/model/'
        load_path = cityflow_trained_save

        if not drop_load:
            print(".......not droping loading, generating random agents.......")
            [ag.load_model(self.episodes) for ag in self.agents]

        else:
            # loaded_agent_list = []
            import re

            base_path = sys.path[0] + Registry.mapping['logger_mapping']['path'].path + '/model/'
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            print(base_path)
            history_dir_path = sys.path[0] + "/history_save/" + Registry.mapping['command_mapping']['setting'].param['world'] + datetime.datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
            # os.makedirs(history_dir_path)
            history_save_path = history_dir_path + "/history.txt"
            # pass_save_path = history_dir_path + "/action_pass.txt"
            num = 200
            candidate_list = []
            
            # for files in os.listdir(load_path):  
            #     print("files", files)
            #     if files.startswith(str(num)):
            #         candidate_list.append(files)
                    # print(files)
            candidate_list.sort(key=lambda l: int(re.findall('\d+', l[3:])[0]))
            print(candidate_list)
            for i in range(len(candidate_list)):
                # [ag.load_model(e="", customized_path=base_path + candidate_list[i]) for ag in self.agents]
                [ag.load_model(e="", customized_path=load_path + str(num) + "_" + str(ag.rank) + ".pt") for ag in
                 self.agents]

        attention_mat_list = []
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        print("-------self.test_steps-------")
        print(self.test_steps)
        # my record files:

        history_record = []
        struc = []
        for a in self.agents:
            a_struc = []
            for ls in a.ob_generator.lanes:
                for l in ls:
                    a_struc.append(l)
            struc.append(a_struc)
        history_record.append(str(struc))

        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    temp = str([str(int(i)).ljust(5, ' ') for i in obs[idx][0]])
                    temp_action = ag.get_action(obs[idx], phases[idx], test=True)
                    # 3 placeholder to align output state
                    actions.append(temp_action)

                    temp +=  ":" + str(temp_action)
                    temp = temp.replace(',', '').replace("'", "")
                    history_record.append(temp)

                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())

                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break

        # if Registry.mapping['command_mapping']['setting'].param['debug']:
        # with open(file=load_path+"/record.txt", mode='a+', encoding='utf-8') as wf:
        #     for line in history_record:
        #         net_info =Registry.mapping['command_mapping']['setting'].param['network']
        #         wf.writelines(net_info + ":   " +"Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
        #             self.metric.real_average_travel_time(), \
        #             self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()
        #             ) + "\n")
        #     # calculate existing vehicles in each phase (fixedtime only)
        #     traj = self.env.world.vehicle_trajectory
        #     path_record = log_passing_lane_actinon(traj, self.world.intersections[0].startlanes)
        #     # write_action_record(pass_save_path, path_record, a_struc)
            

        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
            self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))

        return self.metric

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
