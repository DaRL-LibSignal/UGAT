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

        # Delete files used for gat training to ensure fresh files for training
        path = 'collected'
        files_to_delete = ['esim_train.pkl', 'ereal_train.pkl', 'ereal_train_full.pkl', 'ereal_test_full.pkl', 'esim_train_full.pkl', 'esim_test_full.pkl']
        
        for filename in files_to_delete:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Existing {file_path} has been deleted")
        
        # Initialize GAT models
        if self.gat == True:

            self.total_decision_num = 0
            self.mean_uncertainty = 0
            self.uncertainties_last_episodes = []

            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # Considers number of agents in both input and output dimension calculations to allow for multi-agent setting
            num_agents = len(self.agents_real)
            
            # Initialize centralized GAT models
            if self.gattype == "centralized":
                print(f"\n------- INITIALIZING GAT MODELS CENTRALIZED -------\n")
                gat_path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
                self.forward_model = NN_predictor(self.logger,
                                                (self.agents_real[0].ob_generator.ob_length * num_agents + self.agents_real[0].action_space.n * num_agents),
                                                self.agents_real[0].ob_generator.ob_length * num_agents, self.device, gat_path, 'collected/ereal_train_full.pkl')
                self.inverse_model = UNCERTAINTY_predictor(self.logger, self.agents_real[0].ob_generator.ob_length * num_agents * 2,
                                                self.agents_real[0].action_space.n * num_agents, self.device, gat_path,
                                                'collected/esim_train_full.pkl', backward=True)

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
        # Run for a set number of episodes
        for e in range(self.episodes):
    
            # Sim rollout + collect data
            self.sim_rollout(e)
    
            # Real rollout + collect data
            self.train_test(e)
    
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
    
        :param iteration: int, the current iteration number for logging
        :return: None
        """
        flush = 0
        
        for e in range(self.training_iterations):
            uncertainty_sum = 0
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

                    # Handle centralized GAT
                    if self.gattype == "centralized":
                        # Combine data for all agents
                        combined_state = np.concatenate([state.flatten() for state in last_obs[:len(self.agents_real)]])  # Flatten and concatenate observations
                        one_hot_actions = np.concatenate([
                            idx2onehot(np.array([action]), 8).flatten() for action in actions[:len(self.agents_real)]  # One-hot encode actions
                        ])
                    
                        # Combine observations and one-hot actions
                        joint_state_action = np.concatenate([combined_state, one_hot_actions], axis=0)  # Shape: [combined state and actions]
                    
                        # Convert it to a PyTorch tensor
                        joint_state_action = torch.from_numpy(joint_state_action).float().to(self.device)
                    
                        # Add the batch dimension using unsqueeze
                        joint_state_action = joint_state_action.unsqueeze(0)  # Shape: (1, combined input size)
                    
                        # Get predicted next state from the forward model
                        pred_next_state = self.forward_model.model(joint_state_action)

                        # Convert the combined current state to a PyTorch tensor
                        current_state_tensor = torch.from_numpy(combined_state).float().to(self.device)
                    
                        # Combine current state with predicted next state for the inverse model input
                        inverse_input = torch.cat([current_state_tensor.unsqueeze(0), pred_next_state], dim=1).to(self.device)  # Concatenate along feature axis
                    
                        # Get grounded action from inverse model
                        result = self.inverse_model.model(inverse_input)

                        grounded_action, uncertainty = result[0], result[1]# Use torch.argmax to get the index of the highest value for each agent

                        # Reshape grounded_action to (num_agents, 8)
                        num_agents = grounded_action.shape[1] // 8
                        grounded_action_reshaped = grounded_action.view(num_agents, 8)
                        
                        # Find argmax for each agent (Final grounded actions to use in training)
                        action_indices = torch.argmax(grounded_action_reshaped, dim=1)
                        action_indices = action_indices.cpu()

                        uncertainty_sum += uncertainty.item()

                        # If uncertainity less than dynamic grounding rate, take grounded actions
                        if uncertainty < self.mean_uncertainty:
                            actions = action_indices
                            grounded_action_count += 1
                        
                    actions = actions.flatten()
                    rewards_list = []
                    for _ in range(self.action_interval):
                        # Use grounded action
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

            # Calculate mean uncertainty for this episode
            mean_episode_uncertainty = uncertainty_sum / 360
            self.uncertainties_last_episodes.append(mean_episode_uncertainty)
            
            # Maintain the last 3 episodes of uncertainties
            if len(self.uncertainties_last_episodes) > 3:
                self.uncertainties_last_episodes.pop(0)
            
            # Update mean_uncertainty based on the last 3 episodes
            self.mean_uncertainty = np.mean(self.uncertainties_last_episodes)

            print(f"mean_episode_uncertainty: {mean_episode_uncertainty}")
            print(f"self.uncertainties_last_episodes: {self.uncertainties_last_episodes}")
            print(f"self.mean_uncertainty: {self.mean_uncertainty}")
            
            
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            # Log grounded action count
            self.logger.info("Episode {}, Grounded actions taken: {}, Mean uncertainty (this episode): {}, Check against uncertainty: {}".format(episode, grounded_action_count, mean_episode_uncertainty, self.mean_uncertainty))
    
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
                    self.forward_model.train(100, sign=f'forward', agent_num=len(self.agents_real))

                    # Train the centralized inverse model
                    self.inverse_model.train(100, sign=f'inverse', agent_num=len(self.agents_sim))
                    
    
    def sim_rollout(self, e):
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
    
                # Store the transition (state, action, next_state)
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


    def train_test(self, e):
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

                # Collect state-action-next_state for saving
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
