import torch
import pickle
from sklearn.model_selection import train_test_split
from agent.utils import idx2onehot
from torch import nn, no_grad, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Subset
import random

def save_data_to_pkl(data, file_path):
    """
    Save or append data to a .pkl file.

    Parameters:
        data (list): The data to save.
        file_path (str): Path to the .pkl file.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_data = pickle.load(f)
        data.extend(existing_data)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_and_split_forward_data(
    pkl_file_path, train_pkl_file, test_pkl_file, spices=8, test_size=0.2, random_seed=42, mode="decentralized", num_agents=1
):
    """
    Load and process data for forward models, storing agent-specific data for decentralized mode.
    """
    if not os.path.exists(pkl_file_path):
        raise FileNotFoundError(f"Error: File {pkl_file_path} not found.")

    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    if mode == "jlgat" or mode == "decentralized":
        agent_data = {agent_idx: [] for agent_idx in range(num_agents)}
    
        for record in data:
            agent_idx = record[0]
            states = record[1]
            actions = record[2]
            next_states = record[3]

            for state, action, next_state in zip(states, actions, next_states):
                
                one_hot_actions = torch.tensor(
                np.concatenate([idx2onehot(np.array([action]), spices) for action in actions], axis=0),
                dtype=torch.float32
            )
            
            agent_data[agent_idx].append((
                    torch.tensor(states, dtype=torch.float32),
                    one_hot_actions,
                    torch.tensor(next_states, dtype=torch.float32)
                ))
    
        # Now, split and save data for each agent
        for agent_idx, agent_records in agent_data.items():
            # Separate state, action, and next state
            states = torch.stack([rec[0] for rec in agent_records])
            actions = torch.stack([rec[1] for rec in agent_records])
            targets = torch.stack([rec[2] for rec in agent_records])
            
            # Split the data into train and test sets
            states_train, states_test, actions_train, actions_test, targets_train, targets_test = train_test_split(
                states.numpy(), actions.numpy(), targets.numpy(), test_size=test_size, random_state=random_seed
            )
    
            # Prepare the train and test datasets (keeping state and action separate)
            train_data = [(states_train[i], actions_train[i], targets_train[i]) for i in range(len(states_train))]
            test_data = [(states_test[i], actions_test[i], targets_test[i]) for i in range(len(states_test))]
    
            # Save the data for each agent to separate files
            save_data_to_pkl(train_data, f"{train_pkl_file}_agent_{agent_idx}.pkl")
            save_data_to_pkl(test_data, f"{test_pkl_file}_agent_{agent_idx}.pkl")

    
    elif mode == "centralized":
        agent_data = []
        
        # Combine data for all agents
        for idx, record in enumerate(data):
            states = record[0]  # List of states at time t
            actions = record[1]  # Actions taken
            next_states = record[2]  # States at time t+1
    
            one_hot_actions = np.concatenate([idx2onehot(np.array([action]), spices) for action in actions], axis=0)
    
            # Append the data as a tuple of arrays
            agent_data.append((
                np.array(states).squeeze(1),  # (3, 24)
                one_hot_actions,  # (3, 8)
                np.array(next_states).squeeze(1)  # (3, 24)
            ))
    
        # Instead of converting to a NumPy array, just work with the list directly
        # Create train-test split indices at the record level
        train_idx, test_idx = train_test_split(np.arange(len(agent_data)), test_size=test_size, random_state=random_seed)
    
        # Extract train and test subsets based on indices
        train_data = [agent_data[i] for i in train_idx]
        test_data = [agent_data[i] for i in test_idx]
    
        # Save train and test data
        save_data_to_pkl(train_data, train_pkl_file)
        save_data_to_pkl(test_data, test_pkl_file)




def load_and_split_inverse_data(pkl_file_path, train_pkl_file, test_pkl_file, spices=8, test_size=0.2, random_seed=42, mode="decentralized", num_agents=1):
    """
    Load and process data for inverse models, storing agent-specific data for decentralized mode.
    """
    if not os.path.exists(pkl_file_path):
        raise FileNotFoundError(f"Error: File {pkl_file_path} not found.")

    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    if mode == "decentralized":
        agent_data = {agent_idx: [] for agent_idx in range(num_agents)}

        # Data comes in as (agent idx, individual state, joint neighbor states, joint neighbor actions, next individual state, individual action taken)
        for record in data:
            agent_idx = record[0]
            states = record[1]
            actions = record[2]
            next_states = record[3]
            
            agent_data[agent_idx].append((
                    torch.tensor(states, dtype=torch.float32),
                    torch.tensor(actions, dtype=torch.long),
                    torch.tensor(next_states, dtype=torch.float32)
                ))
    
        # Now, split and save data for each agent
        for agent_idx, agent_records in agent_data.items():
            
            # Separate state, action, and next state
            states = torch.stack([rec[0] for rec in agent_records])
            next_states = torch.stack([rec[2] for rec in agent_records])
            targets = torch.stack([rec[1] for rec in agent_records])
            
            # Split the data into train and test sets
            states_train, states_test, next_states_train, next_states_test, targets_train, targets_test = train_test_split(
                states.numpy(), next_states.numpy(), targets.numpy(), test_size=test_size, random_state=random_seed
            )
    
            # Prepare the train and test datasets (keeping state and action separate)
            train_data = [(states_train[i], next_states_train[i], targets_train[i]) for i in range(len(states_train))]
            test_data = [(states_test[i], next_states_test[i], targets_test[i]) for i in range(len(states_test))]
    
            # Save the data for each agent to separate files
            save_data_to_pkl(train_data, f"{train_pkl_file}_agent_{agent_idx}.pkl")
            save_data_to_pkl(test_data, f"{test_pkl_file}_agent_{agent_idx}.pkl")

    elif mode == "jlgat3":
        agent_data = {agent_idx: [] for agent_idx in range(num_agents)}

        # Data comes in as (agent idx, individual state, joint neighbor states, joint neighbor actions, next individual state, individual action taken)
        for record in data:
            agent_idx = record[0]
            states = record[1]
            n_actions = record[2]
            next_states = record[3]
            ind_action = record[4]

            for state, action, next_state, i_action in zip(states, n_actions, next_states, ind_action):
                
                one_hot_actions = torch.tensor(
                np.concatenate([idx2onehot(np.array([action]), spices) for action in n_actions], axis=0),
                dtype=torch.float32
            )
            
            agent_data[agent_idx].append((
                    torch.tensor(states, dtype=torch.float32),
                    one_hot_actions,
                    torch.tensor(next_states, dtype=torch.float32),
                    torch.tensor(i_action, dtype=torch.long)
                ))
    
        # Now, split and save data for each agent
        for agent_idx, agent_records in agent_data.items():
            # Separate state, action, and next state
            states = torch.stack([rec[0] for rec in agent_records])
            n_actions = torch.stack([rec[1] for rec in agent_records])
            next_states = torch.stack([rec[2] for rec in agent_records])
            targets = torch.stack([rec[3] for rec in agent_records])
            
            # Split the data into train and test sets
            states_train, states_test, n_actions_train, n_actions_test, next_states_train, next_states_test, targets_train, targets_test = train_test_split(
                states.numpy(), n_actions.numpy(), next_states.numpy(), targets, test_size=test_size, random_state=random_seed
            )
    
            # Prepare the train and test datasets (keeping state and action separate)
            train_data = [(states_train[i], n_actions_train[i], next_states_train[i], targets_train[i]) for i in range(len(states_train))]
            test_data = [(states_test[i], n_actions_test[i], next_states_test[i], targets_test[i]) for i in range(len(states_test))]
    
            # Save the data for each agent to separate files
            save_data_to_pkl(train_data, f"{train_pkl_file}_agent_{agent_idx}.pkl")
            save_data_to_pkl(test_data, f"{test_pkl_file}_agent_{agent_idx}.pkl")

    elif mode == "jlgat":
        agent_data = {agent_idx: [] for agent_idx in range(num_agents)}

        # Data comes in as (agent idx, individual state, joint neighbor states, joint neighbor actions, next individual state, individual action taken)
        for record in data:
            agent_idx = record[0]
            states = record[1]
            n_states = record[2]
            n_actions = record[3]
            next_states = record[4]
            ind_action = record[5]

            for state, n_state, action, next_state, i_action in zip(states, n_states, n_actions, next_states, ind_action):
                
                one_hot_actions = torch.tensor(
                np.concatenate([idx2onehot(np.array([action]), spices) for action in n_actions], axis=0),
                dtype=torch.float32
            )
            
            agent_data[agent_idx].append((
                    torch.tensor(states, dtype=torch.float32),
                    torch.tensor(n_states, dtype=torch.float32),
                    one_hot_actions,
                    torch.tensor(next_states, dtype=torch.float32),
                    torch.tensor(i_action, dtype=torch.long)
                ))
    
        # Now, split and save data for each agent
        for agent_idx, agent_records in agent_data.items():
            # Separate state, action, and next state
            states = torch.stack([rec[0] for rec in agent_records])
            n_states = torch.stack([rec[1] for rec in agent_records])
            n_actions = torch.stack([rec[2] for rec in agent_records])
            next_states = torch.stack([rec[3] for rec in agent_records])
            targets = torch.stack([rec[4] for rec in agent_records])
            
            # Split the data into train and test sets
            states_train, states_test, n_states_train, n_states_test, n_actions_train, n_actions_test, next_states_train, next_states_test, targets_train, targets_test = train_test_split(
                states.numpy(), n_states.numpy(), n_actions.numpy(), next_states.numpy(), targets.numpy(), test_size=test_size, random_state=random_seed
            )
    
            # Prepare the train and test datasets (keeping state and action separate)
            train_data = [(states_train[i], n_states_train[i], n_actions_train[i], next_states_train[i], targets_train[i]) for i in range(len(states_train))]
            test_data = [(states_test[i], n_states_test[i], n_actions_test[i], next_states_test[i], targets_test[i]) for i in range(len(states_test))]
    
            # Save the data for each agent to separate files
            save_data_to_pkl(train_data, f"{train_pkl_file}_agent_{agent_idx}.pkl")
            save_data_to_pkl(test_data, f"{test_pkl_file}_agent_{agent_idx}.pkl")

    
    elif mode == "centralized":
        agent_data = []
        
        # Combine data for all agents
        for idx, record in enumerate(data):
            states = record[0]  # List of states at time t
            actions = record[1]  # Actions taken
            next_states = record[2]  # States at time t+1
    
            # one_hot_actions = np.concatenate([idx2onehot(np.array([action]), spices) for action in actions], axis=0)
    
            # Append the data as a tuple of arrays
            agent_data.append((
                np.array(states).squeeze(1),  # (3, 24)
                np.array(next_states).squeeze(1),  # (3, 24)
                actions # (3, 8)
            ))
    
        # Instead of converting to a NumPy array, just work with the list directly
        # Create train-test split indices at the record level
        train_idx, test_idx = train_test_split(np.arange(len(agent_data)), test_size=test_size, random_state=random_seed)
    
        # Extract train and test subsets based on indices
        train_data = [agent_data[i] for i in train_idx]
        test_data = [agent_data[i] for i in test_idx]
    
        # Save train and test data
        save_data_to_pkl(train_data, train_pkl_file)
        save_data_to_pkl(test_data, test_pkl_file)





class NN_predictor(object):
    def __init__(self, logger, state_dim, action_dim, out_dim, DEVICE, model_dir, data_dir, backward=False, history=1, mode=''):
        super(NN_predictor, self).__init__()
        self.epo = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.out_dim = out_dim
        self.model = None
        self.mode = mode
        self.backward = backward
        self.make_model()
        self.DEVICE = DEVICE
        self.model.to(self.DEVICE).float()
        if not backward:
            self.criterion = nn.MSELoss()
            self.learning_rate = 0.0001
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.learning_rate = 0.00001
        
        self.history = history
        self.batch_size = 64
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.online_optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logger = logger

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def predict(self, state, action):
        state, action = state.to(self.DEVICE), action.to(self.DEVICE)
        with torch.no_grad():
            result = self.model(state, action)
        return result

    def make_model(self):
        if self.mode == 'central':
            self.model = Central_N_net(self.state_dim, self.action_dim, self.out_dim, self.backward).float()
        elif self.mode == "jlgat4":
            self.model = N_net_noAction(self.state_dim, self.action_dim, self.out_dim, self.backward).float()
        elif self.mode == "jlgat5":
            self.model = N_net_noState(self.state_dim, self.action_dim, self.out_dim, self.backward).float()
        else:
            self.model = N_net(self.state_dim, self.action_dim, self.out_dim, self.backward).float()

    def load_model(self):
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        name = f"NN_inference_{txt}.pt"
        model_name = os.path.join(self.model_dir, name)
        self.model = N_net(self.in_dim, self.out_dim, self.backward)
        self.model.load_state_dict(torch.load(model_name))
        self.model = self.model.float().to(self.DEVICE)

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        name = f"NN_inference_{txt}.pt"
        model_name = os.path.join(self.model_dir, name)
        torch.save(self.model.state_dict(), model_name)
    
    def train(self, epochs, sign, agent_num=None, max_samples=5000, mode="centralized", jlnet=0):
        """
        Train the model on real data compatible with the data formatting of the forward split function.
        """
        train_loss = 0.0
    
        # Determine dataset path based on mode and agent_num
        if mode == "decentralized" and agent_num is not None:
            dataset_path = f"collected/ereal_train_full_agent_{agent_num}.pkl"
        elif mode == "jlgat" and agent_num is not None:
            dataset_path = f"collected/ereal_train_full_agent_{agent_num}.pkl"
        elif mode == "centralized":
            dataset_path = "collected/ereal_train_full.pkl"
        else:
            raise ValueError("Invalid mode or agent_num configuration for training.")

        # Load the dataset
        full_dataset = PKLDataset(dataset_path)
    
        # Select a subset of the dataset
        if max_samples and max_samples < len(full_dataset):
            subset_size = max_samples
        else:
            subset_size = len(full_dataset)
        subset_indices = random.sample(range(len(full_dataset)), subset_size)
    
        train_dataset = Subset(full_dataset, subset_indices)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
        if self.backward:
            txt = "inverse"
        else:
            txt = "forward"
    
        if agent_num is not None:
            self.logger.info(f"{txt} model: training agent {agent_num}.")
        else:
            print(f"Epoch {self.epo - 1} Training")
    
        # Ensure model is on the correct device
        self.model.to(self.DEVICE)
    
        for e in range(epochs):
            for i, data in enumerate(train_loader):
                # Move data to the device
                state, action, y_true = data  # Assuming dataset returns (state, action, target)
                state = state.to(self.DEVICE, non_blocking=True)
                action = action.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)

                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass with separate inputs
                y_pred = self.model(state, action)

                if mode == "centralized":
                    # Compute the loss
                    loss = self.criterion(y_pred, y_true)
                elif mode == "jlgat" and jlnet == 4:
                    loss = self.criterion(y_pred.squeeze(1), y_true.squeeze(1))
                elif mode == "jlgat" and jlnet == 5:
                    loss = self.criterion(y_pred.squeeze(1), y_true.squeeze(1))
                elif mode == "jlgat":
                    # Compute the loss
                    loss = self.criterion(y_pred, y_true.squeeze(1))
                else:
                    # Compute the loss
                    loss = self.criterion(y_pred.squeeze(1), y_true.squeeze(1))

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
    
                train_loss += loss.item()
    
            # Log progress at the first and last epoch
            if e == 0 or e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f"Epoch {e}: {txt} train average loss {ave_loss}.")
    
                # Evaluate the model
                if self.backward:
                    test_loss = self.testest_inverset(e, txt, agent_num, mode)
                else:
                    test_loss = self.test(e, txt, agent_num, mode, jlnet)
    
            # Reset train loss for the next epoch
            train_loss = 0.0
    
        # Increment the epoch counter
        self.epo += 1
    
        # Return value if inverse training
        if sign == "inverse":
            return 0



    def test(self, e, txt, agent_num=None, mode="centralized", jlnet=0):
        test_loss = 0.0
    
        # Load the validation dataset corresponding to the specified agent_num
        if mode == "decentralized" or mode == "jlgat":
            dataset_path = f'collected/ereal_test_full_agent_{agent_num}.pkl'
        else:
            dataset_path = 'collected/ereal_test_full.pkl'
    
        # Load the dataset from the corresponding .pkl file
        test_dataset = PKLDataset(dataset_path)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
        # Ensure model is on the correct device
        self.model.to(self.DEVICE)
    
        # Disable gradient computation during testing
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # Move data to the device
                state, action, y_true = data  # Assuming dataset returns (state, action, target)
                state = state.to(self.DEVICE, non_blocking=True)
                action = action.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)
                
                if mode == "centralized":
                    # Forward pass with separate inputs
                    y_pred = self.model(state, action)

                    # Compute the loss
                    loss = self.criterion(y_pred, y_true)

                elif mode == "jlgat" and jlnet == 4:
                    # Forward pass with separate inputs
                    y_pred = self.model(state, action)
                    
                    # Compute the loss
                    loss = self.criterion(y_pred.squeeze(1), y_true.squeeze(1))

                elif mode == "jlgat" and jlnet == 5:
                    # Forward pass with separate inputs
                    y_pred = self.model(state, action)
                    
                    # Compute the loss
                    loss = self.criterion(y_pred.squeeze(1), y_true.squeeze(1))

                elif mode == "jlgat":
                    # Forward pass with separate inputs
                    y_pred = self.model(state, action)
                    
                    # Compute the loss
                    loss = self.criterion(y_pred, y_true.squeeze(1))
                    
                else:
                    # Forward pass with separate inputs
                    y_pred = self.model(state, action)

                    # Compute the loss
                    loss = self.criterion(y_pred, y_true.squeeze(1))
    
                # Compute the loss
                # loss = self.criterion(y_pred, y_true)
                test_loss += loss.item()
    
        # Calculate average test loss
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        
        # Switch the model back to training mode (if needed)
        self.model.train()
        
        return test_loss


# Custom Dataset class to load the .pkl file
class PKLDataset(Dataset):
    def __init__(self, pkl_file, flag=''):
        self.flag = flag
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)  # Load the (features, targets) list from the .pkl file
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):

        if self.flag == 'jlg':
            states, n_states, n_actions, pred_state, targets = self.data[idx]
            return (
                torch.tensor(states, dtype=torch.float32),
                torch.tensor(n_states, dtype=torch.float32),
                torch.tensor(n_actions, dtype=torch.float32),
                torch.tensor(pred_state, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32)
            )

        elif self.flag == 'jlg3':
            states, n_actions, pred_state, targets = self.data[idx]
            return (
                torch.tensor(states, dtype=torch.float32),
                torch.tensor(n_actions, dtype=torch.float32),
                torch.tensor(pred_state, dtype=torch.float32),
                targets
            )
        
        elif self.flag == 'central':
            states, next_states, targets = self.data[idx]
            return (
                torch.tensor(states, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.long)
            )
        else:
            states, actions, targets = self.data[idx]
            return (
                torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32)
            )

import torch
import torch.nn as nn
import torch.nn.functional as F

class N_net(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(N_net, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Per-agent processing after concatenation
        self.agent_fc = nn.Linear(128, 128)  # 64 (state) + 64 (action)
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(state_dim[0] * 128, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        # Concatenate state and action per agent
        x = torch.cat((state_out, action_out), dim=-1)  # (batch, num_agents, 128)
        
        # Process per-agent embeddings
        x = F.relu(self.agent_fc(x))  # (batch, num_agents, 128)
        
        # Flatten across agents
        x = x.view(batch_size, -1)  # (batch, num_agents * 128)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = self.dense_5(x)
        
        return x



class N_net_noAction(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(N_net_noAction, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(state_dim[0] * 64 + 64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        x = torch.cat((state_out, action_out), dim=1)  # (batch, num_agents, 128)

        x = x.view(state_out.shape[0], 1, -1)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = self.dense_5(x)
        
        return x



class N_net_noState(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(N_net_noState, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(action_dim[0] * 64 + 64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        x = torch.cat((state_out, action_out), dim=1)  # (batch, num_agents, 128)

        x = x.view(state_out.shape[0], 1, -1)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = self.dense_5(x)
        
        return x


class Central_N_net(nn.Module):
    def __init__(self, state_dim, action_dim, size_out, backward):
        super(Central_N_net, self).__init__()
        self.backward = backward
        
        # Separate input layers per agent
        self.state_fc = nn.Linear(state_dim[-1], 64)  # Encode state
        self.action_fc = nn.Linear(action_dim[-1], 64)  # Encode action
        
        # Per-agent processing after concatenation
        self.agent_fc = nn.Linear(128, 128)  # 64 (state) + 64 (action)
        
        # Fully connected layers after aggregation
        self.dense_2 = nn.Linear(state_dim[0] * 128, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 192)
        self.dense_5 = nn.Linear(int(192 / state_dim[0]), size_out)

    def forward(self, state, action):
        
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = state.shape[:2]
        
        # Process each agent's state and action separately
        state_out = F.relu(self.state_fc(state))  # (batch, num_agents, 64)
        action_out = F.relu(self.action_fc(action))  # (batch, num_agents, 64)
        
        # Concatenate state and action per agent
        x = torch.cat((state_out, action_out), dim=-1)  # (batch, num_agents, 128)
        
        # Process per-agent embeddings
        x = F.relu(self.agent_fc(x))  # (batch, num_agents, 128)

        # Concatenate all states and actions
        x = x.view(batch_size, 1, -1)  # (batch, 1, -1)
        
        # Fully connected layers
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))

        # Reshape output back to per-agent next state
        x = x.view(batch_size, num_agents, -1)  # (batch, num_agents, -1)
        
        x = self.dense_5(x)

        return x


class UNCERTAINTY_predictor(object):
    def __init__(self, logger, ind_state_dim, n_state_dim, action_dim, pred_state_dim, out_dim, DEVICE, model_dir, data_dir, backward=False, history=1, mode=''):
        super(UNCERTAINTY_predictor, self).__init__()
        self.epo = 0
        self.ind_state_dim = ind_state_dim
        self.n_state_dim = n_state_dim
        self.pred_state_dim = pred_state_dim
        self.action_dim = action_dim
        self.out_dim = out_dim
        self.model =None
        self.backward = backward
        self.make_model(mode)
        self.DEVICE = DEVICE
        self.model.to(DEVICE).float()
        if not backward:
            self.criterion = nn.MSELoss()
            self.learning_rate = 0.0001
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.learning_rate = 0.00001
        
        self.history = history
        self.batch_size = 64
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.online_optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logger = logger

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def train(self, epochs, sign, agent_num=None, max_samples=5000, mode="centralized", jlnet=0):
        train_loss = 0.0
        
        # Load the dataset corresponding to the specified agent_num
        if mode == "decentralized" or mode == "jlgat":
            dataset_path = f'collected/esim_train_full_agent_{agent_num}.pkl'
        else:
            dataset_path = 'collected/esim_train_full.pkl'

        # Load the dataset from the corresponding .pkl file
        if mode == "decentralized" or jlnet == 2:
            full_dataset = PKLDataset(dataset_path)
        elif jlnet == 3:
            full_dataset = PKLDataset(dataset_path, 'jlg3')
        elif mode == 'centralized':
            full_dataset = PKLDataset(dataset_path, 'central')
        else:
            full_dataset = PKLDataset(dataset_path, 'jlg')
        
        # Determine subset size based on max_samples
        subset_size = min(max_samples, len(full_dataset)) if max_samples else len(full_dataset)
        
        # Randomly select indices for the subset
        subset_indices = random.sample(range(len(full_dataset)), subset_size)
        
        # Create a subset of the dataset
        train_dataset = Subset(full_dataset, subset_indices)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        
        if agent_num is not None:
            self.logger.info(f'{txt} model, training agent {agent_num}.')
        else:
            print(f"Epoch {self.epo - 1} Training")
        
        # Ensure the model is on the correct device
        self.model.to(self.DEVICE)
    
        for e in range(epochs):
            for i, data in enumerate(train_loader):
                # Move data to the device
                if mode == "decentralized" or jlnet == 2 or mode == "centralized":
                    state, pred_state, y_true = data
                    state = state.to(self.DEVICE, non_blocking=True)
                    pred_state = pred_state.to(self.DEVICE, non_blocking=True)
                    y_true = y_true.to(self.DEVICE, non_blocking=True)
                elif jlnet == 3:
                    state, n_action, pred_state, y_true = data
                    state = state.to(self.DEVICE, non_blocking=True)
                    n_action = n_action.to(self.DEVICE, non_blocking=True)
                    pred_state = pred_state.to(self.DEVICE, non_blocking=True)
                    y_true = y_true.to(self.DEVICE, non_blocking=True)
                else:
                    state, n_state, n_action, pred_state, y_true = data
                    state = state.to(self.DEVICE, non_blocking=True)
                    n_state = n_state.to(self.DEVICE, non_blocking=True)
                    n_action = n_action.to(self.DEVICE, non_blocking=True)
                    pred_state = pred_state.to(self.DEVICE, non_blocking=True)
                    y_true = y_true.to(self.DEVICE, non_blocking=True)
                

                # Zero the gradients
                self.optimizer.zero_grad()
    
                # Forward pass with separate inputs
                if mode == "decentralized" or jlnet == 2 or mode == "centralized":
                    result = self.model(state, pred_state)
                elif jlnet == 3:
                    result = self.model(state, n_action, pred_state)
                else:
                    result = self.model(state, n_state, n_action, pred_state)
        
                y_pred, uncertainty = result[0], result[1]

                # standard loss
                if mode == "decentralized":
                    loss = self.criterion(y_pred.squeeze(1), y_true.squeeze().long())
                elif jlnet == 2:
                    loss = self.criterion(y_pred.permute(0, 2, 1), y_true.squeeze(-1).long())
                elif jlnet == 3:
                    loss = self.criterion(y_pred.permute(0, 2, 1), y_true.long())
                elif mode == "centralized":
                    loss = self.criterion(y_pred.permute(0, 2, 1), y_true.squeeze(-1))
                else:
                    loss = self.criterion(y_pred, y_true.squeeze().long())
                
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            
            # Log progress at the first and last epoch
            if e == 0 or e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                
                # Evaluate the model
                if self.backward:
                    test_loss = self.test_inverse(e, txt, agent_num, mode, jlnet)
                else:
                    test_loss = self.test(e, txt)
            
            # Reset train loss for the next epoch
            train_loss = 0.0
        
        # Increment epoch counter
        self.epo += 1
    
        # Return uncertainty if `sign` is 'inverse'
        if sign == 'inverse':
            return uncertainty


    def test_inverse(self, e, txt, agent_num=None, mode="centralized", jlnet=0):
        test_loss = 0.0
        
        # Load the validation dataset corresponding to the specified agent_num
        if mode == "decentralized" or mode == "jlgat":
            dataset_path = f'collected/esim_test_full_agent_{agent_num}.pkl'
        else:
            dataset_path = 'collected/esim_test_full.pkl'
    
        # Load the dataset from the corresponding .pkl file
        if mode == "decentralized" or jlnet == 2 or mode == "centralized":
            test_dataset = PKLDataset(dataset_path)
        elif jlnet == 3:
            test_dataset = PKLDataset(dataset_path, 'jlg3')
        else:
            test_dataset = PKLDataset(dataset_path, 'jlg')
        
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
        # Ensure the model is on the correct device
        self.model.to(self.DEVICE)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # Move data to the device
                if mode == "decentralized" or jlnet == 2 or mode == "centralized":
                    state, pred_state, y_true = data
                    state = state.to(self.DEVICE, non_blocking=True)
                    pred_state = pred_state.to(self.DEVICE, non_blocking=True)
                    y_true = y_true.to(self.DEVICE, non_blocking=True)
                elif jlnet == 3:
                    state, n_action, pred_state, y_true = data
                    state = state.to(self.DEVICE, non_blocking=True)
                    n_action = n_action.to(self.DEVICE, non_blocking=True)
                    pred_state = pred_state.to(self.DEVICE, non_blocking=True)
                    y_true = y_true.to(self.DEVICE, non_blocking=True)
                else:
                    state, n_state, n_action, pred_state, y_true = data
                    state = state.to(self.DEVICE, non_blocking=True)
                    n_state = n_state.to(self.DEVICE, non_blocking=True)
                    n_action = n_action.to(self.DEVICE, non_blocking=True)
                    pred_state = pred_state.to(self.DEVICE, non_blocking=True)
                    y_true = y_true.to(self.DEVICE, non_blocking=True)
                
                # Forward pass with separate inputs
                if mode == "decentralized" or jlnet == 2 or mode == "centralized":
                    result = self.model(state, pred_state)
                elif jlnet == 3:
                    result = self.model(state, n_action, pred_state)
                else:
                    result = self.model(state, n_state, n_action, pred_state)
        
                y_pred, uncertainty = result[0], result[1]

                # standard loss
                # Forward pass with separate inputs
                if mode == "decentralized":
                    loss = self.criterion(y_pred.squeeze(1), y_true.squeeze().long())
                elif jlnet == 2:
                    loss = self.criterion(y_pred.permute(0, 2, 1), y_true.squeeze(-1).long())
                elif jlnet == 3:
                    loss = self.criterion(y_pred.permute(0, 2, 1), y_true.long())
                elif mode == "centralized":
                    loss = self.criterion(y_pred.permute(0, 2, 1), y_true.squeeze(-1).long())
                else:
                    loss = self.criterion(y_pred, y_true.squeeze().long())
                
                # Accumulate testing loss
                test_loss += loss.item()
        
        # Calculate average test loss
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        
        # Reset the model to training mode
        self.model.train()
        
        return test_loss


    def predict(self, x):
            x = x.to(self.DEVICE)
            with no_grad():
                output = self.model.forward(x)
                result, uncertainty = output[0], output[1]
            return result, uncertainty

    def make_model(self, mode):
        if mode == 'dec' or mode == 'wo_action':
            self.model = Dec_Inverse_N_net(self.ind_state_dim, self.pred_state_dim, self.out_dim, self.backward).float()
        elif mode == 'wo_state':
            self.model = Inverse_N_Net_No_State(self.ind_state_dim, self.action_dim, self.pred_state_dim, self.out_dim, self.backward).float()
        elif mode == 'central':
            self.model = Central_Inverse_N_net(self.ind_state_dim, self.pred_state_dim, self.out_dim, self.backward).float()
        else:
            self.model = Inverse_N_net(self.ind_state_dim, self.n_state_dim, self.action_dim, self.pred_state_dim, self.out_dim, self.backward).float()



class Inverse_N_net(nn.Module):
    def __init__(self, ind_state_dim, n_state_dim, n_action_dim, pred_state_dim, size_out, backward):
        super(Inverse_N_net, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)

        # Shared encoders for each neighbor
        self.n_state_encoder = nn.Linear(n_state_dim[-1], 64)
        self.n_action_encoder = nn.Linear(n_action_dim[-1], 64)
        
        # Encode concatenated neighbor states and actions
        self.neighbor_encoder = nn.Linear(128, 128)

        # Final concatenation: 64 (state) + 64 (pred_state) + 64 (aggregated neighbors) = 192
        self.dense_1 = nn.Linear(256 + 128 * n_state_dim[0], 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)


    def forward(self, state, n_state, n_action, pred_state):
            
        # Infer the number of neighbors dynamically
        num_neighbors = n_state.shape[1]

        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))

        # Encode each neighbor's state and action separately
        n_state_out = F.relu(self.n_state_encoder(n_state))  # Shape: [batch, num_neighbors, 64]
        n_action_out = F.relu(self.n_action_encoder(n_action))  # Shape: [batch, num_neighbors, 64]

        # Concatenate each neighbor's encoded state and action, then process
        neighbor_rep = F.relu(self.neighbor_encoder(torch.cat((n_state_out, n_action_out), dim=2)))  # [batch, num_neighbors, 64]
        
        batch_size = neighbor_rep.shape[0]

        # Concatenate with individual state and predicted state
        x = torch.cat((state_out, pred_state_out, neighbor_rep.view(batch_size, 1, -1)), dim=2)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        logits = self.EDL_layer(x).squeeze(1)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u


class Dec_Inverse_N_net(nn.Module):
    def __init__(self, ind_state_dim, pred_state_dim, size_out, backward):
        super(Dec_Inverse_N_net, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)
        
        # Final concatenation: 128 (state) + 128 (pred_state) = 256
        self.dense_1 = nn.Linear(ind_state_dim[0] * 128 + 128, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)
    
    def forward(self, ind_state, pred_state):
        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(ind_state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))

        # Concatenate encoded states
        x = torch.cat((state_out.view(state_out.shape[0], 1, -1), pred_state_out), dim=-1)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        logits = self.EDL_layer(x)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u



class Inverse_N_Net_No_State(nn.Module):
    def __init__(self, ind_state_dim, n_action_dim, pred_state_dim, size_out, backward):
        super(Inverse_N_Net_No_State, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)
        self.n_action_encoder = nn.Linear(n_action_dim[-1], 128)

        # Encode concatenated neighbor states and actions
        self.state_predAction = nn.Linear(128, 128)
    
        self.dense_1 = nn.Linear(256 + 128 * n_action_dim[0], 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)

    def forward(self, state, n_action, pred_state):
        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))
    
        # Encode each neighbor's action separately
        n_action_out = F.relu(self.n_action_encoder(n_action))
    
        # Concatenate along dimension 1 (feature dimension)
        x = F.relu(self.state_predAction(torch.cat((n_action_out, pred_state_out), dim=1)))
        
        x = x.view(pred_state_out.shape[0], 1, -1)

        x = torch.cat((x, state_out), dim=-1)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        logits = self.EDL_layer(x)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u



class Central_Inverse_N_net(nn.Module):
    def __init__(self, ind_state_dim, pred_state_dim, size_out, backward):
        super(Central_Inverse_N_net, self).__init__()
        self.backward = backward

        # Separate encoders for each input
        self.state_encoder = nn.Linear(ind_state_dim[-1], 128)
        self.pred_state_encoder = nn.Linear(pred_state_dim[-1], 128)

        # Per-agent processing after concatenation
        self.agent_fc = nn.Linear(256, 128)
        
        # Final concatenation: 128 (state) + 128 (pred_state) = 256
        self.dense_1 = nn.Linear(ind_state_dim[0] * 128, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 192)
        self.dense_4 = nn.Linear(int(192 / ind_state_dim[0]), 20)

        # EDL Layer
        self.EDL_layer = nn.Linear(20, size_out)
    
    def forward(self, ind_state, pred_state):
        # state, action shape: (batch, num_agents, state_dim/action_dim)
        batch_size, num_agents = ind_state.shape[:2]
        
        # Encode individual state and predicted state
        state_out = F.relu(self.state_encoder(ind_state))
        pred_state_out = F.relu(self.pred_state_encoder(pred_state))

        # Concatenate state and action per agent
        x = torch.cat((state_out, pred_state_out), dim=-1)  # (batch, num_agents, 256)

        x = F.relu(self.agent_fc(x))

        # Concatenate all states and actions
        x = x.view(batch_size, 1, -1)  # (batch, 1, -1)

        # Forward pass
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        # Reshape output back to per-agent next state
        x = x.view(batch_size, num_agents, -1)  # (batch, num_agents, -1)
        
        x = F.relu(self.dense_4(x))

        logits = self.EDL_layer(x)
        
        # Generate evidence
        evidence = F.relu(logits)

        K = 8
        alpha = evidence + 1
        
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        return logits, u
