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

def load_and_split_forward_data(pkl_file_path, train_pkl_file, test_pkl_file, spices=8, test_size=0.2, random_seed=42, mode="decentralized", num_agents=1):
    """
    Load data from a .pkl file, process it, convert actions to one-hot encoding using idx2onehot, 
    and split into train-test sets. The train and test sets are stored in separate .pkl files.

    Parameters:
        pkl_file_path (str): Path to the .pkl file.
        train_pkl_file (str): Path to the train data .pkl file.
        test_pkl_file (str): Path to the test data .pkl file.
        spices (int): Number of possible actions (the size of the action space) for one-hot encoding.
        test_size (float): Proportion of the data to include in the test split.
        random_seed (int): Random seed for reproducibility.
        mode (str): "centralized" or "decentralized", determines how to process data.
        num_agents (int): Number of agents in the simulation (used for centralized mode).

    Returns:
        None
    """

    # Debugging: Check input file existence
    if not os.path.exists(pkl_file_path):
        raise FileNotFoundError(f"Error: File {pkl_file_path} not found.")

    # Load data from .pkl file
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    # Initialize lists to store states, actions, and next states
    state_t_list = []
    actions_list = []
    state_t_plus_1_list = []

    if mode == "decentralized":
        # Process records independently for each agent
        for idx, record in enumerate(data):
            states = record[0]  # List of states at time t
            actions = record[1]  # Actions taken
            next_states = record[2]  # States at time t+1

            for j, (state, action, next_state) in enumerate(zip(states, actions, next_states)):
                state_t_list.append(state.flatten())
                state_t_plus_1_list.append(next_state.flatten())

                one_hot_action = idx2onehot(np.array([action]), spices)
                actions_list.append(one_hot_action.flatten())

    elif mode == "centralized":
        # Combine data for all agents
        for idx, record in enumerate(data):
            states = record[0]  # List of states at time t
            actions = record[1]  # Actions taken
            next_states = record[2]  # States at time t+1

            # Flatten and concatenate data for all agents
            combined_state = np.concatenate([state.flatten() for state in states[:num_agents]])
            combined_next_state = np.concatenate([next_state.flatten() for next_state in next_states[:num_agents]])

            one_hot_actions = np.concatenate([
                idx2onehot(np.array([action]), spices).flatten() for action in actions[:num_agents]
            ])

            state_t_list.append(combined_state)
            state_t_plus_1_list.append(combined_next_state)
            actions_list.append(one_hot_actions)

    else:
        raise ValueError("Invalid mode. Please select either 'centralized' or 'decentralized'.")

    # Convert lists to PyTorch tensors
    state_t_tensor = torch.stack([torch.tensor(state, dtype=torch.float32) for state in state_t_list])
    actions_tensor = torch.stack([torch.tensor(action, dtype=torch.float32) for action in actions_list])
    state_t_plus_1_tensor = torch.stack([torch.tensor(next_state, dtype=torch.float32) for next_state in state_t_plus_1_list])

    # Combine inputs (state_t and actions) as features
    features = torch.cat((state_t_tensor, actions_tensor), dim=1)
    targets = state_t_plus_1_tensor

    # Split into train and test sets
    features_train, features_test, targets_train, targets_test = train_test_split(
        features.numpy(), targets.numpy(), test_size=test_size
    )

    # Prepare the data for saving
    train_data = [(features_train[i], targets_train[i]) for i in range(len(features_train))]
    test_data = [(features_test[i], targets_test[i]) for i in range(len(features_test))]

    # Save or append the train and test data to their respective .pkl files
    save_data_to_pkl(train_data, train_pkl_file)
    save_data_to_pkl(test_data, test_pkl_file)


def load_and_split_inverse_data(pkl_file_path, train_pkl_file, test_pkl_file, spices=8, test_size=0.2, random_seed=42, mode="decentralized", num_agents=1):
    """
    Load data from a .pkl file, process it, and split into train-test sets for inverse learning. 
    The features will be the next state and current state, and the target will be the actions. 
    The train and test sets are stored in separate .pkl files.

    Parameters:
        pkl_file_path (str): Path to the .pkl file.
        train_pkl_file (str): Path to the train data .pkl file.
        test_pkl_file (str): Path to the test data .pkl file.
        spices (int): Number of possible actions (the size of the action space) for one-hot encoding.
        test_size (float): Proportion of the data to include in the test split.
        random_seed (int): Random seed for reproducibility.
        mode (str): "centralized" or "decentralized", determines how to process data.
        num_agents (int): Number of agents in the simulation (used for centralized mode).

    Returns:
        None
    """

    # Debugging: Check input file existence
    if not os.path.exists(pkl_file_path):
        raise FileNotFoundError(f"Error: File {pkl_file_path} not found.")

    # Load data from .pkl file
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    # Initialize lists to store states, actions, and next states
    state_t_list = []
    actions_list = []
    state_t_plus_1_list = []

    if mode == "decentralized":
        # Process records independently for each agent
        for idx, record in enumerate(data):
            states = record[0]  # List of states at time t
            actions = record[1]  # Actions taken
            next_states = record[2]  # States at time t+1

            for j, (state, action, next_state) in enumerate(zip(states, actions, next_states)):
                state_t_list.append(state.flatten())
                state_t_plus_1_list.append(next_state.flatten())

                one_hot_action = idx2onehot(np.array([action]), spices)
                actions_list.append(one_hot_action.flatten())

    elif mode == "centralized":
        # Combine data for all agents
        for idx, record in enumerate(data):
            states = record[0]  # List of states at time t
            actions = record[1]  # Actions taken
            next_states = record[2]  # States at time t+1

            # Flatten and concatenate data for all agents
            combined_state = np.concatenate([state.flatten() for state in states[:num_agents]])
            combined_next_state = np.concatenate([next_state.flatten() for next_state in next_states[:num_agents]])

            one_hot_actions = np.concatenate([
                idx2onehot(np.array([action]), spices).flatten() for action in actions[:num_agents]
            ])

            state_t_list.append(combined_state)
            state_t_plus_1_list.append(combined_next_state)
            actions_list.append(one_hot_actions)

    else:
        raise ValueError("Invalid mode. Please select either 'centralized' or 'decentralized'.")

    # Convert lists to PyTorch tensors
    state_t_tensor = torch.stack([torch.tensor(state, dtype=torch.float32) for state in state_t_list])
    actions_tensor = torch.stack([torch.tensor(action, dtype=torch.float32) for action in actions_list])
    state_t_plus_1_tensor = torch.stack([torch.tensor(next_state, dtype=torch.float32) for next_state in state_t_plus_1_list])

    # Combine inputs (state_t and state_t_plus_1) as features
    features = torch.cat((state_t_tensor, state_t_plus_1_tensor), dim=1)
    targets = actions_tensor

    # Split into train and test sets
    features_train, features_test, targets_train, targets_test = train_test_split(
        features.numpy(), targets.numpy(), test_size=test_size
    )

    # Prepare the data for saving
    train_data = [(features_train[i], targets_train[i]) for i in range(len(features_train))]
    test_data = [(features_test[i], targets_test[i]) for i in range(len(features_test))]

    # Save or append the train and test data to their respective .pkl files
    save_data_to_pkl(train_data, train_pkl_file)
    save_data_to_pkl(test_data, test_pkl_file)


class NN_predictor(object):
    def __init__(self, logger, in_dim, out_dim, DEVICE, model_dir, data_dir, backward=False, history=1):
        super(NN_predictor, self).__init__()
        self.epo = 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model =None
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

    def predict(self, x):
        x = x.to(self.DEVICE)
        with no_grad():
            result = self.model.forward(x)
        return result

    def make_model(self):
        self.model = N_net(self.in_dim, self.out_dim, self.backward).float()

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
    
    def train(self, epochs, sign, agent_num=None, max_samples=5000):
        train_loss = 0.0
        
        # Load the full training dataset from the .pkl file
        full_dataset = PKLDataset('collected/ereal_train_full.pkl')
        
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
    
        # Ensure model is on the correct device
        self.model.to(self.DEVICE)
    
        for e in range(epochs):
            record_quantile = []
            for i, data in enumerate(train_loader):
                # Move data to the device
                x, y_true = data
                x = x.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)
                
                # Zero the gradients
                self.optimizer.zero_grad()
    
                # Forward pass
                y_pred = self.model(x)
    
                # Compute the loss
                loss = self.criterion(y_pred, y_true)
                
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
                    test_loss = self.testest_inverset(e, txt)
                else:
                    test_loss = self.test(e, txt)
            
            # Reset train loss for the next epoch
            train_loss = 0.0
    
        # Increment the epoch counter
        self.epo += 1
    
        # Return value if inverse training
        if sign == 'inverse':
            return 0


    def test(self, e, txt):
        test_loss = 0.0
    
        # Load the validation data from the .pkl file
        test_dataset = PKLDataset('collected/ereal_test_full.pkl')
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
        # Ensure model is on the correct device
        self.model.to(self.DEVICE)
    
        # Disable gradient computation during testing
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # Move data to the device
                x, y_true = data
                x = x.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)
    
                # Forward pass
                y_pred = self.model(x)
    
                # Compute the loss
                loss = self.criterion(y_pred, y_true)
                test_loss += loss.item()
    
        # Calculate average test loss
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        
        # Switch the model back to training mode (if needed)
        self.model.train()
        
        return test_loss


# Custom Dataset class to load the .pkl file
class PKLDataset(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)  # Load the (features, targets) list from the .pkl file
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        features, targets = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

class N_net(nn.Module):
    def __init__(self, size_in, size_out, backward):
        super(N_net, self).__init__()
        self.backward = backward

        self.dense_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size_in, 64)
        )
        # self.norm = nn.BatchNorm1d(64)
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)

    def var_torch(self, shape, init=None):
        if init is None:
            data_ini = torch.empty(size=shape)
            std = (2 / shape[0]) ** 0.5
            torch.nn.init.trunc_normal_(data_ini, std=std)  # In-place modification of data_ini
            init = data_ini

        return init

        return init

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x)) #（64， 500）
        x = self.dense_5(x)

        return x


class UNCERTAINTY_predictor(object):
    def __init__(self, logger, in_dim, out_dim, DEVICE, model_dir, data_dir, backward=False, history=1):
        super(UNCERTAINTY_predictor, self).__init__()
        self.epo = 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model =None
        self.backward = backward
        self.make_model()
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

    def train(self, epochs, sign, agent_num=None, max_samples=5000):
        train_loss = 0.0
        
        # Load the full training dataset from the .pkl file
        full_dataset = PKLDataset('collected/esim_train_full.pkl')
        
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
                x, y_true = data
                
                # Move data to the GPU
                x = x.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                result = self.model(x)
                y_pred, uncertainty = result[0], result[1]

                num_agents = y_pred.shape[1] // 8

                y_pred = y_pred.view(y_true.size(0) * num_agents, 8)
                y_true = y_true.view(y_true.size(0), num_agents, 8).argmax(dim=-1)

                y_true = y_true.view(y_true.size(0) * num_agents)
                
                # Compute the loss
                standard_loss = self.criterion(y_pred, y_true)
                
                # Backward pass and optimization
                standard_loss.backward()
                self.optimizer.step()
                
                # Accumulate training loss
                train_loss += standard_loss.item()
            
            # Log progress at the first and last epoch
            if e == 0 or e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                
                # Evaluate the model
                if self.backward:
                    test_loss = self.test_inverse(e, txt)
                else:
                    test_loss = self.test(e, txt)
            
            # Reset train loss for the next epoch
            train_loss = 0.0
        
        # Increment epoch counter
        self.epo += 1
    
        # Return uncertainty if `sign` is 'inverse'
        if sign == 'inverse':
            return uncertainty


    def test_inverse(self, e, txt):
        test_loss = 0.0
        
        # Load the testing data from the .pkl file
        test_dataset = PKLDataset('collected/esim_test_full.pkl')
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
        # Ensure the model is on the correct device
        self.model.to(self.DEVICE)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                
                # Move data to the GPU
                x = x.to(self.DEVICE, non_blocking=True)
                y_true = y_true.to(self.DEVICE, non_blocking=True)
                
                # Forward pass
                result = self.model(x)
                y_pred, uncertainty = result[0], result[1]

                num_agents = y_pred.shape[1] // 8

                y_pred = y_pred.view(y_true.size(0) * num_agents, 8)
                y_true = y_true.view(y_true.size(0), num_agents, 8).argmax(dim=-1)

                y_true = y_true.view(y_true.size(0) * num_agents)
                
                # Compute the loss
                loss = self.criterion(y_pred, y_true)
                
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

    def make_model(self):
        self.model = Inverse_N_net(self.in_dim, self.out_dim, self.backward).float()

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


class Inverse_N_net(nn.Module):
    def __init__(self, size_in, size_out, backward):
        super(Inverse_N_net, self).__init__()
        self.backward = backward

        self.dense_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size_in, 64)
        )
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        
        # Changed output size from 8 to size_out setting to support multiple agents
        self.EDL_layer = nn.Linear(20, size_out, bias=True)

        self.lmb = torch.FloatTensor([0.005])
        
    def var_torch(shape, init=None):
        if init is None:
            data_ini = torch.empty(size=shape)
            std = (2 / shape[0]) ** 0.5
            init = torch.nn.init.trunc_normal_(tensor=data_ini, std=std)

        return init
    

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x)) #（64， 500）

        K = 8 

        W_4_EDL_layer = self.dense_4.weight
        W_end_EDL_layer = self.EDL_layer.weight

        # Added to move calculations to GPU device
        DEVICE = 'cuda:0'
        
        self.lmb = self.lmb.to(DEVICE)

        l2_loss = (self.l2_penalty(W_4_EDL_layer) + self.l2_penalty(W_end_EDL_layer)) * self.lmb

        logits = self.EDL_layer(x)
        evidence = self.relu_evidence(logits)
        alpha = evidence + 1
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty
        
        return logits, u, alpha, l2_loss

    # This function to generate evidence is used for the first example
    def relu_evidence(self, logits):
        relu_net = torch.nn.ReLU()
        return relu_net(logits)

    def l2_penalty(self, w):
        return (w**2).sum() / 2
