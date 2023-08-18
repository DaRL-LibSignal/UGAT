import numpy as np
from copy import deepcopy
import torch
from torch import nn, no_grad
import torch.nn.functional as F
from  torch import optim
from torch.utils.data import DataLoader, Dataset
import os
from agent.utils import idx2onehot
import pickle as pkl
import random


def quantile_loss(y_pred, y_true):
    """
    not used, only for previous explore of quantile
    """
    quantiles = [0.025, 0.5, 0.975]
    losses = []
    sigmoid_fun = nn.Sigmoid()
    cross_entropy = nn.CrossEntropyLoss()
    for i, q in enumerate(quantiles):

        ###########v1:
        # errors =  y_true - y_pred
        # losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))
        ###########v2:
        # map the logit into the (0, 1) as probablity
        sigmoid_y_pred = sigmoid_fun(y_pred)
        errors =  cross_entropy(sigmoid_y_pred, y_true)
        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss

def log_passing_lane_actinon(traj, lanes, fix_time=30):
    # TODO: only for one intersection and fixedtime agent now
    # save in {time: {lanes: num}} format
    lanes_dict = {l:0 for l in lanes}
    record = {i: deepcopy(lanes_dict) for i in range(120)}
    # preprocess time
    for k in traj.keys():
        route = traj[k]
        tmp_road = route[0][0]
        tmp_interval =  (route[0][1]+route[0][2]-1) // fix_time # -1 for integer result, 5 sec yellow ensures this 
        record[tmp_interval][tmp_road] += 1 
    return record

def write_action_record(path, record, struc, fix_time=30):
    result = []
    result.append(str(struc))
    for t in range(int(3600/ fix_time)):
        tmp = []
        for r in struc:
            tmp.append(str(record[t][r]).ljust(5, ' '))
        result.append(str(tmp).replace(",", "").replace("'", ""))
    # temp = str([str(int(i)).ljust(5, ' ') for i in obs[idx][0]])
    with open(file=path, mode='w+', encoding='utf-8') as wf:
        for line in result:
            wf.writelines(line + "\n")

class NN_dataset(Dataset):
    def __init__(self, feature, target):
        self.len = len(feature)
        self.features = torch.from_numpy(feature).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, idx):
        return self.features[idx, :], self.target[idx]

    def __len__(self):
        return self.len

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
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        # self.online_optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate * 0.1, momentum=0.9)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.online_optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logger = logger

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def load_dataset(self):
        train_data = generate_forward_dataset(self.data_dir, backward=self.backward, history=self.history)
        # train val split
        split_point = int(train_data['x_train'].shape[0] * 0.8)
        # shuffled when create
        if self.x_train is not None:
            self.x_train = np.concatenate((self.x_train, train_data['x_train'][: split_point]))
            self.y_train = np.concatenate((self.y_train, train_data['y_train'][: split_point]))
            self.x_val = np.concatenate((self.x_val, train_data['x_train'][split_point :]))
            self.y_val = np.concatenate((self.y_val, train_data['y_train'][split_point :]))     
        else:       
            self.x_train = train_data['x_train'][: split_point]
            self.y_train = train_data['y_train'][: split_point]
            self.x_val = train_data['x_train'][split_point :]
            self.y_val = train_data['y_train'][split_point :]
        
        # shuffle in dataloader
        print('dataset batch size: ', self.y_train.shape[0])


    def predict(self, x):
        x = x.to(self.DEVICE)
        with no_grad():
            output = self.model.forward(x)
            result, uncertainty = output[0], output[1]
        return result, uncertainty

    def train(self, epochs, writer, sign):
        train_loss = 0.0
        train_dataset = NN_dataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        print(f"Epoch {self.epo - 1} Training")
        # epoch_quantiles = []
        for e in range(epochs):
            for i, data in enumerate(train_loader):
                x, y_true = data
                self.optimizer.zero_grad()
                x.to(self.DEVICE)
                y_true.to(self.DEVICE)
                result  = self.model(x)
                y_pred, uncertainty = result[0], result[1]

                # get l2_regularier_loss
                alpha_val, l2_loss = result[2], result[3]

                # standard loss
                standard_loss = self.criterion(y_pred, y_true)

                # classic_loss
                generated_loss = loss_v2(y_pred, y_true)

                # ablation study:
                loss = standard_loss + generated_loss + l2_loss  # v_original

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
 
            if e == 0:
                ave_loss = train_loss/ len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                test_loss = self.test(e, txt)

                if sign == 'inverse':
                    writer.add_scalar("ave_train_Loss/start_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_inverse_test", test_loss, self.epo)
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=0, stop=(self.epo+1) * len(epoch_quantiles)))
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/start_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_forward_test", test_loss, self.epo)

            elif e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                test_loss = self.test_inverse(e, txt)

                if sign == 'inverse':
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=(self.epo) * len(epoch_quantiles), stop=(self.epo +1)* len(epoch_quantiles)))
                    writer.add_scalar("ave_train_Loss/end_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_inverse_test", test_loss, self.epo)
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/end_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_forward_test", test_loss, self.epo)

            train_loss = 0.0
        self.epo += 1
        
        if sign == 'inverse':
            # print("uncertainty now is:"+str(uncertainty))
            # return the uncertainty if inverse:
            return uncertainty

    
    def test(self, e, txt):
        test_loss = 0.0
        test_dataset = NN_dataset(self.x_val, self.y_val)
        test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)
        with no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                x.to(self.DEVICE)
                y_true.to(self.DEVICE)
                result = self.model(x)
                y_pred, uncertainty = result[0], result[1]
                # print(y_pred)
                # print(y_true)
                # normal loss, different from the training part
                loss = self.criterion(y_pred, y_true)
                test_loss += loss.item()
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        return test_loss
    
    def test_inverse(self, e, txt):
        test_loss = 0.0
        test_dataset = NN_dataset(self.x_val, self.y_val)
        test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)
        with no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                x.to(self.DEVICE)
                y_true.to(self.DEVICE)
                result = self.model(x)
                y_pred, uncertainty = result[0], result[1]

                # get l2_regularier_loss
                alpha_val, l2_loss = result[2], result[3]

                # standard loss
                standard_loss = self.criterion(y_pred, y_true)

                # classic_loss
                generated_loss = loss_v2(y_pred, y_true)
                loss = standard_loss + generated_loss + l2_loss

                test_loss += loss.item()
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        return test_loss
    
    def make_model(self):
        # v1:
        # self.model = N_net(self.in_dim, self.out_dim, self.backward).float()
        # v2: add loss to optimze
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
        torch.save(self.m34del.state_dict(), model_name)

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

    def load_dataset(self):
        train_data = generate_forward_dataset(self.data_dir, backward=self.backward, history=self.history)
        # train val split
        split_point = int(train_data['x_train'].shape[0] * 0.8)
        # shuffled when create
        if self.x_train is not None:
            self.x_train = np.concatenate((self.x_train, train_data['x_train'][: split_point]))
            self.y_train = np.concatenate((self.y_train, train_data['y_train'][: split_point]))
            self.x_val = np.concatenate((self.x_val, train_data['x_train'][split_point :]))
            self.y_val = np.concatenate((self.y_val, train_data['y_train'][split_point :]))     
        else:       
            self.x_train = train_data['x_train'][: split_point]
            self.y_train = train_data['y_train'][: split_point]
            self.x_val = train_data['x_train'][split_point :]
            self.y_val = train_data['y_train'][split_point :]
        
        # shuffle in dataloader
        print('dataset batch size: ', self.y_train.shape[0])


    def predict(self, x):
        x = x.to(self.DEVICE)
        with no_grad():
            result = self.model.forward(x)
        return result

    def train(self, epochs, writer, sign):
        train_loss = 0.0
        train_dataset = NN_dataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        print(f"Epoch {self.epo - 1} Training")
        # epoch_quantiles = []
        for e in range(epochs):
            record_quantile = []
            for i, data in enumerate(train_loader):
                x, y_true = data
                self.optimizer.zero_grad()
                x.to(self.DEVICE)
                y_true.to(self.DEVICE)
                result = self.model(x)
                y_pred, u = result[0], result[1]
                loss = self.criterion(y_pred, y_true)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if e == 0:
                ave_loss = train_loss/ len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                if self.backward:
                    test_loss = self.testest_inverset(e, txt)
                else:
                    test_loss = self.test(e, txt)

                if sign == 'inverse':
                    writer.add_scalar("ave_train_Loss/start_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_inverse_test", test_loss, self.epo)
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=0, stop=(self.epo+1) * len(epoch_quantiles)))
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/start_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_forward_test", test_loss, self.epo)

            elif e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                test_loss = self.test(e, txt)

                if sign == 'inverse':
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=(self.epo) * len(epoch_quantiles), stop=(self.epo +1)* len(epoch_quantiles)))
                    writer.add_scalar("ave_train_Loss/end_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_inverse_test", test_loss, self.epo)
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/end_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_forward_test", test_loss, self.epo)

            train_loss = 0.0
        self.epo += 1
        
        if sign == 'inverse':
            # return the uncertainty if inverse:
            return 0
        # import matplotlib.pyplot as plt
        # plt.scatter(np.arange(len(epoch_quantiles)), epoch_quantiles)
        # plt.show()

    def test(self, e, txt):
        test_loss = 0.0
        test_dataset = NN_dataset(self.x_val, self.y_val)
        test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)
        with no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                x.to(self.DEVICE)
                y_true.to(self.DEVICE)
                y_pred, uncertainty = self.model(x)
                loss = self.criterion(y_pred, y_true)
                test_loss += loss.item()
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        return test_loss
    


    def make_model(self):
        self.model = N_net(self.in_dim, self.out_dim, self.backward).float()
        # self.model = LSTMPredictor(self.in_dim, self.out_dim, self.backward).float()

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
        torch.save(self.m34del.state_dict(), model_name)

##############################################

"""
LSTM based model:
"""


class LSTMPredictor(nn.Module):
    def __init__(self, size_in, size_out, backward, n_hidden=64):
        super(LSTMPredictor, self).__init__()
        self.size_in = size_in
        self.hidden = n_hidden
        self.lstm1 = nn.LSTMCell(size_in, self.hidden)  # input, x
        self.lstm2 = nn.LSTMCell(self.hidden, self.hidden)
        self.dense_4 = nn.Linear(self.hidden, 128)
        self.dense_5 = nn.Linear(128, 20)
        self.linear = nn.Linear(20, size_out)  # output: y

    def forward(self, x, future=0):

        global output
        outputs = []
        n_samples = x.size(0)
        h_t = torch.zeros(n_samples, self.hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden, dtype=torch.float32)

        h_t2 = torch.zeros(n_samples, self.hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden, dtype=torch.float32)

        # for input_t in x.split(1, dim=1):
        #     # N, 1
        h_t, c_t = self.lstm1(x, (h_t, c_t))
        h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        output = self.dense_4(h_t2)  # input the hidden state, and output the prediction
        output = self.dense_5(output)  # input the hidden state, and output the prediction
        output = self.linear(output)  # input the hidden state, and output the prediction

        # outputs = torch.cat(outputs, dim=1)
        return output

#################################
"""
N_net(self.in_dim, self.out_dim, self.backward).float()
the alpha prediction
"""
class Alpha_net(nn.Module):
    def __init__(self):
        super(Alpha_net, self).__init__()
        self.dense_1 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(1, 8)
        )
        # self.norm = nn.BatchNorm1d(64)
        self.dense_2 = nn.Linear(8, 32)
        self.dense_3 = nn.Linear(32, 64)
        self.dense_4 = nn.Linear(64, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):

        direct_x = self.out(x.clone())

        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.out(x)
        # if self.backward:
        #     x = F.softmax(x, dim=1)
        return direct_x, x


#################################
"""
origin N_net based model:
"""

# This function to generate evidence is used for the first example
def relu_evidence(logits):
    relu_net = torch.nn.ReLU()
    return relu_net(logits)


# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits):
    return torch.exp(torch.clip(logits, -10, 10))


# This one is another alternative and
# usually behaves better than the relu_evidence
def softplus_evidence(logits):
    softplus_net = torch.nn.Softplus()
    return softplus_net(logits)

def var_torch(shape, init=None):
    if init is None:
        data_ini = torch.empty(size=shape)
        std = (2 / shape[0]) ** 0.5
        init = torch.nn.init.trunc_normal_(tensor=data_ini, std=std)

    return init

def KL(alpha):
    # beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    K = 8
    beta = torch.ones((1, K))
    # S_alpha = tf.reduce_sum(alpha, axis=1, keep_dims=True)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    # S_beta = tf.reduce_sum(beta, axis=1, keep_dims=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    # lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keep_dims=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def loss_EDL(p, alpha, global_step, annealing_step):
    # annealing_step3:10*n_batches
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    A = torch.sum(p * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = torch.Tensor(np.array(np.min([1.0, global_step / annealing_step])))

    alp = E * (1 - p) + 1
    B = annealing_coef * KL(alp)
    result = A + B
    print(result)
    return (result)

def loss_v2(logits, labels):
    loss_k = torch.mean(-torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1))
    return loss_k

def l2_penalty(w):
    return (w**2).sum() / 2


class Inverse_N_net(nn.Module):
    def __init__(self, size_in, size_out, backward):
        super(Inverse_N_net, self).__init__()
        self.backward = backward

        self.dense_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size_in, 64)
        )
        # self.norm = nn.BatchNorm1d(64)
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        # self.dense_5 = nn.Linear(20, size_out)
        self.EDL_layer = nn.Linear(20, 8, bias=True)

        self.lmb = torch.FloatTensor([0.005])
        # self.dense_4 = nn.Linear(128, 500)
        # self.dense_5 = nn.Linear(20, size_out)
    def var_torch(shape, init=None):
        if init is None:
            data_ini = torch.empty(size=shape)
            std = (2 / shape[0]) ** 0.5
            init = torch.nn.init.trunc_normal_(tensor=data_ini, std=std)

        return init
    

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        # x = self.norm(x)
        # x = F.batch_norm(x)
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x)) #（64， 500）

        K = 8 

        W_4_EDL_layer = self.dense_4.weight
        W_end_EDL_layer = self.EDL_layer.weight
        # B_EDL_layer = self.EDL_layer.bias
        l2_loss = (l2_penalty(W_4_EDL_layer)+l2_penalty(W_end_EDL_layer)) * self.lmb

        logits = self.EDL_layer(x)
        evidence = relu_evidence(logits)
        alpha = evidence + 1
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty
        
        return logits, u, alpha, l2_loss




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
        # self.dense_4 = nn.Linear(128, 500)
        # self.dense_5 = nn.Linear(20, size_out)
    def var_torch(shape, init=None):
        if init is None:
            data_ini = torch.empty(size=shape)
            std = (2 / shape[0]) ** 0.5
            init = torch.nn.init.trunc_normal_(tensor=data_ini, std=std)

        return init
    

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        # x = self.norm(x)
        # x = F.batch_norm(x)
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x)) #（64， 500）

        # global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        

        # uncertainty:
        K = 8 # how many classes in there?
        W4 = var_torch([20, K])
        b4 = var_torch([K])
        out3 = torch.clone(x)
        logits = torch.matmul(out3, W4) + b4
        evidence = relu_evidence(logits)
        alpha = evidence + 1
        u = K / torch.sum(alpha, dim=1, keepdim=True)  # uncertainty

        # loss = torch.sum(loss_EDL(Y, alpha, global_step, annealing_step))

        x = self.dense_5(x)

        # if self.backward:
        #     x = F.softmax(x, dim=1)
        # print(u)
        return x, u

def generate_forward_dataset(file, action=8, backward=False, history=1):
    with open(file, 'rb') as f:
        contents = pkl.load(f)

    feature = list()
    target = list()
    if backward:
        assert history == 1
        for e in range(contents[0].__len__()):
            for s in range(360):
                x = np.concatenate((contents[0][e][s], contents[0][e][s+1]), axis=1)
                feature.append(x)
                y = idx2onehot(contents[1][e][s], action)
                target.append(y)
    else:
        unit_size = np.concatenate((contents[0][0][0], idx2onehot(contents[1][0][0][0], action)), axis=1).shape[1]
        input_size = history * unit_size
        for e in range(contents[0].__len__()):
            seq = np.zeros((1, input_size))
            for s in range(360):
                x = np.concatenate((contents[0][e][s], idx2onehot(contents[1][e][s][0], action)), axis=1)
                seq = np.concatenate((seq, x), axis=1)
                feature.append(seq[:, -input_size :])
                y = contents[0][e][s+1]
                target.append(y)

    feature= np.concatenate(feature, axis=0)
    target = np.concatenate(target, axis=0)
    total_idx = len(target)
    sample_idx = range(total_idx)
    sample_idx = random.sample(sample_idx, len(sample_idx))
    x_train = feature[sample_idx]
    y_train = target[sample_idx]
    dataset = {'x_train': x_train, 'y_train': y_train}
    return dataset

# def generate_forward_dataset(file, action=8, backward = False):
#     with open(file, 'rb') as f:
#         contents = pkl.load(f)

#     feature = list()
#     target = list()
#     if backward:
#         for e in range(contents[0].__len__()):
#             for s in range(360, 0, -1):
#                 x = np.concatenate((contents[0][e][s], idx2onehot(contents[1][e][s-1][0], action)), axis=1)
#                 feature.append(x)
#                 y = contents[0][e][s-1]
#                 target.append(y)
#     else:
#         for e in range(contents[0].__len__()):
#             for s in range(360):
#                 x = np.concatenate((contents[0][e][s], idx2onehot(contents[1][e][s][0], action)), axis=1)
#                 feature.append(x)
#                 y = contents[0][e][s+1]
#                 target.append(y)

#     feature= np.concatenate(feature, axis=0)
#     target = np.concatenate(target, axis=0)
#     total_idx = len(target)
#     sample_idx = range(total_idx)
#     sample_idx = random.sample(sample_idx, len(sample_idx))
#     x_train = feature[sample_idx[: int(0.8 * total_idx)]]
#     y_train = target[sample_idx[: int(0.8 * total_idx)]]
#     x_test = feature[sample_idx[int(0.8 * total_idx) :]]
#     y_test = target[sample_idx[int(0.8 * total_idx) :]]
#     dataset = {'x_train': x_train, 'y_train': y_train, 'x_val': x_test, 'y_val': y_test}
#     return dataset
