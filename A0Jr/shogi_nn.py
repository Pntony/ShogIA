import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mcts_nn import Self_play


torch.set_printoptions(profile="full")


class ConvBlock(nn.Module):

    def __init__(self, nb_in_planes):
        super().__init__()
        # Bias is disabled because it's already managed by batch normalization.
        self.conv = nn.Conv2d(nb_in_planes, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
    
    def forward(self, t):
        t = F.relu(self.bn(self.conv(t)))
        return t


class ResBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
    
    def forward(self, t):
        residual = t
        t = F.relu(self.bn1(self.conv1(t)))
        t = self.bn2(self.conv2(t))
        t = t + residual
        t = F.relu(t)
        return t


class OutBlock(nn.Module):

    def __init__(self):
        super().__init__()

        # Policy head.
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 139, kernel_size=1, stride=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Value head.
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1*9*9, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, t):

        # Policy head.
        log_p = F.relu(self.bn1(self.conv1(t)))
        log_p = self.conv2(log_p)
        log_p = self.logsoftmax(log_p.flatten(start_dim=1)).reshape(log_p.shape)

        # Value head.
        v = F.relu(self.bn2(self.conv3(t)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        v = v.squeeze()

        return log_p, v


class ShogiNNet(nn.Module):

    def __init__(self, nb_res_blocks):
        super().__init__()
        self.nb_res_blocks = nb_res_blocks
        self.convBlock = ConvBlock(44)
        for i in range(nb_res_blocks):
            setattr(self, f"resBlock{i}", ResBlock())
        self.outBlock = OutBlock()
    
    def forward(self, t):
        t = self.convBlock(t)
        for i in range(self.nb_res_blocks):
            t = getattr(self, f"resBlock{i}")(t)
        t = self.outBlock(t)
        return t


# Train the neural network "nnet" with the specified training set with "nb_epochs" epochs
# and the specified hyperparameters.
###
def mini_train(nnet, train_set, params, i):
###

    batch_size = params["batch_size"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    nb_epochs = params["nb_epochs"]

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(nnet.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(nb_epochs):
        for batch in data_loader:

            s, pi, z = batch
            s, pi, z = s.cuda(), pi.cuda(), z.cuda()

            log_p, v = nnet(s)
            loss = ((z - v) ** 2).sum() - (pi * log_p).sum()
            if torch.isnan(loss):
                raise Exception("Infinite value found in the loss.")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ###
        print(f"\niter: {i}/{params['nb_iterations']} ; "\
            f"epoch: {epoch+1}/{nb_epochs} ; " \
            f"nb ended games: {params['ended_games']}")
        ###

def train_nnet(params, nnet=None, tree=None):

    if not nnet:
        nnet = ShogiNNet(params["nb_res_blocks"])
        nnet.cuda()
    for i in range(params["nb_iterations"]):
        if i % params["nb_iter_per_tree_reset"] == 0:
            ###
            ai = Self_play(nnet, params)
            ###
        train_set = ai.build_train_set(params["nb_games_per_iter"], i)
        ###
        mini_train(nnet, train_set, params, i)
        ###
        torch.save(nnet.state_dict(), f"nnet parameters/{params['file_name']}.pt")
        ai.update(nnet)
    
    return nnet