import numpy as np
import torch
import torch.nn as nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dModel
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              nn.init.calculate_gain('relu'))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AtariNatureEncoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size, output_size):
        super().__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels
        
        init_ = lambda m: init(m,
                  nn.init.orthogonal_,
                  lambda x: nn.init.constant_(x, 0),
                  nn.init.calculate_gain('relu'))
        
        self.conv1 = init_(nn.Conv2d(img_channels, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32*4*4, latent_size))
        self.fc2 = init_(nn.Linear(latent_size, output_size))


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class AtariQofMuEncoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, action_size, latent_size, action_latent_size, output_size):
        super().__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels
        
        init_ = lambda m: init(m,
                  nn.init.orthogonal_,
                  lambda x: nn.init.constant_(x, 0),
                  nn.init.calculate_gain('relu'))
        
        self.conv1 = init_(nn.Conv2d(img_channels, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        
        self.action_fc1 = init_(nn.Linear(action_size, action_latent_size))
        self.action_fc2 = init_(nn.Linear(action_latent_size, action_latent_size))
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32*4*4 + action_latent_size, latent_size))
        self.fc2 = init_(nn.Linear(latent_size, output_size))


    def forward(self, obs, action): # pylint: disable=arguments-differ
        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))
        obs = self.flatten(obs)

        action_latent = F.relu(self.action_fc1(action))
        action_latent = F.relu(self.action_fc2(action_latent))

        x = torch.cat([obs, action_latent], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MuConv2dModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape, 
            img_channels,
            latent_size,
            action_size,
            output_max=1,
            ):
        super().__init__()
        self._output_max = output_max
        #self._obs_ndim = len(observation_shape.observation)
        print(observation_shape)
        self._obs_ndim = len(observation_shape.observation)
        self.model = AtariNatureEncoder(img_channels, latent_size, action_size)

    def forward(self, observation, prev_action, prev_reward):
        observation = observation.observation
        lead_dim, T, B, shape = infer_leading_dims(observation, self._obs_ndim)
        mu = self._output_max * torch.tanh(self.model(observation.view(T * B, *shape)))
        mu = restore_leading_dims(mu, lead_dim, T, B)
        return mu


class PiConv2dModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.model = AtariNatureEncoder(img_channels, latent_size, action_size*2)

        # self.mlp = MlpModel(
        #     input_size=int(np.prod(observation_shape)),
        #     hidden_sizes=hidden_sizes,
        #     output_size=action_size * 2,
        # )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        mu, log_std = torch.chunk(output, 2, dim=-1)
        # output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class QofMuConv2dModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape, 
            img_channels,
            latent_size,
            action_latent_size,
            action_size
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape.observation)
        self.model = AtariQofMuEncoder(img_channels, action_size, latent_size, action_latent_size, output_size=1)

    def forward(self, observation, prev_action, prev_reward, action):
        lead_dim, T, B, shape = infer_leading_dims(observation,
            self._obs_ndim)

        q = self.model(
                observation.view(T * B, *shape),
                action.view(T * B, -1))
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class VMlpModel(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            img_channels,
            latent_size,
            action_size=None,  # Unused but accept kwarg.
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape.observation)
        self.model = AtariNatureEncoder(img_channels, latent_size, 1)
        # self.mlp = MlpModel(
        #     input_size=int(np.prod(observation_shape)),
        #     hidden_sizes=hidden_sizes,
        #     output_size=1,
        # )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        v = self.model(observation.view(T * B, *shape))
        v = restore_leading_dims(v, lead_dim, T, B)
        return v
