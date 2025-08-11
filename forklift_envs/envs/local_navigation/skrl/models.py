import torch
import torch.nn as nn
from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin

from forklift_envs.learning.models import MODEL_REGISTRY, get_activation, register_model

def get_activation(activation_name):
    """Get the activation function by name."""
    activation_fns = {
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if activation_name not in activation_fns:
        raise ValueError(f"Activation function {activation_name} not supported.")
    return activation_fns[activation_name]

@register_model("GaussianPolicyConv")
class GaussianNeuralNetwork(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        print("Gaussian Neural Network Initialization")

        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        # self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        # if self.encoder_input_size is not None:
        #     self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
        #     in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # Split the states into proprioception and heightmap if the heightmap is used.
        # print(">>> Critic receives states['states'] of shape:", states["states"].shape)
        print(states["states"][:, :self.mlp_input_size].shape)
        x = states["states"][:, :self.mlp_input_size]
        # if self.encoder_input_size is None:
        #     x = states["states"]
        # else:
        #     encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
        #     x = states["states"][:, 0:self.mlp_input_size]
        #     x = torch.cat([x, encoder_output], dim=1)
        
        # print("[Check] Guassian Neural Network Input(state): ", x)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}

@register_model("ValueNetworkConv")
class DeterministicNeuralNetwork(DeterministicMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=3,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        # self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        # if self.encoder_input_size is not None:
        #     self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
        #     in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        # if self.encoder_input_size is None:
        #     x = states["states"]
        # else:
        #     x = states["states"][:, :self.mlp_input_size]
        #     encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:-1])
        #     x = torch.cat([x, encoder_output], dim=1)
        x = states["states"][:, :self.mlp_input_size]

        for layer in self.mlp:
            x = layer(x)

        return x, {}