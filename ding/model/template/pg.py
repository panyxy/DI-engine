from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn
from easydict import EasyDict

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ..common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, \
        MultiHead, RegressionHead, ReparameterizationHead



@MODEL_REGISTRY.register('pg')
class PolicyGradient(nn.Module):
    r"""
    Overview:
        The Policy Gradient network.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType, EasyDict],
            action_space: str,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        """
        Overview:
            Initailize the ContinuousBC Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's shape, such as 128, (156, ).
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's shape, such as 4, (3, ), \
                EasyDict({'action_type_shape': 3, 'action_args_shape': 4}).
            - action_space (:obj:`str`): The type of action space, \
                including [``regression``, ``reparameterization``].
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor head.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for actor head.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` \
                after each FC layer, if ``None`` then default set to ``nn.ReLU()``.
            - norm_type (:obj:`Optional[str]`): The type of normalization to after network layer (FC, Conv), \
                see ``ding.torch_utils.network`` for more details.
        """
        super(PolicyGradient, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization']
        if self.action_space == 'regression':
            self.actor = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                RegressionHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.action_space == 'reparameterization':
            self.actor = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                ReparameterizationHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    sigma_type='conditioned',
                    activation=activation,
                    norm_type=norm_type
                )
            )

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            The unique execution (forward) method of ContinuousBC method.
            Arguments:
                - inputs (:obj:`torch.Tensor`): Observation data, defaults to tensor.
            Returns:
                - output (:obj:`Dict`): Output dict data, including differnet key-values among distinct action_space.
        """
        if self.action_space == 'regression':
            x = self.actor(inputs)
            return {'action': x['pred']}

        elif self.action_space == 'reparameterization':
            x = self.actor(inputs)
            return {'logit': x}
