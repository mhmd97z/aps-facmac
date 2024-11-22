REGISTRY = {}

from .mlp_agent import MLPAgent
from .rnn_agent import RNNAgent
from .comix_agent import CEMAgent, CEMRecurrentAgent
from .qmix_agent import QMIXRNNAgent, FFAgent
from .gnn_agent import GNNAgent

REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["gnn"] = GNNAgent