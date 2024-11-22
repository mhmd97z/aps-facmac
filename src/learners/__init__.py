from .cq_learner import CQLearner
from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .maddpg_learner import MADDPGLearner
from .maddpg_learner_discrete import MADDPGDiscreteLearner
from .aps_learner import ApsLearner

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner
REGISTRY["aps_learner"] = ApsLearner
