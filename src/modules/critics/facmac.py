import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, LayerNorm, TransformerConv
from utils.graph_storage import GraphBatchStorage

class FACMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


class FACMACDiscreteCriticGNN(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACDiscreteCriticGNN, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        hc = [self.input_shape, 128, 128]
        num_layers = len(hc)
        heads = 2
        aggr = 'sum'

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = hc[i]
            out_channels = int(hc[i+1] / heads)
            conv = HeteroConv({
                ('channel', 'same_ue', 'channel'):
                TransformerConv(in_channels, out_channels,
                                heads=heads, dropout=0.0, root_weight=True, concat=True),
                ('channel', 'same_ap', 'channel'):
                TransformerConv(in_channels, out_channels,
                                heads=heads, dropout=0.0, root_weight=True, concat=True)
            }, aggr=aggr)
            self.convs.append(conv)
            self.norms.append(LayerNorm(hc[i+1]))

        self.lin = Linear(hc[-1], 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def concat_action_to_graph(self, inputs, actions):
        # actions: batch*steps*agents*actions
        # inputs: batch*steps*graphs
        inputs = inputs.clone()
        for batch in range(actions.shape[0]):
            for step in range(actions.shape[1]):
                inputs[batch, step]['channel']['x'] = th.cat(
                    (inputs[batch, step]['channel']['x'], 
                     actions[batch][step]), 
                    dim=1)
        return inputs

    def forward(self, batch, actions=None, hidden_state=None):
        if actions is not None:
            batch = self.concat_action_to_graph(batch, actions)
    
        if isinstance(batch, GraphBatchStorage):
            out = []
            for b in range(actions.shape[0]):
                this_batch = batch[b, :]
                # print("this_batch: ", this_batch.batch_dict['channel'].shape)
                batch_q, _ = self(this_batch)
                out.append(batch_q.reshape((actions.shape[1], actions.shape[2])))
            return th.stack(out), None

        if hasattr(batch['channel'], 'batch'):
            channel_batch = batch['channel'].batch
        else:
            channel_batch = None
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {'channel': norm(x_dict['channel'].relu(), channel_batch)}
        q = self.lin(x_dict['channel'])

        return q, hidden_state
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape
