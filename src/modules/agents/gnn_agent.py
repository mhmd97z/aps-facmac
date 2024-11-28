import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, LayerNorm, TransformerConv


class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args
        hc = [input_shape, 32, 32]
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

        self.lin = Linear(2*hc[-1], args.n_actions)

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)

    def init_hidden(self):
        return None

    def forward(self, batch, hidden_state, actions=None):
        batch = batch[0]
        # print("batch in gnn forward: ", batch)
        if hasattr(batch['channel'], 'batch'):
            channel_batch = batch['channel'].batch
        else:
            channel_batch = None
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        embedding = []
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            tmp = norm(x_dict['channel'].relu(), channel_batch)
            x_dict = {'channel': tmp}
            embedding.append(tmp)
        embedding = th.cat(embedding, dim=1)
        actions = self.lin(embedding)

        return actions, hidden_state
