import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False


def _init_input_modules(g, ntype, hidden_dims):
    module_dict = nn.ModuleDict()

    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m

    return module_dict


class LinearProjector(nn.Module):
    def __init__(self, full_graph, ntype, hidden_dims):
        super().__init__()
        self.ntype = ntype
        self.inputs = _init_input_modules(full_graph, ntype, hidden_dims)

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith('__len'):
                continue
            module = self.inputs[feature]
            result = module(data)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)


class GATLayer(nn.Module):
    def __init__(self, input_dims):
        super(GATLayer, self).__init__()
        self.additive_attn_fc = nn.Linear(3 * input_dims, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.additive_attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        x = torch.cat([edges.src['z_src'], edges.dst['z_t'], edges.data['z_c']], dim=1)
        attention = self.additive_attn_fc(x)
        return {'attn': F.leaky_relu(attention)}

    def forward(self, block):
        block.apply_edges(self.edge_attention)
        attention = edge_softmax(block, block.edata['attn'])
        return attention


class MultiHeadGATLayer(nn.Module):
    def __init__(self, input_dims, num_heads, merge='mean'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(input_dims))
        self.merge = merge

    def forward(self, block):
        head_outs = [attn_head(block) for attn_head in self.heads]
        if self.merge == 'mean':
            return torch.mean(torch.stack(head_outs), 0)
        else:  # concatenate
            return torch.cat(head_outs, dim=0)


class MultiSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, gat_num_heads, act=F.relu):
        super().__init__()
        self.multi_head_gat_layer = MultiHeadGATLayer(input_dims, gat_num_heads)
        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.dropout = nn.Dropout(0.5)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def _transfer_raw_input(self, edges):
        return {'z_src_c': torch.mul(edges.src['z_src'], edges.data['z_c']),
                'z_t_c': torch.mul(edges.dst['z_t'], edges.data['z_c'])}

    def _node_integration(self, edges):
        return {'neighbors': edges.data['z_src_c'] * edges.data['a_mean'],
                'targets': edges.data['z_t_c'] * edges.data['a_mean']}

    def forward(self, block, h, context_node, attn_index=None):
        h_src, h_dst = h
        with block.local_scope():
            # transfer raw input feature
            z_src = self.act(self.Q(self.dropout(h_src)))
            z_c = self.act(context_node)
            block.srcdata['z_src'] = z_src
            block.dstdata['z_t'] = h_dst
            block.edata['z_c'] = z_c

            # getting attention
            attention = self.multi_head_gat_layer(block)
            if attn_index is not None:  # attn_index : index of attention which not in context id
                attention[attn_index] = 0
            block.edata['a_mean'] = attention

            # aggregation
            block.apply_edges(self._transfer_raw_input)
            block.apply_edges(self._node_integration)
            block.update_all(fn.copy_e('neighbors', 'm'), fn.sum('m', 'ns'))
            block.update_all(fn.copy_e('targets', 'm'), fn.sum('m', 'ts'))

            # normalize for context query
            if attn_index is not None:
                neighbor = block.dstdata['ns'] / (attention.shape[0] - sum(attn_index).item())
                target = block.dstdata['ts'] / (attention.shape[0] - sum(attn_index).item())
            else:
                neighbor = block.dstdata['ns'] / attention.shape[0]
                target = block.dstdata['ts'] / attention.shape[0]

            # normalize
            z = self.act(self.W(self.dropout(torch.cat([neighbor, target], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


class MultiSAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers, gat_num_heads):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(MultiSAGEConv(hidden_dims, hidden_dims, hidden_dims, gat_num_heads))

    def forward(self, blocks, h, context_blocks, attn_index=None):
        for idx, (layer, block, context_node) in enumerate(zip(self.convs, blocks, context_blocks)):
            if (attn_index is not None) and (idx == 1):
                h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
                h = layer(block, (h, h_dst), context_node, attn_index)
            else:
                h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
                h = layer(block, (h, h_dst), context_node)
        return h


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()
        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score
