import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import layers
from sampler import ItemToItemBatchSampler, NeighborSampler, PinSAGECollator


class MultiSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, ctype, hidden_dims, n_layers, gat_num_heads):
        super().__init__()
        self.nodeproj = layers.LinearProjector(full_graph, ntype, hidden_dims)
        self.contextproj = layers.LinearProjector(full_graph, ctype, hidden_dims)
        self.multisage = layers.MultiSAGENet(hidden_dims, n_layers, gat_num_heads)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks, context_blocks):
        h_item = self.get_representation(blocks, context_blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_representation(self, blocks, context_blocks, context_id=None):
        if context_id:
            return self.get_context_query(blocks, context_blocks, context_id)
        else:
            h_item = self.nodeproj(blocks[0].srcdata)
            h_item_dst = self.nodeproj(blocks[-1].dstdata)
            z_c = self.contextproj(context_blocks[0])
            z_c_dst = self.contextproj(context_blocks[-1])
            h = h_item_dst + self.multisage(blocks, h_item, (z_c, z_c_dst))
            return h

    def get_context_query(self, blocks, context_blocks, context_id):
        # check sub-graph contains context id
        context_id = context_blocks[-1]['_ID'][0].item()
        print(context_id)
        print(context_blocks[-1]['_ID'])
        context_index = (context_id == context_blocks[-1]['_ID']).nonzero(as_tuple=True)[0]
        if context_index.size()[0] == 0:  # if context id not in sub-graph, only random sample context using for repr
            print("context not in sub graph")
            return self.get_representation(blocks, context_blocks)
        else:  # if context id in sub-graph, get MultiSAGE's context query
            print("execute context query")
            attn_index = torch.ones(context_blocks[-1]['_ID'].shape[0], dtype=bool)
            attn_index[context_index] = False
            h_item = self.nodeproj(blocks[0].srcdata)
            h_item_dst = self.nodeproj(blocks[-1].dstdata)
            z_c = self.contextproj(context_blocks[0])
            z_c_dst = self.contextproj(context_blocks[-1])
            h = h_item_dst + self.multisage(blocks, h_item, (z_c, z_c_dst), attn_index)
            return h


def train(dataset, args):
    g = dataset['train-graph']
    context_ntype = dataset['context-type']
    item_ntype = dataset['item-type']
    device = torch.device(args.device)
    batch_sampler = ItemToItemBatchSampler(g, context_ntype, item_ntype, args.batch_size)
    neighbor_sampler = NeighborSampler(
        g, context_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype, context_ntype)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)

    # Model
    model = MultiSAGEModel(g, item_ntype, context_ntype, args.hidden_dims, args.num_layers, args.gat_num_heads).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_list = []

    # For each batch of head-tail-negative triplets
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in range(args.batches_per_epoch):
            pos_graph, neg_graph, blocks, context_blocks = next(dataloader_it)
            loss = model(pos_graph, neg_graph, blocks, context_blocks).mean()
            loss_list.append(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print status
            if batch_id % 10 == 0:
                print("num_epochs:", epoch_id, "||", "batches_per_epoch:", batch_id, "||", "loss:", loss)

    # Evaluate
    model.eval()
    with torch.no_grad():
        h_item_batches = []
        for blocks, context_blocks in dataloader_test:
            h_item_batch = model.get_representation(blocks, context_blocks)
            h_item_batches.append(h_item_batch)
        h_item = torch.cat(h_item_batches, 0)

    return model, h_item
