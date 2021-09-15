import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import layers
from sampler import ItemToItemBatchSampler, NeighborSampler, PinSAGECollator


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, hidden_dims, n_layers):
        super().__init__()
        self.proj = layers.LinearProjector(full_graph, ntype, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_representation(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_representation(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, args):
    g = dataset['train-graph']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    device = torch.device(args.device)
    
    # sampling
    batch_sampler = ItemToItemBatchSampler(g, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype)
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

    # model
    model = PinSAGEModel(g, item_ntype, args.hidden_dims, args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_list = []

    # train in each batch
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in range(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            loss = model(pos_graph, neg_graph, blocks).mean()
            loss_list.append(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print status
            if batch_id % 500 == 0:
                print("num_epochs:", epoch_id, "||", "batches_per_epoch:", batch_id, "||", "loss:", loss)

        # evaluate
        model.eval()
        with torch.no_grad():
            h_item_batches = []
            for blocks in dataloader_test:
                h_item_batches.append(model.get_representation(blocks))
            h_item = torch.cat(h_item_batches, 0)

    return h_item
