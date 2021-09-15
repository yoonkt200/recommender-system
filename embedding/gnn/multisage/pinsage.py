from collections import Counter

import numpy as np

from .. import backend as F
from .. import convert
from .. import transform
from .randomwalks import random_walk
from .neighbor import select_topk
from ..base import EID
from .. import utils


class RandomWalkNeighborSampler(object):
    def __init__(self, G, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, metapath=None, weight_column='weights'):
        assert G.device == F.cpu(), "Graph must be on CPU."
        self.G = G
        self.weight_column = weight_column
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals

        if metapath is None:
            if len(G.ntypes) > 1 or len(G.etypes) > 1:
                raise ValueError('Metapath must be specified if the graph is homogeneous.')
            metapath = [G.canonical_etypes[0]]
        start_ntype = G.to_canonical_etype(metapath[0])[0]
        end_ntype = G.to_canonical_etype(metapath[-1])[-1]
        if start_ntype != end_ntype:
            raise ValueError('The metapath must start and end at the same node type.')
        self.ntype = start_ntype

        self.metapath_hops = len(metapath)
        self.metapath = metapath
        self.full_metapath = metapath * num_traversals
        restart_prob = np.zeros(self.metapath_hops * num_traversals)
        restart_prob[self.metapath_hops::self.metapath_hops] = termination_prob
        self.restart_prob = F.zerocopy_from_numpy(restart_prob)

    def _make_context_dict(self, paths):
        dom_context_dict = {}
        pair_context_dict = {}

        # make pair context dict
        for path in paths.tolist():
            if path[1] != -1:
                if (path[0] != -1) and (path[2] != -1):
                    context = path[1]
                    pair = (path[0], path[2])
                    pair_context_dict[pair] = context
            if path[3] != -1:
                if (path[2] != -1) and (path[4] != -1):
                    context = path[3]
                    pair = (path[2], path[4])
                    pair_context_dict[pair] = context

        # make context for single nodes
        for item_nodes, ctx_nodes in zip(paths[:, [0, 2, 4]].tolist(), paths[:, [1, 3]].tolist()):
            for item in item_nodes:
                if item == -1:
                    continue
                for ctx in ctx_nodes:
                    if ctx == -1:
                        continue
                    else:
                        if item in dom_context_dict:
                            if ctx in dom_context_dict[item]:
                                dom_context_dict[item][ctx] += 1
                            else:
                                dom_context_dict[item][ctx] = 1
                        else:
                            dom_context_dict[item] = {}
                            dom_context_dict[item][ctx] = 1

        # set dorminant context for dst nodes
        for k, v in dom_context_dict.items():
            dom_context_dict[k] = Counter(v).most_common(1)[0][0]

        return (dom_context_dict, pair_context_dict)

    # pylint: disable=no-member
    def __call__(self, seed_nodes):
        seed_nodes = utils.prepare_tensor(self.G, seed_nodes, 'seed_nodes')

        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, hi = random_walk(
            self.G, seed_nodes, metapath=self.full_metapath, restart_prob=self.restart_prob)
        src = F.reshape(paths[:, self.metapath_hops::self.metapath_hops], (-1,))
        dst = F.repeat(paths[:, 0], self.num_traversals, 0)
        src_mask = (src != -1)
        src = F.boolean_mask(src, src_mask)
        dst = F.boolean_mask(dst, src_mask)
        context_dicts = self._make_context_dict(paths)

        # count the number of visits and pick the K-most frequent neighbors for each node
        neighbor_graph = convert.heterograph(
            {(self.ntype, '_E', self.ntype): (src, dst)},  # data dict
            {self.ntype: self.G.number_of_nodes(self.ntype)}  # num node dict
        )
        neighbor_graph = transform.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = select_topk(neighbor_graph, self.num_neighbors, self.weight_column)
        selected_counts = F.gather_row(counts, neighbor_graph.edata[EID])
        neighbor_graph.edata[self.weight_column] = selected_counts
        return neighbor_graph, context_dicts


class PinSAGESampler(RandomWalkNeighborSampler):
    def __init__(self, G, ntype, other_type, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, weight_column='weights'):
        metagraph = G.metagraph()
        fw_etype = list(metagraph[ntype][other_type])[0]
        bw_etype = list(metagraph[other_type][ntype])[0]
        super().__init__(G, num_traversals,
                         termination_prob, num_random_walks, num_neighbors,
                         metapath=[fw_etype, bw_etype], weight_column=weight_column)
