 
import torch
from torch_geometric.data import Data
from torch.nn import functional as F

from .layers import GraphConvolution, GraphAggregator, GraphMatchingConvolution
from .utils import adj_matrix_to_edge_index
from .utils import create_batch, trim_feats
from .utils import acc_f1


class GenericGNN(torch.nn.Module):
	def __init__(self, args):
		super(GenericGNN, self).__init__()
		self.args = args
		if args.n_classes > 2:
			self.f1_average = 'micro'
		else:
			self.f1_average = 'binary'
		self.layers = torch.nn.ModuleList()
		self.layers.append(GraphConvolution(self.args.feat_dim, self.args.dim, args))
		for _ in range(self.args.num_layers - 1):
			self.layers.append(
				GraphConvolution(self.args.dim, self.args.dim, args),
			)
		self.aggregator = GraphAggregator(self.args.dim, self.args.dim, self.args)
		self.cls = torch.nn.Linear(self.args.dim, self.args.n_classes)

	def compute_emb(self, feats, adjs, sizes):
		# batch = create_batch(sizes)
		# batch = torch.tensor(batch, dtype=torch.int64)
		edge_index = adj_matrix_to_edge_index(adjs)
		batch = create_batch(sizes)
		feats = trim_feats(feats, sizes)
		for i in range(self.args.num_layers):
			# convolution
			feats, edge_index = self.layers[i](feats, edge_index)
		# aggregator
		feats = self.aggregator(feats, edge_index, batch)
		return feats

	def forward(self, feats_1, adjs_1, feats_2, adjs_2, sizes_1, sizes_2):
		# computing the embedding
		emb_1 = self.compute_emb(feats_1, adjs_1, sizes_1)
		emb_2 = self.compute_emb(feats_2, adjs_2, sizes_2)
		outputs = torch.cat((emb_1, emb_2), 1)
		outputs = outputs.reshape(outputs.size(0), -1)

		# classification
		outputs = self.cls.forward(outputs)
		return outputs

	def compute_metrics(self, outputs, labels, split, backpropagate=False):
		outputs = F.log_softmax(outputs, dim=1)
		loss = F.nll_loss(outputs, labels)
		if backpropagate:
			loss.backward()
		if split == "train":
			verbose = True
		else:
			verbose = True
		acc, f1 = acc_f1(
			outputs, labels, average=self.f1_average, logging=self.args.logging, verbose=verbose)
		metrics = {'loss': loss, 'acc': acc, 'f1': f1}
		return metrics, outputs.shape[0]

	def init_metric_dict(self):
		return {'acc': -1, 'f1': -1}

	def has_improved(self, m1, m2):
		return m1["acc"] < m2["acc"]

class GraphMatchingNetwork(torch.nn.Module):
	def __init__(self, args):
		super(GraphMatchingNetwork, self).__init__()
		self.args = args
		if args.n_classes > 2:
			self.f1_average = 'micro'
		else:
			self.f1_average = 'binary'
		self.layers = torch.nn.ModuleList()
		self.layers.append(GraphMatchingConvolution(
			self.args.feat_dim, self.args.dim, args
		))
		for _ in range(self.args.num_layers - 1):
			self.layers.append(
				GraphMatchingConvolution(
					self.args.dim, self.args.dim, args
				)
			)
		self.aggregator = GraphAggregator(self.args.dim, self.args.dim, self.args)
		self.cls = torch.nn.Linear(self.args.dim * 2, self.args.n_classes)


	def compute_emb(self, feats, edge_index, batch):
		# data = Data(x=feats, edge_index=edge_index, batch=batch)
		for i in range(self.args.num_layers):
			# convolution
			feats, edge_index, batch = self.layers[i](feats, edge_index, batch)
		# aggregator
		feats = self.aggregator(feats, edge_index, batch)
		return feats, edge_index, batch

	def combine_pair_embedding(self, feats_1, adjs_1, feats_2, adjs_2, sizes_1, sizes_2):
		sizes = torch.cat([sizes_1, sizes_2], dim=0)
		feats = torch.cat([feats_1, feats_2], dim=0)
		feats = trim_feats(feats, sizes)
		edge_index_1 = adj_matrix_to_edge_index(adjs_1)
		edge_index_2 = adj_matrix_to_edge_index(adjs_2)
		edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
		batch = create_batch(sizes)
		feats = feats.to(self.args.device)
		edge_index = edge_index.to(self.args.device)
		batch = batch.to(self.args.device)
		return feats, edge_index, batch

	def forward(self, feats_1, adjs_1, feats_2, adjs_2, sizes_1, sizes_2):
		# computing the embedding
		feats, edge_index, batch = self.combine_pair_embedding(feats_1, adjs_1, feats_2, adjs_2, sizes_1, sizes_2)
		emb, _, _ = self.compute_emb(feats, edge_index, batch)
		emb_1 = emb[:emb.shape[0] // 2, :]
		emb_2 = emb[emb.shape[0] // 2:, :]
		outputs = torch.cat((emb_1, emb_2), 1)
		outputs = outputs.reshape(outputs.size(0), -1)

		# classification
		outputs = self.cls.forward(outputs)
		return outputs

	def compute_metrics(self, outputs, labels, split, backpropagate=False):
		outputs = F.log_softmax(outputs, dim=1)
		loss = F.nll_loss(outputs, labels)
		if backpropagate:
			loss.backward()
		if split == "train":
			verbose = True
		else:
			verbose = True
		acc, f1 = acc_f1(
			outputs, labels, average=self.f1_average, logging=self.args.logging, verbose=verbose)
		metrics = {'loss': loss, 'acc': acc, 'f1': f1}
		return metrics, outputs.shape[0]

	def init_metric_dict(self):
		return {'acc': -1, 'f1': -1}

	def has_improved(self, m1, m2):
		return m1["acc"] < m2["acc"]
