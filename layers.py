import torch
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_scatter import scatter_mean

from .utils import batch_block_pair_attention


class GraphConvolution(MessagePassing):
	def __init__(self, in_channels, out_channels, args, aggr="add"):
		super(GraphConvolution, self).__init__(aggr=aggr)
		self.args = args
		self.lin_node = torch.nn.Linear(in_channels, out_channels)
		self.lin_message = torch.nn.Linear(out_channels * 2, out_channels)
		self.lin_passing = torch.nn.Linear(out_channels + in_channels, out_channels)
		self.batch_norm = BatchNorm(out_channels)

	def forward(self, x, edge_index):
		x = self.lin_node(x)
		return self.propagate(edge_index, x=x)

	def message(self, edge_index_i, x_i, x_j):
		m = self.lin_message(torch.cat([x_i, x_j], dim=1))
		return m

	def update(self, aggr_out, edge_index, x):
		aggr_out = self.lin_passing(torch.cat([x, aggr_out]))
		aggr_out = self.batch_norm(aggr_out)
		return aggr_out

class GraphMatchingConvolution(MessagePassing):
	def __init__(self, in_channels, out_channels, args, aggr="add"):
		super(GraphMatchingConvolution, self).__init__(aggr=aggr)
		self.args = args
		self.lin_node = torch.nn.Linear(in_channels, out_channels)
		self.lin_message = torch.nn.Linear(out_channels * 2, out_channels)
		self.lin_passing = torch.nn.Linear(out_channels + in_channels, out_channels)
		self.batch_norm = BatchNorm(out_channels)

	def forward(self, x, edge_index, batch):
		x_transformed = self.lin_node(x)
		return self.propagate(edge_index, x=x_transformed, original_x=x, batch=batch)

	def message(self, edge_index_i, x_i, x_j):
		x = torch.cat([x_i, x_j], dim=1)
		m = self.lin_message(x)
		return m

	def update(self, aggr_out, edge_index, x, original_x, batch):
		n_graphs = torch.unique(batch).shape[0]
		cross_graph_attention = batch_block_pair_attention(original_x, batch, n_graphs)
		attention_input = original_x - cross_graph_attention
		aggr_out = self.lin_passing(torch.cat([aggr_out, attention_input], dim=1))
		aggr_out = self.batch_norm(aggr_out)
		return aggr_out, edge_index, batch

class GraphAggregator(torch.nn.Module):
	def __init__(self, in_channels, out_channels, args):
		super(GraphAggregator, self).__init__()
		self.lin = torch.nn.Linear(in_channels, out_channels)
		self.lin_gate = torch.nn.Linear(in_channels, out_channels)
		self.lin_final = torch.nn.Linear(out_channels, out_channels)
		self.args = args

	def forward(self, x, edge_index, batch):
		# print("x:", x.shape)
		x_states = self.lin(x)
		x_gates = torch.nn.functional.softmax(self.lin_gate(x), dim=1)
		x_states = x_states * x_gates
		# print("x_states:", x_states.shape)
		# print("batch:", batch.shape)
		x_states = scatter_mean(x_states, batch, dim=0)
		x_states = self.lin_final(x_states)
		return x_states
