import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class new(nn.Module):
	def __init__(self,params):
		super(new,self).__init__()
		self.layers=nn.ModuleList();
		self.dropout=params.dropout;
		if params.nlayers==1:
			self.layers.append(nn.Linear(params.dof,params.noutput));
		else:
			self.layers.append(nn.Linear(params.dof,params.nh));
			for i in range(1,params.nlayers-1):
				self.layers.append(nn.Linear(params.nh,params.nh));
			self.layers.append(nn.Linear(params.nh,params.noutput));
		return;
	#
	def forward(self,x):
		batch=x.shape[0];
		x=x.view(batch,-1);
		x=self.layers[0](x);
		for i in range(1,len(self.layers)):
			x=F.relu(x);
			x=F.dropout(x,self.dropout,self.training);
			x=self.layers[i](x);
		return x;