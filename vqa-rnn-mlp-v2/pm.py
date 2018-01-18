#Parameter manager
#Author: Xiao Lin
#Implements simple variational parameter distributions

#Initialize: list of tensors same size as target parameters
#Forward output: list of variables

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#Dropout distribution
# w=Bernoulli(p).*theta
class dropout(nn.Module):
	def __init__(self,params,p=0.5):
		super(dropout,self).__init__()
		self.params=nn.ParameterList();
		self.p=p;
		for param in params:
			self.params.append(nn.Parameter(param.clone()));
		return;
	#
	def forward(self,seed):
		#Store rng
		rng_cpu=torch.get_rng_state();
		rng_gpu=torch.cuda.get_rng_state();
		torch.manual_seed(seed);
		torch.cuda.manual_seed(seed);
		mask=[];
		for param in self.params:
			mask.append(Variable(param.data.clone().bernoulli_(self.p)));
		#Recover rng
		torch.set_rng_state(rng_cpu);
		torch.cuda.set_rng_state(rng_gpu);
		#Compute output	
		out=[];
		for i,param in enumerate(self.params):
			out.append(mask[i]*param);
		return out;

#-----------Usage------------
#Initialize
#x=torch.zeros(1,5).normal_();
#net=nn.Linear(5,3);
#pm=dropout([p.data for p in net.parameters()]);
#opt=torch.optim.RMSprop(pm.parameters(),lr=3e-4);

#Forward-backward
#pm.zero_grad();
#for iter in range(16):
#	seed=torch.IntTensor(1).random_()[0]
#	net.zero_grad();
#	w=pm(seed);
#	for i,param in enumerate(net.parameters()):
#		param.data[:]=w[i].data;
#	y=net(Variable(x)).sum();
#	y.backward();
#	dw=[param.grad for param in net.parameters()];
#	torch.autograd.backward(w,dw);

#Update parameters
#opt.step();

