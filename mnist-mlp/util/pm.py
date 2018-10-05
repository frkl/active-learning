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
import math

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
	def noise(self,seed=None):
		if seed is None:
			eps=[];
			for param in self.params:
				eps.append(Variable(param.data.clone().bernoulli_(self.p)));
			return eps;
		else:
			rng_cpu=torch.get_rng_state();
			rng_gpu=torch.cuda.get_rng_state();
			torch.manual_seed(seed);
			torch.cuda.manual_seed(seed);
			#generate noise
			eps=[];
			for param in self.params:
				eps.append(Variable(param.data.clone().bernoulli_(self.p)));
			#Recover rng
			torch.set_rng_state(rng_cpu);
			torch.cuda.set_rng_state(rng_gpu);
			return eps;
	
	def forward(self,seed=None):
		noise=self.noise(seed);
		out=[];
		for i,param in enumerate(self.params):
			out.append(noise[i]*self.params[i]);
		return out;
	
	def impose(self,w,params):
		for i,param in enumerate(params):
			param.data[:]=w[i].data;
		return;
	
	def propagate(self,w,params):
		dw=[param.grad for param in params];
		torch.autograd.backward(w,dw);
		return;


class gaussian(nn.Module):
	def __init__(self,params,std=1e-1):
		super(gaussian,self).__init__()
		self.mean=nn.ParameterList();
		self.lnstd=nn.ParameterList();
		lnstd=math.log(std);
		for param in params:
			self.mean.append(nn.Parameter(param.clone()));
			self.lnstd.append(nn.Parameter(param.clone().fill_(lnstd/10)));
		return;
	#
	def noise(self,seed=None):
		if seed is None:
			eps=[];
			for param in self.mean:
				eps.append(Variable(param.data.clone().normal_(0,1)));
			return eps;
		else:
			rng_cpu=torch.get_rng_state();
			rng_gpu=torch.cuda.get_rng_state();
			torch.manual_seed(seed);
			torch.cuda.manual_seed(seed);
			#generate noise
			eps=[];
			for param in self.mean:
				eps.append(Variable(param.data.clone().normal_(0,1)));
			#Recover rng
			torch.set_rng_state(rng_cpu);
			torch.cuda.set_rng_state(rng_gpu);
			return eps;
	
	
	def forward(self,seed=None,eval=False):
		if eval:
			return self.mean;
		else:
			noise=self.noise(seed);
			out=[];
			for i,param in enumerate(self.mean):
				out.append(noise[i]*torch.exp((self.lnstd[i]*10).clamp(-20,20))+self.mean[i]);
			return out;
	
	
	def impose(self,w,params):
		for i,param in enumerate(params):
			param.data[:]=w[i].data;
		return;
	
	def propagate(self,w,params):
		dw=[param.grad for param in params];
		torch.autograd.backward(w,dw);
		return;
	
	def lnp(self,w):
		lnp_entropy=[];
		for i,param in enumerate(self.lnstd):
			lnp_entropy.append(-param.sum()*10);
		lnp_entropy=sum(lnp_entropy);
		lnp_std=[]
		for i,param in enumerate(w):
			lnp_std.append(-(((w[i]-self.mean[i])**2)/2/(torch.exp((self.lnstd[i]*10).clamp(-20,20)*2)+1e-10)).sum());
		lnp_std=sum(lnp_std);
		return lnp_entropy+lnp_std;

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

