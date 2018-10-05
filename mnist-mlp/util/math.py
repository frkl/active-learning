import torch
import math
import torch.nn as nn
from torch.autograd import Variable,grad
import torch.nn.functional as F

def random(N):
	return int(torch.LongTensor(1).random_(N));

def logsumexp_v(inputs,dim=None,keepdim=False):
	return (inputs-F.log_softmax(inputs,dim=dim)).mean(dim,keepdim=keepdim);

def logsumexp(inputs,dim=None,keepdim=False):
	return (inputs-F.log_softmax(Variable(inputs),dim=dim).data).mean(dim,keepdim=keepdim);

def logmeanexp_v(inputs,dim=None,keepdim=False):
	return (inputs-F.log_softmax(inputs,dim=dim)).mean(dim,keepdim=keepdim)-math.log(inputs.size(dim));

def logmeanexp(inputs,dim=None,keepdim=False):
	return (inputs-F.log_softmax(Variable(inputs),dim=dim).data).mean(dim,keepdim=keepdim)-math.log(inputs.size(dim));

def precision(score,label):
	_,ind=score.sort(dim=0,descending=True);
	_,rank=ind.sort(dim=0);
	ap=[];
	for c in range(score.size(1)):
		if label[:,c].sum()==0:
			continue; #no positive label: trivial class
		else:
			pos=label[:,c].nonzero().view(-1);
			N=pos.size(0);
			rank_pos=rank[:,c].clone()[pos];
			sr,_=rank_pos.sort(dim=0);
			prec=(torch.arange(1,N+1)/(sr.float().cpu()+1)).mean();
			ap.append(prec);
	if len(ap)==0:
		return 0.0;
	else:
		return sum(ap)/len(ap);
