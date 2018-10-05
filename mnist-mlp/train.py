#Python 2/3 compatible headers
from __future__ import unicode_literals,division
from builtins import int
from builtins import range

#System packages
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import scipy
import scipy.misc
import math
import time
import argparse
import sys
import re
import importlib

#Our libraries
import data.data as data
import util.math
import util.file
import util.session_manager as session_manager		#Manage runs


#Command line options
parser=argparse.ArgumentParser(description='Contextual distance distillation')
#	General
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_train', type=str, default='data/data_train.pt')
parser.add_argument('--data_test', type=str, default='data/data_test.pt')
#	Model
parser.add_argument('--arch', type=str, default='arch.mlp')
parser.add_argument('--nh', type=int, default=1024)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)
#	Active selection
parser.add_argument('--method', type=str, default='entropy') #entropy, curiosity, goal
parser.add_argument('--init', type=int, default=50)
parser.add_argument('--rounds', type=int, default=100)
parser.add_argument('--inc', type=int, default=10)
parser.add_argument('--bnn', type=int, default=50)
parser.add_argument('--restart', action="store_true", default=False)
#	Optimization
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--times', type=int, default=3)
parser.add_argument('--ensemble', type=int, default=5)

params=parser.parse_args();
assert torch.cuda.is_available(),"You running on CPUs?"
torch.manual_seed(params.seed) #Fix random seed.
torch.cuda.manual_seed(params.seed);

params.argv=sys.argv;

#Load data
data_train=data.load(params.data_train);
data_test=data.load(params.data_test);
params.stuff=data_train.preprocess();
_=data_test.preprocess(params.stuff);

data_train.cuda();
data_test.cuda();
params.dof=data_train.dof;
params.noutput=data_train.noutput;

#Create Networks
arch=importlib.import_module(params.arch);
net=arch.new(params).cuda();
import util.pm as pm_proto
pms=[];
for i in range(0,params.ensemble):
	pms.append(pm_proto.dropout([p.data for p in net.parameters()]));

#Create session: keep everything in one folder
session=session_manager.Session(); #Create session
torch.save({'params':params},session.file('params.pt'));
tmp=vars(params);
tmp=dict([(k,tmp[k]) for k in tmp if not(k=='stuff')]);
print(tmp);
util.file.write_json(session.file('params.json'),tmp); #Write a human-readable parameter json
session.file('model','dummy');

#Randomly initialize active learning
subset=set(torch.randperm(data_train.size())[:params.init].tolist());
#Pre-allocate score matrix since it can get big
score_train=torch.Tensor(data_train.size(),params.bnn,params.noutput).cuda();
score_test=torch.Tensor(data_test.size(),params.bnn,params.noutput).cuda();
t0=time.time();
for iter_al in range(0,params.rounds):
	#Train ensemble of models
	#Optimization setup
	npts=len(subset);
	niter_per_epoch=max(math.ceil(npts/params.batch),5);
	subset_ind=torch.LongTensor(list(subset)).cuda();
	if params.restart:
		pms=[];
		for i in range(0,params.ensemble):
			pms.append(pm_proto.dropout([p.data for p in net.parameters()]));
	
	#Train a bunch of ensembles
	session.log('Iter %d, %d iter per epoch, time %f'%(iter_al,niter_per_epoch,time.time()-t0));
	net.train();
	for eid in range(0,params.ensemble):
		opt=optim.Adam(pms[eid].parameters(),lr=params.lr);
		for iter in range(1,niter_per_epoch*params.epochs+1):
			pms[eid].zero_grad();
			total_loss=0;
			for t in range(0,params.times):
				net.zero_grad();
				w=pms[eid]();
				pms[eid].impose(w,net.parameters())
				im,label=data_train.batch_sub(params.batch,subset_ind);
				im=Variable(im.cuda());
				label=Variable(label.cuda());
				score=net(im);
				loss=F.cross_entropy(score,label)/params.times;
				loss.backward();
				total_loss=total_loss+float(loss);
				pms[eid].propagate(w,net.parameters());
			opt.step();
			if iter%(niter_per_epoch*10)==0:
				session.log('\tensemble %d, epoch %.04f, loss %f, time %f'%(eid,iter/niter_per_epoch,total_loss,time.time()-t0));
	
	session.log('Iter %d, models trained, time %f'%(iter_al,time.time()-t0));
	#Randomly selects models for testing
	ensemble_ind=torch.LongTensor(params.bnn).random_(params.ensemble);
	seeds=torch.LongTensor(params.bnn).random_(1073741824);
	
	#Evaluate test accuracy
	net.eval();
	npts_test=data_test.size();
	test_acc=[];
	for i in range(0,npts_test,params.batch):
		r=min(npts_test,i+params.batch);
		for j in range(0,params.bnn):
			eid=int(ensemble_ind[j]);
			seed=float(seeds[j]);
			w=pms[eid](seed);
			pms[eid].impose(w,net.parameters());
			im,label=data_test.batch_eval(i,r);
			im=Variable(im.cuda());
			score_test[i:r,j,:]=F.log_softmax(net(im),dim=1).data;
		score=util.math.logmeanexp(score_test[i:r,:,:],dim=1);
		_,pred=score.max(dim=1);
		test_acc.append(label.eq(pred).float().sum()/npts_test);
	test_acc=sum(test_acc);
	session.log('Iter %d, test acc %f, time %f'%(iter_al,test_acc,time.time()-t0));
	
	#Evaluate scores for training pool examples
	net.eval();
	npts=data_train.size();
	for i in range(0,npts,params.batch):
		r=min(npts,i+params.batch);
		for j in range(0,params.bnn):
			eid=int(ensemble_ind[j]);
			seed=float(seeds[j]);
			w=pms[eid](seed);
			pms[eid].impose(w,net.parameters());
			im,label=data_train.batch_eval(i,r);
			im=Variable(im.cuda());
			score_train[i:r,j,:]=F.log_softmax(net(im),dim=1).data;
	session.log('Iter %d, train scores computed, time %f'%(iter_al,time.time()-t0));
	
	#Select new examples according to protocol
	if params.method=='entropy':
		logpy=util.math.logmeanexp(score_train,dim=1); #not very scalable but let's make do with it
		entropy=-(logpy*torch.exp(logpy)).sum(1);
		top_e,ind=entropy.sort(0,True);
		nsel=min(data_train.size()-len(subset),params.inc);
		n=0;
		total_entropy=0;
		for i in range(0,data_train.size()):
			if not(int(ind[i]) in subset):
				subset.add(ind[i]);
				n=n+1;
				total_entropy=total_entropy+float(top_e[i]);
				if n>=nsel:
					break;
		session.log('Iter %d, selected %d using entropy, entropy %f, time %f'%(iter_al,n,total_entropy,time.time()-t0));
	elif params.method=='curiosity':
		logpy=util.math.logmeanexp(score_train,dim=1,keepdim=True);
		model_param_mi=(torch.exp(score_train)*(score_train-logpy)).mean(1).sum(1);
		top_mi,ind=model_param_mi.sort(0,True);
		nsel=min(data_train.size()-len(subset),params.inc);
		n=0;
		total_gain=0;
		for i in range(0,data_train.size()):
			if not(int(ind[i]) in subset):
				subset.add(ind[i]);
				n=n+1;
				total_gain=total_gain+float(top_mi[i]);
				if n>=nsel:
					break;
		session.log('Iter %d, selected %d using curiosity, param information gain %f, time %f'%(iter_al,n,total_gain,time.time()-t0));
	elif params.method=='goal':
		#Compute average test vector
		avg_fim_test=[];
		for i in range(0,data_test.size(),params.batch):
			r=min(data_test.size(),params.batch);
			fim=score_test[i:r,:,:]-torch.logmeanexp(score_test[i:r,:,:],dim=1,keepdim=True);
			fim=torch.bmm(torch.exp(fim),torch.exp(score_test[i:r,:,:].clone().permute(0,2,1)))/params.bnn;
			avg_fim_test.append(fim.sum(0,keepdim=True)/data_test.size());
		avg_fim_test=torch.cat(avg_fim_test,dim=0);
		avg_fim_test=avg_fim_test.mean(0);
		#Compute best train-test matching
		train_test_mi=torch.Tensor(data_train.size()).cuda();
		for i in range(0,data_train.size(),params.batch):
			r=min(data_train.size(),params.batch);
			fim=score_train[i:r,:,:]-torch.logmeanexp(score_train[i:r,:,:],dim=1,keepdim=True);
			fim=torch.bmm(torch.exp(fim),torch.exp(score_train[i:r,:,:].clone().permute(0,2,1)))/params.bnn;
			train_test_mi[i:r]=(torch.mm(fim.view(r-i,-1),avg_fim_test.view(-1,1))-1)/2
		
		top_mi,ind=train_test_mi.sort(0,True);
		nsel=min(data_train.size()-len(subset),params.inc);
		n=0;
		total_gain=0;
		for i in range(0,data_train.size()):
			if not(int(ind[i]) in subset):
				subset.add(ind[i]);
				n=n+1;
				total_gain=total_gain+float(top_mi[i]);
				if n>=nsel:
					break;
		session.log('Iter %d, selected %d using goal-drive learning, testset information gain %f, time %f'%(iter_al,n,total_gain,time.time()-t0));
	
	torch.save({'pms':[pm.state_dict() for pm in pms],'subset':subset},session.file('model%02d.pt'%iter_al))
	