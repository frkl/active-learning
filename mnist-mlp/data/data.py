import torch
import util.math

class load:
	def __init__(self,fname):
		self.data=torch.load(fname);
		self.noutput=10;
		self.dof=784;
	
	def size(self):
		return self.data['im'].shape[0];
	
	def preprocess(self,stuff=None):
		if stuff is None:
			stuff={};
			stuff['mean']=self.data['im'].mean(0,keepdim=True);
			stuff['std']=self.data['im'].std(0,keepdim=True);
		self.data['im']=self.data['im']-stuff['mean'];
		self.data['im']=self.data['im']/(stuff['std']+1e-9)
		return;
	
	def cuda(self):
		self.data['im']=self.data['im'].cuda();
		self.data['label']=self.data['label'].cuda();
		return;
	
	def batch(self,batch):
		ind=torch.LongTensor(batch).cuda().random_(self.data.shape[0]);
		im=self.data['im'][ind,:].cuda();
		label=self.data['label'][ind].cuda();
		return im,label;
	
	def batch_sub(self,batch,subset):
		ind=torch.LongTensor(batch).cuda().random_(subset.shape[0]);
		ind=subset[ind];
		im=self.data['im'][ind,:].cuda();
		label=self.data['label'][ind].cuda();
		return im,label;
	
	def batch_eval(self,s,e):
		im=self.data['im'][s:e,:].cuda();
		label=self.data['label'][s:e].cuda();
		return im,label;
	