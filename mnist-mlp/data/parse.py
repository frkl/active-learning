import torch
import scipy.io

d=scipy.io.loadmat('mnist_all.mat')


data_train={};
data_test={};
data_train['im']=[];
data_train['label']=[];
data_test['im']=[];
data_test['label']=[];
for digit in range(0,10):
	imtrain=torch.from_numpy(d['train%d'%digit]).view(-1,28,28).float();
	imtest=torch.from_numpy(d['test%d'%digit]).view(-1,28,28).float();
	data_train['im'].append(imtrain);
	data_train['label'].append(torch.LongTensor(imtrain.shape[0]).fill_(digit));
	data_test['im'].append(imtest);
	data_test['label'].append(torch.LongTensor(imtest.shape[0]).fill_(digit));

data_train['im']=torch.cat(data_train['im'],dim=0);
data_train['label']=torch.cat(data_train['label'],dim=0);
data_test['im']=torch.cat(data_test['im'],dim=0);
data_test['label']=torch.cat(data_test['label'],dim=0);


torch.save(data_train,'data_train.pt');
torch.save(data_test,'data_test.pt');
