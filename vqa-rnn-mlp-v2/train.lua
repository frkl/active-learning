cmd = torch.CmdLine();
cmd:text('Actively learn a VQA model');
cmd:text('Strategy')
cmd:option('-method','curiosity','Active learning strategy. {random, entropy, curiosity, goal_avg} (not reported in paper, goal_max and mi_train)');
cmd:option('-init',50000,'Initial samples');
cmd:option('-inc',2000,'How many examples to select each round');
cmd:option('-maxIter',200,'How many rounds');
cmd:option('-ntrials',50,'How many dropout samples to compute strategy scores');
cmd:option('-seed',1234,'random seed');
 
cmd:text('Dataset')
cmd:option('-data','../../dataset/balanced_vqa/dataset_train.t7','Dataset for training');
cmd:option('-data_test','../../dataset/balanced_vqa/dataset_val.t7','Dataset for validation, directly evaluate accuracy on the fly');
cmd:option('-train_on_sel',true,'Train only on examples whose answers are in top 1000 answers');
cmd:option('-test_on_binary',false,'Test set is only yes/no questions');
cmd:option('-test_on_number',false,'Test set is only number questions');
cmd:option('-test_on_other',false,'Test set is only other questions');


cmd:text('Model parameters');
cmd:option('-nanswers',1000,'Number of most frequent answers to use');
cmd:option('-nhword',200,'Word embedding size');
cmd:option('-nh',512,'RNN size');
cmd:option('-nlayers',2,'RNN layers');
cmd:option('-nhcommon',1024,'Common embedding size'); 

cmd:text('Optimization parameters');
cmd:option('-reset',false,'Reset model parameter in every active learning iteration.');
cmd:option('-ntimes',8,'Number of model parameter samples per iteration during training');
cmd:option('-batch',128,'Batch size for training');
cmd:option('-batch_eval',500,'Batch size for making predictions');
cmd:option('-lr',3e-4,'Learning rate');
cmd:option('-decay',150,'Learning rate decay in epochs');
cmd:option('-l2',0,'l2 decay. Not enabled in code. See ForwardBackward()');
cmd:option('-epochs',100,'Number of epochs for training');
cmd:option('-init_epochs',100,'Number of epochs for the first active learning iteration.');
params=cmd:parse(arg);





require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
require 'optim_updates'
require 'utils.lua'
RNN=require('word_RNN'); --Implementation of RNN
Net=require('net');
PM=require('param_gen'); --Implementation of a model parameter generator



--Initialize CPU/GPU seeds
torch.manualSeed(params.seed);
cutorch.manualSeed(params.seed);
seeds={};

print('Loading dataset');
function sequence_length(seq)
	local v=seq:gt(0):long():sum(2):view(-1):long();
	return v;
end
dataset=torch.load(params.data);
dataset.question.tokens,params.question_dictionary=encode_sents(dataset.question.question);
dataset.question.question=nil;
collectgarbage();
dataset.question.labels,params.answer_dictionary=encode_sents(dataset.question.answer,nil,params.nanswers);
dataset.question.answer=nil;
collectgarbage();
params.nhsent=params.nh*params.nlayers*2; --Using both cell and hidden of LSTM
params.noutput=params.nanswers+1; -- +1 for UNK
params.nhoutput=1; --RNN with dummy output dim=1
params.nhimage=dataset.image.fvs:size(2);

if params.train_on_sel then
	params.noutput=params.nanswers;
	valid_q=dataset.question.labels:le(params.nanswers):long():sum(2):gt(0);
	print(string.format('%d valid examples for training',valid_q:long():sum()))
	valid_ind=torch.range(1,dataset.question.labels:size(1))[valid_q]:long();
	dataset.question.tokens=dataset.question.tokens:index(1,valid_ind);
	dataset.question.labels=dataset.question.labels:index(1,valid_ind);
	tmp={};
	for i=1,valid_ind:size(1) do
		tmp[i]=dataset.question.imname[valid_ind[i]];
	end
	dataset.question.imname=tmp;
end


--Process tokens into ids and process answers into ids.
dataset_test=torch.load(params.data_test);
dataset_test.question.tokens,_=encode_sents(dataset_test.question.question,params.question_dictionary);
dataset_test.question.question=nil;
collectgarbage();
dataset_test.question.labels,_=encode_sents(dataset_test.question.answer,params.answer_dictionary,params.nanswers);
dataset_test.question.answer=nil;
dataset_test.image.fvs=nil;
collectgarbage();
if params.test_on_binary then
	valid_q=torch.ones(dataset_test.question.tokens:size(1)):eq(1);
	for i=1,dataset_test.question.tokens:size(1) do
		if dataset_test.question.answer_type[i]~='yes/no' then
			valid_q[i]=0;
		end
	end
	print(string.format('%d valid examples',valid_q:long():sum()))
	valid_ind=torch.range(1,dataset_test.question.labels:size(1))[valid_q]:long();
	dataset_test.question.tokens=dataset_test.question.tokens:index(1,valid_ind);
	dataset_test.question.labels=dataset_test.question.labels:index(1,valid_ind);
	tmp={};
	for i=1,valid_ind:size(1) do
		tmp[i]=dataset_test.question.imname[valid_ind[i]];
	end
	dataset_test.question.imname=tmp;
end
if params.test_on_number then
	valid_q=torch.ones(dataset_test.question.tokens:size(1)):eq(1);
	for i=1,dataset_test.question.tokens:size(1) do
		if dataset_test.question.answer_type[i]~='number' then
			valid_q[i]=0;
		end
	end
	print(string.format('%d valid examples',valid_q:long():sum()))
	valid_ind=torch.range(1,dataset_test.question.labels:size(1))[valid_q]:long();
	dataset_test.question.tokens=dataset_test.question.tokens:index(1,valid_ind);
	dataset_test.question.labels=dataset_test.question.labels:index(1,valid_ind);
	tmp={};
	for i=1,valid_ind:size(1) do
		tmp[i]=dataset_test.question.imname[valid_ind[i]];
	end
	dataset_test.question.imname=tmp;
end
if params.test_on_other then
	valid_q=torch.ones(dataset_test.question.tokens:size(1)):eq(1);
	for i=1,dataset_test.question.tokens:size(1) do
		if dataset_test.question.answer_type[i]~='other' then
			valid_q[i]=0;
		end
	end
	print(string.format('%d valid examples',valid_q:long():sum()))
	valid_ind=torch.range(1,dataset_test.question.labels:size(1))[valid_q]:long();
	dataset_test.question.tokens=dataset_test.question.tokens:index(1,valid_ind);
	dataset_test.question.labels=dataset_test.question.labels:index(1,valid_ind);
	tmp={};
	for i=1,valid_ind:size(1) do
		tmp[i]=dataset_test.question.imname[valid_ind[i]];
	end
	dataset_test.question.imname=tmp;
end

print('Building network');
function VQA_PW(nhA,nhB,nhcommon,noutput)
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(q));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Normalize(2)(i)));
	local output=nn.Linear(nhcommon,noutput)(nn.CMulTable()({qc,ic}));
	return nn.gModule({q,i},{output});
end
Q_embedding_net=Net:wrap(nn.Sequential():add(nn.LookupTable(table.getn(params.question_dictionary)+1,params.nhword)),true);
Q_encoder_net=RNN:new(RNN.unit.lstm(params.nhword,params.nh,params.nhoutput,params.nlayers,0.0),math.max(dataset.question.tokens:size(2),dataset_test.question.tokens:size(2)),true);
multimodal_net=Net:wrap(VQA_PW(params.nhsent,params.nhimage,params.nhcommon,params.noutput),true);
--Param generator
pm={};
pm.Q_embedding_net=PM:new(Q_embedding_net.w:size(1),Q_embedding_net.w);
pm.Q_embedding_net.w:uniform(-0.0001,0.0001); --initialize with sth small, so that UNK embedding is small.
pm.Q_encoder_net=PM:new(Q_encoder_net.w:size(1),Q_encoder_net.w);
pm.multimodal_net=PM:new(multimodal_net.w:size(1),multimodal_net.w);
--Criterion
criterion=nn.CrossEntropyCriterion():cuda();
logsoftmax=nn.LogSoftMax():cuda();
--Create dummy states and gradients
dummy_state=torch.DoubleTensor(params.nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(params.nhoutput):fill(0):cuda();



print('Initializing session');
paths.mkdir('sessions')
Session=require('session_manager');
session=Session:init('./sessions');
basedir=session:new(params);
paths.mkdir(paths.concat(basedir,'model'));
paths.mkdir(paths.concat(basedir,'mask'));

log_file=paths.concat(basedir,string.format('log.txt',1));
function Log(msg)
	local f = io.open(log_file, "a")
	print(msg);
	f:write(msg..'\n');
	f:close()
end
function majority_answer(arr,K)
	local cnt={};
	for k=1,arr:size(1) do
		if arr[k]>0 and arr[k]<=K then
			cnt[arr[k]]=cnt[arr[k]] or 0;
			cnt[arr[k]]=cnt[arr[k]]+1;
		end
	end
	local majority=0;
	local cnt_majority=0;
	for k,v in pairs(cnt) do
		if v>cnt_majority then
			majority=k;cnt_majority=v;
		end
	end
	return majority;
end
--Batch function
dataset.question.majority_answer=torch.LongTensor(dataset.question.labels:size(1));
for i=1,dataset.question.labels:size(1) do
	dataset.question.majority_answer[i]=majority_answer(dataset.question.labels[i],params.noutput);
end
function dataset:batch_train(batch_size,mask)
	local timer = torch.Timer();
	local nqs=mask:size(1);
	local qinds=torch.LongTensor(batch_size):fill(0);
	local labels=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		while true do
			qinds[i]=mask[torch.random(nqs)];
			local answer=self.question.majority_answer[qinds[i]];
			if answer>0 then
				iminds[i]=self.image.lookup[self.question.imname[qinds[i]]];
				labels[i]=answer;
				break;
			end
		end
	end
	local fv_sorted_q=sort_by_length_left_aligned(self.question.tokens:index(1,qinds),true);
	local fv_im=self.image.fvs:index(1,iminds);
	return fv_sorted_q,fv_im:cuda(),labels:cuda();
end
function dataset:batch_eval(s,e)
	local timer = torch.Timer();
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		iminds[i]=self.image.lookup[self.question.imname[qinds[i]]];
	end
	local fv_sorted_q=sort_by_length_left_aligned(self.question.tokens:index(1,qinds),true);
	local fv_im=self.image.fvs:index(1,iminds);
	return fv_sorted_q,fv_im:cuda();
end
function dataset_test:batch_eval(s,e)
	local timer = torch.Timer();
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		iminds[i]=dataset.image.lookup[self.question.imname[qinds[i]]];
	end
	local fv_sorted_q=sort_by_length_left_aligned(self.question.tokens:index(1,qinds),true);
	local fv_im=dataset.image.fvs:index(1,iminds);
	return fv_sorted_q,fv_im:cuda();
end
--Network forward backward
function ForwardBackward(batch,mask,ntimes)
	--Grab a batch--
	local timer = torch.Timer();
	local f=0;
	--clear gradients--
	pm.Q_embedding_net.dw:zero();
	pm.Q_encoder_net.dw:zero();
	pm.multimodal_net.dw:zero();
	for t=1,ntimes do
		--clear gradients--
		Q_embedding_net.dw:zero();
		Q_encoder_net.dw:zero();
		multimodal_net.dw:zero();
		--sample parameter--
		Q_embedding_net.w[{}]=pm.Q_embedding_net:forward();
		Q_encoder_net.w[{}]=pm.Q_encoder_net:forward();
		multimodal_net.w[{}]=pm.multimodal_net:forward();
		local fv_Q,fv_I,labels=dataset:batch_train(batch,mask);
		--Forward/backward
		local embedding_Q=Q_embedding_net.deploy:forward(fv_Q.words);
		local state_Q,_=Q_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_Q.map_to_sequence:size(1),1),embedding_Q,fv_Q.batch_sizes);
		local tv_Q=state_Q:index(1,fv_Q.map_to_sequence);
		local scores=multimodal_net.deploy:forward({tv_Q,fv_I});
		f=f+criterion:forward(scores,labels);
		local dscores=criterion:backward(scores,labels);
		local tmp=multimodal_net.deploy:backward({tv_Q,fv_I},dscores);
		local dstate_Q=tmp[1]:index(1,fv_Q.map_to_rnn);
		local _,dembedding_Q=Q_encoder_net:backward(torch.repeatTensor(dummy_state:fill(0),batch,1),embedding_Q,fv_Q.batch_sizes,dstate_Q,dummy_output);
		Q_embedding_net.deploy:backward(fv_Q.words,dembedding_Q);
		--Q_encoder_net.dw:add(params.l2/mask:size(1)*Q_encoder_net.w)
		--Q_embedding_net.dw:add(params.l2/mask:size(1)*Q_embedding_net.w)
		--multimodal_net.dw:add(params.l2/mask:size(1)*multimodal_net.w)
		--Q_encoder_net.dw:clamp(-5,5);
		--update parameter models
		pm.Q_embedding_net:backward(Q_embedding_net.dw);
		pm.Q_encoder_net:backward(Q_encoder_net.dw);
		pm.multimodal_net:backward(multimodal_net.dw);
	end
	f=f/ntimes;
	pm.Q_embedding_net.dw:div(ntimes);
	pm.Q_encoder_net.dw:div(ntimes);
	pm.multimodal_net.dw:div(ntimes);
	return f;
end
jt=nn.JoinTable(2):cuda();
--Network forward for evaluation
function Forward(s,e,T)
	local fv_Q,fv_I=dataset:batch_eval(s,e);
	--Store RNG
	local rng1=torch.getRNGState():clone();
	local rng2=cutorch.getRNGState():clone();
	local scores={};
	for t=1,T do
		--sample parameter--
		torch.manualSeed(seeds[t][1]);
		cutorch.manualSeed(seeds[t][2]);
		Q_embedding_net.w[{}]=pm.Q_embedding_net:forward();
		Q_encoder_net.w[{}]=pm.Q_encoder_net:forward();
		multimodal_net.w[{}]=pm.multimodal_net:forward();
		--Forward
		local embedding_Q=Q_embedding_net.deploy:forward(fv_Q.words);
		local state_Q,_=Q_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_Q.map_to_sequence:size(1),1),embedding_Q,fv_Q.batch_sizes);
		local tv_Q=state_Q:index(1,fv_Q.map_to_sequence);
		local scores_t=logsoftmax:forward(multimodal_net.deploy:forward({tv_Q,fv_I}));
		scores[t]=torch.reshape(scores_t,scores_t:size(1),1,scores_t:size(2));
	end
	--Restore RNG
	torch.setRNGState(rng1);
	cutorch.setRNGState(rng2);
	return jt:forward(scores):double();
end
function Forward_test(s,e,T)
	--Store RNG
	local rng1=torch.getRNGState():clone();
	local rng2=cutorch.getRNGState():clone();
	local fv_Q,fv_I=dataset_test:batch_eval(s,e);
	local scores={};
	for t=1,T do
		--sample parameter--
		torch.manualSeed(seeds[t][1]);
		cutorch.manualSeed(seeds[t][2]);
		Q_embedding_net.w[{}]=pm.Q_embedding_net:forward();
		Q_encoder_net.w[{}]=pm.Q_encoder_net:forward();
		multimodal_net.w[{}]=pm.multimodal_net:forward();
		--Forward
		local embedding_Q=Q_embedding_net.deploy:forward(fv_Q.words);
		local state_Q,_=Q_encoder_net:forward(torch.repeatTensor(dummy_state:fill(0),fv_Q.map_to_sequence:size(1),1),embedding_Q,fv_Q.batch_sizes);
		local tv_Q=state_Q:index(1,fv_Q.map_to_sequence);
		local scores_t=logsoftmax:forward(multimodal_net.deploy:forward({tv_Q,fv_I}));
		scores[t]=torch.reshape(scores_t,scores_t:size(1),1,scores_t:size(2));
	end
	--Restore RNG
	torch.setRNGState(rng1);
	cutorch.setRNGState(rng2);
	return jt:forward(scores):double();
end
--Math functions
function logsumexp(s,dim)
	if dim==nil then
		local tmp = torch.max(s);
		return torch.log(torch.sum(torch.exp(s-tmp))) + tmp;
	else
		local tmp = torch.max(s,dim);
		return torch.log(torch.sum(torch.exp(s-tmp:expandAs(s)),dim)) + tmp;
	end
end
function logmeanexp(s,dim)
	if dim==nil then
		local tmp = torch.max(s);
		return torch.log(torch.mean(torch.exp(s-tmp))) + tmp;
	else
		local tmp = torch.max(s,dim);
		return torch.log(torch.mean(torch.exp(s-tmp:expandAs(s)),dim)) + tmp;
	end
end
--Our fast mutual information approximation
--The intuition is to find a "kernel expansion" for mutual information, so a dot product of two vectors computes the mutual information between two variables.
function mike(s)
	local p_a_qi_theta=s:cuda();
	local ntrials=s:size(1);
	local noutput=s:size(2);
	p_a_qi_theta_exp=torch.exp(p_a_qi_theta);
	p_a_qi=logmeanexp(p_a_qi_theta,1);
	mi_exp=p_a_qi_theta_exp*torch.exp(torch.csub(p_a_qi_theta,p_a_qi:expandAs(p_a_qi_theta))):t()/ntrials;
	mi_exp=mi_exp:view(-1);
	return mi_exp;
end
function rankmi(s,K)
	local avgmiv=torch.zeros(K*K):cuda()
	for i=1,s:size(1) do
		avgmiv:add(mike(s[i][{{1,K}}]));
	end
	avgmiv:div(s:size(1));
	local mi=torch.zeros(s:size(1));
	for i=1,s:size(1) do
		mi[i]=(torch.dot(mike(s[i][{{1,K}}]),avgmiv)-1)/2;
	end
	return mi;
end
function entropy(s)
	local K=s:size(2);
	local ent=torch.zeros(s:size(1));
	for i=1,s:size(1) do
		local p=logmeanexp(s[i][{{1,K}}]:cuda(),1):view(-1);
		ent[i]=-torch.cmul(torch.exp(p),p):sum()
	end
	return ent;
end
function curiosity(s)
	local curiosity=torch.zeros(s:size(1));
	local K=s:size(2);
	for i=1,s:size(1) do
		local pcond=s[i][{{1,K}}]:cuda();
		local p=logmeanexp(s[i][{{1,K}}]:cuda(),1):view(-1);
		curiosity[i]=torch.cmul(torch.exp(pcond-math.log(K)),pcond-p:repeatTensor(K,1)):sum();
	end
	return curiosity;
end
function rank_corr(v1,v2)
	local tmp=torch.zeros(v1:size(1),v1:size(1));
	local tmp1=v1:view(-1,1):repeatTensor(1,v1:size(1));
	local tmp2=v1:view(1,-1):repeatTensor(v1:size(1),1);
	local tmp3=v2:view(-1,1):repeatTensor(1,v1:size(1));
	local tmp4=v2:view(1,-1):repeatTensor(v1:size(1),1);
	local a=tmp1:gt(tmp2):double();
	local a=a*2-1;
	local b=tmp3:gt(tmp4):double();
	local b=b*2-1;
	return torch.dot(a,b)/a:norm(2)/b:norm(2);
end
--Computes mutual information between x and y using definition. Input p(x|theta) and p(y|theta) as x1 and x2.
function mi(s1,s2)
	local p_a_qi_theta_a=s1;
	local p_a_qi_theta_b=s2;
	local ntrials=s1:size(1);
	local noutput=s1:size(2);
	local pab=torch.repeatTensor(p_a_qi_theta_a:reshape(ntrials,noutput,1),1,1,noutput)+torch.repeatTensor(p_a_qi_theta_b:reshape(ntrials,1,noutput),1,noutput,1);
	pab=logmeanexp(pab,1):reshape(noutput,noutput);
	local pa=logmeanexp(p_a_qi_theta_a,1):reshape(noutput);
	local pb=logmeanexp(p_a_qi_theta_b,1):reshape(noutput);
	return -torch.sum(torch.cmul(pa,torch.exp(pa)))-torch.sum(torch.cmul(pb,torch.exp(pb)))+torch.sum(torch.cmul(pab,torch.exp(pab)));
end


--Initialize mask
npts=math.ceil(dataset.question.tokens:size(1));
npts_test=math.ceil(dataset_test.question.tokens:size(1));
mask=torch.randperm(npts)[{{1,params.init}}]:clone():long();
--mask=torch.range(1,params.init):long();
torch.save(paths.concat(basedir,'mask',string.format('mask%d.t7',1)),mask);
--Active learning loop
local timer = torch.Timer();
winit={};
winit.Q_embedding_net=pm.Q_embedding_net.w:clone();
winit.Q_encoder_net=pm.Q_encoder_net.w:clone();
winit.multimodal_net=pm.multimodal_net.w:clone();
for iter=1,params.maxIter do
	--grab new seeds
	for i=1,params.ntrials do
		seeds[i]={};
		seeds[i][1]=torch.random(1048576);
		seeds[i][2]=torch.random(1048576);
		seeds[i][3]=torch.random(1048576);
	end
	--train model
	if params.reset then
		pm.Q_embedding_net.w[{}]=winit.Q_embedding_net;
		pm.Q_encoder_net.w[{}]=winit.Q_encoder_net;
		pm.multimodal_net.w[{}]=winit.multimodal_net;
	end
	--set up optimization
	Log(string.format('Iter %d',iter));
	--Optimization
	local niter_per_epoch=math.max(math.ceil(mask:size(1)/params.batch/params.ntimes),5);
	local max_iter;
	if iter==1 then 
		max_iter=params.init_epochs*niter_per_epoch;
	else 
		max_iter=params.epochs*niter_per_epoch; 
	end
	local decay=math.exp(math.log(0.1)/params.decay/niter_per_epoch);
	local running_avg=nil;
	local opt={};
	opt.Q_encoder_net={learningRate=params.lr,decay=decay};
	opt.Q_embedding_net={learningRate=params.lr,decay=decay};
	opt.multimodal_net={learningRate=params.lr,decay=decay};
	Q_encoder_net:training();
	Q_embedding_net.deploy:training();
	multimodal_net.deploy:training();
	--optimization loop
	for i=1,max_iter do
		local f=ForwardBackward(params.batch,mask,params.ntimes);
		rmsprop(pm.Q_embedding_net.w,pm.Q_embedding_net.dw,opt.Q_embedding_net);
		rmsprop(pm.Q_encoder_net.w,pm.Q_encoder_net.dw,opt.Q_encoder_net); 
		rmsprop(pm.multimodal_net.w,pm.multimodal_net.dw,opt.multimodal_net);
		opt.Q_embedding_net.learningRate=opt.Q_embedding_net.learningRate*opt.Q_embedding_net.decay;
		opt.Q_encoder_net.learningRate=opt.Q_encoder_net.learningRate*opt.Q_encoder_net.decay;
		opt.multimodal_net.learningRate=opt.multimodal_net.learningRate*opt.multimodal_net.decay;
		running_avg=running_avg or f;
		running_avg=running_avg*0.95+f*0.05;
		if i%(niter_per_epoch*1)==0 then
			print(string.format('epoch %d/%d, trainloss %f, learning rate %f, time %f',i/niter_per_epoch,params.epochs,running_avg,opt.Q_embedding_net.learningRate,timer:time().real));
		end
	end
	torch.save(paths.concat(basedir,'model',string.format('model%d.t7',iter)),{Q_embedding_net=pm.Q_embedding_net.w,Q_encoder_net=pm.Q_encoder_net.w,multimodal_net=pm.multimodal_net.w,trainloss=running_avg});
	Log(string.format('\tt_model %f',timer:time().real));
	
	--compute score
	Q_encoder_net:evaluate();
	Q_embedding_net.deploy:evaluate();
	multimodal_net.deploy:evaluate();
	
	--compute score and add examples
	nsel=math.min(params.inc,npts-mask:size(1));
	local trainloss=torch.zeros(npts);
	local testloss=torch.zeros(npts_test);
	local pred_train=torch.LongTensor(npts);
	local pred_test=torch.LongTensor(npts_test);
	local train_score;
	if params.method=='curiosity' then
		--compute testing predictions
		for i=1,npts_test,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts_test);
			if i%31==0 then
				print(string.format('\ttesting test %d/%d %f',i,npts_test,timer:time().real));
			end
			local scores=Forward_test(i,r,params.ntrials);
			testloss[{{i,r}}],pred_test[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		--Add training examples that maximizes curiosity
		local mi_model_train=torch.zeros(npts);
		for i=1,npts,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts);
			if i%31==0 then
				print(string.format('\ttesting train %d/%d %f',i,npts,timer:time().real));
			end
			local scores=Forward(i,r,params.ntrials);
			mi_model_train[{{i,r}}]=curiosity(scores);
			trainloss[{{i,r}}],pred_train[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		train_score=mi_model_train;
	elseif params.method=='entropy' then
		--compute testing predictions
		for i=1,npts_test,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts_test);
			if i%31==0 then
				print(string.format('\ttesting test %d/%d %f',i,npts_test,timer:time().real));
			end
			local scores=Forward_test(i,r,params.ntrials);
			testloss[{{i,r}}],pred_test[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		--Add training examples that maximizes entropy
		local entropy_train=torch.zeros(npts);
		for i=1,npts,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts);
			if i%31==0 then
				print(string.format('\ttesting train %d/%d %f',i,npts,timer:time().real));
			end
			local scores=Forward(i,r,params.ntrials);
			entropy_train[{{i,r}}]=entropy(scores);
			trainloss[{{i,r}}],pred_train[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		train_score=entropy_train;
	elseif params.method=='random' then
		--compute testing predictions
		for i=1,npts_test,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts_test);
			if i%31==0 then
				print(string.format('\ttesting test %d/%d %f',i,npts_test,timer:time().real));
			end
			local scores=Forward_test(i,r,params.ntrials);
			testloss[{{i,r}}],pred_test[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		--Add training examples randomly
		--Evaluate train just because other methods do
		for i=1,npts,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts);
			if i%31==0 then
				print(string.format('\ttesting train %d/%d %f',i,npts,timer:time().real));
			end
			local scores=Forward(i,r,params.ntrials);
			trainloss[{{i,r}}],pred_train[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		train_score=torch.zeros(npts):uniform(0,1);
	elseif params.method=='goal_avg' then
		--avg mi kernel for test
		local avgmiv=torch.zeros(params.ntrials*params.ntrials):cuda();
		--compute test predictions and average mi kernel vector on the fly
		for i=1,npts_test,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts_test);
			if i%31==0 then
				print(string.format('\ttesting test %d/%d %f',i,npts_test,timer:time().real));
			end
			local scores=Forward_test(i,r,params.ntrials);
			for j=1,r-i+1 do
				avgmiv:add(mike(scores[j]));
			end
			testloss[{{i,r}}],pred_test[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		avgmiv:div(npts_test);
		--use avg mi kernel on test to compute mi criterion for train
		local mi_train=torch.zeros(npts);
		for i=1,npts,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts);
			if i%31==0 then
				print(string.format('\ttesting train %d/%d %f',i,npts,timer:time().real));
			end
			local scores=Forward(i,r,params.ntrials);
			for j=1,r-i+1 do
				mi_train[i+j-1]=(torch.dot(avgmiv,mike(scores[j]))-1)/2; --computes goal-driven learning score for train
			end
			trainloss[{{i,r}}],pred_train[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		train_score=mi_train;
	elseif params.method=='goal_max' then
		--Store all mi kernels for test
		local miv=torch.CudaTensor(npts_test,params.ntrials*params.ntrials):zero();
		--compute testing predictions and average mi kernel vector on the fly
		for i=1,npts_test,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts_test);
			if i%31==0 then
				print(string.format('\ttesting test %d/%d %f',i,npts_test,timer:time().real));
			end
			local scores=Forward_test(i,r,params.ntrials);
			for j=1,r-i+1 do
				miv[i+j-1]=mike(scores[j]);
			end
			testloss[{{i,r}}],pred_test[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		--use test mi kernel to compute mi criterion for train. Pick max mi as goal-driven learning score.
		local mi_train=torch.zeros(npts);
		for i=1,npts,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts);
			if i%31==0 then
				print(string.format('\ttesting train %d/%d %f',i,npts,timer:time().real));
			end
			local scores=Forward(i,r,params.ntrials);
			for j=1,r-i+1 do
				local miv_j=mike(scores[j]);
				local mi_j=(miv*miv_j:view(-1,1)-1)/2;
				mi_train[i+j-1]=mi_j:max();
			end
			trainloss[{{i,r}}],pred_train[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		train_score=mi_train;
	elseif params.method=='mi_train' then
		--Only test prediction
		for i=1,npts_test,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts_test);
			if i%31==0 then
				print(string.format('\ttesting test %d/%d %f',i,npts_test,timer:time().real));
			end
			local scores=Forward_test(i,r,params.ntrials);
			testloss[{{i,r}}],pred_test[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		--Store all mi kernels for train
		local avgmiv_train=torch.zeros(params.ntrials*params.ntrials):cuda();
		for i=1,npts,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts);
			if i%31==0 then
				print(string.format('\ttesting train #1 %d/%d %f',i,npts,timer:time().real));
			end
			local scores=Forward(i,r,params.ntrials);
			for j=1,r-i+1 do
				avgmiv_train:add(mike(scores[j]));
			end
			trainloss[{{i,r}}],pred_train[{{i,r}}]=logmeanexp(scores:cuda(),2):double():reshape(r-i+1,params.noutput)[{{},{1,params.nanswers}}]:max(2);
		end
		avgmiv_train:div(npts);
		--use train mi kernel to compute an mi criterion for train.
		--The intuition is to find training examples that best improves training accuracy
		local mi_train=torch.zeros(npts);
		for i=1,npts,params.batch_eval do
			r=math.min(i+params.batch_eval-1,npts);
			if i%31==0 then
				print(string.format('\ttesting train #2 %d/%d %f',i,npts,timer:time().real));
			end
			local scores=Forward(i,r,params.ntrials);
			for j=1,r-i+1 do
				mi_train[i+j-1]=(torch.dot(avgmiv_train,mike(scores[j]))-1)/2;
			end
		end
		train_score=mi_train;
	end
	
	--Selection criterion information
	tmp,_=torch.sort(train_score,1,true);
	avg=train_score:mean();
	top1=tmp:max();
	topK=tmp[{{1,nsel}}]:mean();
	--Make selection
	train_score:scatter(1,mask:long(),-1e10);
	_,tmp=torch.sort(train_score,1,true);
	sel=tmp[{{1,nsel}}]:long()
	--Accuracy
	--pred_train[pred_train:eq(params.nanswers+1)]=0;
	--pred_test[pred_test:eq(params.nanswers+1)]=0;
	local correct_count_train=torch.repeatTensor(pred_train:view(-1,1),1,dataset.question.labels:size(2)):eq(dataset.question.labels[{{1,npts},{}}]):double():sum(2);
	local correct_count_test=torch.repeatTensor(pred_test:view(-1,1),1,dataset_test.question.labels:size(2)):eq(dataset_test.question.labels[{{1,npts_test},{}}]):double():sum(2);
	
	acc_train=torch.cmin(correct_count_train:double()/3,1):mean(); --0.33/0.66/1 for matching 1,2,3+ human answers out of 10.
	acc_test=torch.cmin(correct_count_test:double()/3,1):mean();
	
	acc_train_loo=torch.cmin(correct_count_train:double()*0.3,1):mean(); --0.3/0.6/0.9/1 for matching 1,2,3,4+ human answers out of 10.
	acc_test_loo=torch.cmin(correct_count_test:double()*0.3,1):mean();
	
	Log(string.format('\tt_score %f',timer:time().real));
	Log(string.format('\tOPT INFO'));
	Log(string.format('\trunning_loss %f',running_avg));
	Log(string.format('\ttrainloss %f',-trainloss:mean()));
	Log(string.format('\ttestloss %f',-testloss:mean()));
	Log(string.format('\ttrainacc %f',acc_train));
	Log(string.format('\ttestacc %f',acc_test));
	
	Log(string.format('\ttrainacc_loo %f',acc_train_loo));
	Log(string.format('\ttestacc_loo %f',acc_test_loo));
	
	Log(string.format('\tCRITERION INFO'));
	Log(string.format('\tmax %f',top1));
	Log(string.format('\ttop%d %f',params.inc,topK));
	Log(string.format('\trandom %f',avg));
	
	mask=torch.cat(mask,sel,1);
	torch.save(paths.concat(basedir,'mask',string.format('mask%d.t7',iter+1)),mask);
	collectgarbage();
end
