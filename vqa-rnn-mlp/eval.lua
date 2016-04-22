cmd = torch.CmdLine();
cmd:text('Actively learn a VQA model');
cmd:text('Strategy')
cmd:option('-session','','The folder name of the session');
cmd:option('-ntrials',30,'How many dropout samples');
cmd:option('-seed',1234,'random seed');
cmd:text('Dataset')
cmd:option('-data','../../vqakb/dataset/imqa/dataset_val.t7','Dataset for eval');
params=cmd:parse(arg);





require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
require '../utils/RNNUtils'
function AxB(nhA,nhB,nhcommon,noutput,dropout)
	dropout = dropout or 0 
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(nn.Normalize(2)(i))));
	local output=nn.Linear(nhcommon,noutput)(nn.Dropout(0.5)(nn.CMulTable()({qc,ic})));
	return nn.gModule({q,i},{output});
end

--initialize a set of seeds
torch.manualSeed(params.seed);
cutorch.manualSeedAll(params.seed);
seeds={};
for i=1,params.ntrials do
	seeds[i]={};
	seeds[i][1]=torch.random(1048576);
	seeds[i][2]=torch.random(1048576);
	seeds[i][3]=torch.random(1048576);
end

print('Loading dataset');
dataset=torch.load(params.data);

print('Right aligning words');
dataset['question_lengths']=sequence_length(dataset['question_tokens']);
dataset['question_tokens']=right_align(dataset['question_tokens'],dataset['question_lengths']);
collectgarbage();

print('Initializing session');
local cjson=require('cjson');
local function readAll(file)
    local f = io.open(file, "r")
	if f==nil then
		error({msg='Failed to open file',file=file});
	end
    local content = f:read("*all");
    f:close()
    return content;
end
local function loadJson(fname)
	local t=readAll(fname);
	return cjson.decode(t);
end
basedir=paths.concat('./sessions',params.session);
params_train=loadJson(paths.concat(basedir,'_session_config.json'));
logsoftmax=nn.LogSoftMax():cuda();
--Create dummy gradients
nhdummy=1;
dummy_state=torch.DoubleTensor(params_train.nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(nhdummy):fill(0):cuda();



--Rewrite dropout forward for testing
function nn.Dropout:updateOutput(input)
	local seed=torch.random();
	if self.inplace then
		self.output = input
	else
		self.output:resizeAs(input):copy(input)
	end
	if self.p > 0 then
		if self.train then
			--stock dropout during training
			self.noise:resizeAs(input)
			self.noise:bernoulli(1-self.p)
			if self.v2 then
				self.noise:div(1-self.p)
			end
			self.output:cmul(self.noise)
		else
			--correlated dropout during testing
			local tmp=input:size();
			tmp[1]=1;
			self.noise:resize(tmp)
			self.noise:bernoulli(1-self.p)
			if self.v2 then
				self.noise:div(1-self.p)
			end
			self.noise=self.noise:expandAs(input)
			self.output:cmul(self.noise)
		end
	end
	torch.manualSeed(seed);
	cutorch.manualSeedAll(seed);
	return self.output
end
function dataset:batch_eval(s,e)
	local timer = torch.Timer();
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		iminds[i]=dataset['question_imids'][qinds[i]];
	end
	local fv_sorted_q=sort_encoding_lookup_right_align(dataset['question_tokens']:index(1,qinds),dataset['question_lengths']:index(1,qinds),vocabulary_size_q);
	fv_sorted_q.words=fv_sorted_q.words:cuda();
	fv_sorted_q.map_to_rnn=fv_sorted_q.map_to_rnn:cuda();
	fv_sorted_q.map_to_sequence=fv_sorted_q.map_to_sequence:cuda();
	
	local fv_im=dataset['image_fvs']:index(1,iminds);
	return fv_sorted_q,fv_im:cuda(),qids;
end
--dem math
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

log_file=paths.concat(basedir,string.format('log_test.txt',1));
function Log(msg)
	local f = io.open(log_file, "a")
	print(msg);
	f:write(msg..'\n');
	f:close()
end
print(params);
print(params_train);

--Initialize mask
npts=dataset['question_tokens']:size(1);
--Active learning loop
scores=torch.DoubleTensor(npts,params_train.ntrials,params_train.noutput):fill(0);
local timer = torch.Timer();
for iter=1,params_train.maxIter do
	Log(string.format('Iter %d',iter));
	--load network
	local tmp=torch.load(paths.concat(basedir,'model',string.format('model%d.t7',iter)));
	local embedding_net={};
	local encoder_net={};
	local multimodal_net={};
	embedding_net.deploy=tmp.embedding_net;
	encoder_net.deploy=dupe_rnn(tmp.encoder_net,dataset['question_tokens']:size(2));
	multimodal_net.deploy=tmp.multimodal_net;
	--Network forward for evaluation
	function Forward(s,e,t)
		local fv_q,fv_im,qids,batch_size=dataset:batch_eval(s,e);
		local question_max_length=fv_q.batch_sizes:size(1);
		torch.manualSeed(seeds[t][1]);
		cutorch.manualSeedAll(seeds[t][1]);
		local word_embedding_q=embedding_net.deploy:forward(fv_q.words);
		local states_q,_=rnn_forward_dropout(encoder_net.deploy,torch.repeatTensor(dummy_state,e-s+1,1),word_embedding_q,fv_q.batch_sizes,seeds[t][2]);
		local tv_q=states_q[question_max_length+1]:index(1,fv_q.map_to_sequence);
		torch.manualSeed(seeds[t][3]);
		cutorch.manualSeedAll(seeds[t][3]);
		local scores=multimodal_net.deploy:forward({tv_q,fv_im});
		return logsoftmax:forward(scores):clone();
	end
	--compute score
	embedding_net.deploy:training();
	for i=1,#encoder_net.deploy do
		encoder_net.deploy[i]:training();
	end
	multimodal_net.deploy:training();
	for i=1,npts,params_train.batch do
		if i%(params_train.batch*10)==1 then
			print(string.format('Iter %d, time %f',i,timer:time().real));
		end
		r=math.min(i+params_train.batch-1,npts);
		for t=1,params.ntrials do
			scores[{{i,r},t,{}}]=Forward(i,r,t):double();
		end
	end
	_,pred=torch.max(logmeanexp(scores,2):reshape(npts,params_train.noutput),2);
	acc=pred:eq(dataset.answer_labels:long()):sum()/pred:size(1);
	Log(string.format('\tt_score %f',timer:time().real));
	Log(string.format('\tacc %f',acc));
end