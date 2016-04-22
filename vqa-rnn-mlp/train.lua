cmd = torch.CmdLine();
cmd:text('Actively learn a VQA model');
cmd:text('Strategy')
cmd:option('-method','mi','Active learning strategy. {random, mi, mi-decorr, entropy, confidence, confidence-2}');
cmd:option('-init',50,'Initial samples');
cmd:option('-inc',10,'How many examples to select each round');
cmd:option('-maxIter',200,'How many rounds');
cmd:option('-ntrials',30,'How many dropout samples');
cmd:option('-seed',1234,'random seed');
cmd:text('Dataset')
cmd:option('-data','../../vqakb/dataset/imqa/dataset_train.t7','Dataset for training');
cmd:text('Model parameters');
cmd:option('-nhword',200,'Word embedding size');
cmd:option('-nh',512,'RNN size');
cmd:option('-nhcommon',1024,'Common embedding size');
cmd:option('-nlayers',2,'RNN layers');
cmd:text('Optimization parameters');
cmd:option('-batch',500,'Batch size (Adjust base on GRAM and dataset size)');
cmd:option('-lr',3e-4,'Learning rate');
cmd:option('-decay',150,'Learning rate decay in epochs');
cmd:option('-l2',0,'L2 regularizer');
cmd:option('-epochs',300,'Epochs');
params=cmd:parse(arg);












require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
require '../utils/optim_updates'
require '../utils/RNNUtils'
LSTM=require('../utils/LSTM');
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
params.nhsent=params.nlayers*params.nh*2;
params.question_dictionary_size=table.getn(dataset['question_dictionary']);
params.nhimage=dataset['image_fvs']:size(2);
params.noutput=table.getn(dataset['answer_dictionary']);

print('Right aligning words');
dataset['question_lengths']=sequence_length(dataset['question_tokens']);
dataset['question_tokens']=right_align(dataset['question_tokens'],dataset['question_lengths']);
collectgarbage();


print('Building network');
nhdummy=1;
embedding_net={};
encoder_net={};
multimodal_net={};

embedding_net.net=nn.LookupTable(params.question_dictionary_size,params.nhword):cuda();
encoder_net.net=LSTM.lstm_dropout(params.nhword,params.nh,nhdummy,params.nlayers,0.5):cuda();
multimodal_net.net=nn.Sequential():add(AxB(params.nhsent,params.nhimage,params.nhcommon,params.noutput,0.5)):cuda();

embedding_net.w,embedding_net.dw=embedding_net.net:getParameters();
encoder_net.w,encoder_net.dw=encoder_net.net:getParameters();
multimodal_net.w,multimodal_net.dw=multimodal_net.net:getParameters();
embedding_net.w:uniform(-0.08, 0.08);
encoder_net.w:uniform(-0.08, 0.08);

embedding_net.deploy=embedding_net.net:clone('weight','bias','gradWeight','gradBias');
encoder_net.deploy=dupe_rnn(encoder_net.net,dataset['question_tokens']:size(2));
multimodal_net.deploy=multimodal_net.net:clone('weight','bias','gradWeight','gradBias');

criterion=nn.CrossEntropyCriterion():cuda();
logsoftmax=nn.LogSoftMax():cuda();


--Create dummy gradients
dummy_state=torch.DoubleTensor(params.nhsent):fill(0):cuda();
dummy_output=torch.DoubleTensor(nhdummy):fill(0):cuda();




print('Initializing session');
paths.mkdir('sessions')
Session=require('../utils/session_manager');
session=Session:init('./sessions');
basedir=session:new(params);
paths.mkdir(paths.concat(basedir,'model'));
paths.mkdir(paths.concat(basedir,'mask'));

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
--Data provider
function dataset:batch_train(batch_size,mask)
	local timer = torch.Timer();
	local nqs=mask:size(1);
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=mask[torch.random(nqs)];
		iminds[i]=self['question_imids'][qinds[i]];
	end
	
	local fv_sorted_q=sort_encoding_lookup_right_align(self['question_tokens']:index(1,qinds),self['question_lengths']:index(1,qinds));
	fv_sorted_q.words=fv_sorted_q.words:cuda();
	fv_sorted_q.map_to_rnn=fv_sorted_q.map_to_rnn:cuda();
	fv_sorted_q.map_to_sequence=fv_sorted_q.map_to_sequence:cuda();
	local fv_im=self['image_fvs']:index(1,iminds);
	local labels=self['answer_labels']:index(1,qinds);
	return fv_sorted_q,fv_im:cuda(),labels:cuda();
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
--Network forward backward
function ForwardBackward(batch_size,mask)
	embedding_net.dw:zero();
	encoder_net.dw:zero();
	multimodal_net.dw:zero();
	local fv_q,fv_im,labels=dataset:batch_train(batch_size,mask);
	local question_max_length=fv_q.batch_sizes:size(1);
	--embedding forward--
	local word_embedding_q=embedding_net.deploy:forward(fv_q.words);
	--encoder forward--
	local states_q,_=rnn_forward_dropout(encoder_net.deploy,torch.repeatTensor(dummy_state:fill(0),batch_size,1),word_embedding_q,fv_q.batch_sizes,torch.random());
	--multimodal/criterion forward--
	local tv_q=states_q[question_max_length+1]:index(1,fv_q.map_to_sequence);
	local scores=multimodal_net.deploy:forward({tv_q,fv_im});
	local f=criterion:forward(scores,labels);
	--multimodal/criterion backward--
	local dscores=criterion:backward(scores,labels);
	local tmp=multimodal_net.deploy:backward({tv_q,fv_im},dscores);
	local dtv_q=tmp[1]:index(1,fv_q.map_to_rnn);
	--encoder backward
	local _,dword_embedding_q=rnn_backward(encoder_net.deploy,dtv_q,dummy_output,states_q,word_embedding_q,fv_q.batch_sizes);
	--embedding backward--
	embedding_net.deploy:backward(fv_q.words,dword_embedding_q);
	--summarize f and gradient
	encoder_net.dw:add(params.l2/mask:size(1),encoder_net.w);
	embedding_net.dw:add(params.l2/mask:size(1),embedding_net.w);
	multimodal_net.dw:add(params.l2/mask:size(1),multimodal_net.w);
	encoder_net.dw:clamp(-10,10);
	return f;
end
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
function mst(d)
	local nvtx=d:size(1);
	local connected=d:clone():zero();
	local total=0;
	local es=d:reshape(nvtx*nvtx);
	local nu,ind=torch.sort(es,1,true);
	local j=1;
	for i=1,nvtx-1 do
		local found=0;
		while found==0 do
			x=(ind[j]-1)%nvtx+1;
			y=(ind[j]-x)/nvtx+1;
			j=j+1;
			if x<y then
				if connected[x][y]==0 then
					found=1;
					total=total+nu[j];
					connected[x][y]=1;
					connected[y][x]=1;
					connected[x][x]=1;
					connected[y][y]=1;
					for m=1,nvtx do
						for n=1,nvtx do
							if connected[x][m]==1 and connected[y][n]==1 then
								connected[m][n]=1;
								connected[x][n]=1;
								connected[y][m]=1;
								connected[n][m]=1;
								connected[n][x]=1;
								connected[m][y]=1;
							elseif connected[x][m]==1 then
								connected[y][m]=1;
								connected[m][y]=1;
							elseif connected[y][n]==1 then
								connected[x][n]=1;
								connected[n][x]=1;
							end
						end
					end
				end
			end
		end
	end
	return total;
end

log_file=paths.concat(basedir,string.format('log.txt',1));
function Log(msg)
	local f = io.open(log_file, "a")
	print(msg);
	f:write(msg..'\n');
	f:close()
end
print(params);

--Initialize mask
npts=dataset['question_tokens']:size(1);
mask=torch.randperm(npts)[{{1,params.init}}]:clone():long();
torch.save(paths.concat(basedir,'mask',string.format('mask%d.t7',1)),mask);
--Active learning loop
scores=torch.DoubleTensor(npts,params.ntrials,params.noutput):fill(0);
local timer = torch.Timer();
for iter=1,params.maxIter do
	--train model
	--set up optimization
	Log(string.format('Iter %d',iter));
	niter_per_epoch=math.ceil(mask:size(1)/params.batch);
	local opt_embedding={};
	local opt_encoder={};
	local opt_multimodal={};
	opt_embedding.maxIter=niter_per_epoch*params.epochs;
	opt_embedding.learningRate=params.lr;
	opt_embedding.running_avg=0;
	opt_embedding.decay=math.exp(math.log(0.1)/params.decay/niter_per_epoch);
	opt_encoder.learningRate=params.lr;
	opt_multimodal.learningRate=params.lr;
	embedding_net.deploy:training();
	for i=1,#encoder_net.deploy do
		encoder_net.deploy[i]:training();
	end
	multimodal_net.deploy:training();
	--optimization loop
	for i=1,opt_embedding.maxIter do
		local f=ForwardBackward(params.batch,mask);
		rmsprop(embedding_net.w,embedding_net.dw,opt_embedding);
		rmsprop(encoder_net.w,encoder_net.dw,opt_encoder);
		rmsprop(multimodal_net.w,multimodal_net.dw,opt_multimodal);
		opt_embedding.learningRate=opt_embedding.learningRate*opt_embedding.decay;
		opt_encoder.learningRate=opt_encoder.learningRate*opt_embedding.decay;
		opt_multimodal.learningRate=opt_multimodal.learningRate*opt_embedding.decay;
		if i==1 then
			opt_embedding.running_avg=f;
		else
			opt_embedding.running_avg=opt_embedding.running_avg*0.95+f*0.05;
		end
		if i%(niter_per_epoch*1)==0 then
			print(string.format('epoch %d/%d, trainloss %f, learning rate %f, time %f',i/niter_per_epoch,params.epochs,opt_embedding.running_avg,opt_embedding.learningRate,timer:time().real));
		end
	end
	torch.save(paths.concat(basedir,'model',string.format('model%d.t7',iter)),{embedding_net=embedding_net.net,encoder_net=encoder_net.net,multimodal_net=multimodal_net.net,trainloss=opt_embedding.running_avg});
	Log(string.format('\tt_model %f',timer:time().real));
	
	--compute score
	embedding_net.deploy:evaluate();
	for i=1,#encoder_net.deploy do
		encoder_net.deploy[i]:evaluate();
	end
	multimodal_net.deploy:evaluate();
	for i=1,npts,params.batch do
		r=math.min(i+params.batch-1,npts);
		for t=1,params.ntrials do
			scores[{{i,r},t,{}}]=Forward(i,r,t):double();
		end
	end
	_,pred=torch.max(logmeanexp(scores,2):reshape(npts,params.noutput),2);
	acc=pred:eq(dataset.answer_labels:long()):sum()/pred:size(1);
	Log(string.format('\tt_score %f',timer:time().real));
	Log(string.format('\ttrainloss %f',opt_embedding.running_avg));
	Log(string.format('\ttrainacc %f',acc));
	
	--update mask
	entropy=torch.DoubleTensor(npts);
	expected_entropy=torch.DoubleTensor(npts);
	for i=1,npts do
		p_a_qi_theta=scores[{i,{},{}}]:cuda();
		p_a_qi=logmeanexp(p_a_qi_theta,1);
		entropy[i]=-torch.cmul(p_a_qi,torch.exp(p_a_qi)):sum();
		expected_entropy[i]=-torch.cmul(p_a_qi_theta,torch.exp(p_a_qi_theta)):mean(1):sum();
	end
	curiosity=entropy-expected_entropy;
	nu,ind=torch.sort(curiosity,1,true);
	sel={};
	nsel=math.min(params.inc,npts-mask:size(1));
	Log(string.format('\tmax_curiosity %f',curiosity:max()));
	Log(string.format('\ttop%d %f',params.inc,nu[{{1,nsel}}]:mean()));
	
	if params.method=='mi-decorr' then
		correlations={};
		ops=0;
		local curiosity_clone=curiosity:clone();
		curiosity_clone:index(1,mask):fill(-1e5);
		for i=1,nsel do
			candidate=-1;
			best_inc=-1;
			local nu,ind=torch.sort(curiosity_clone,1,true);
			for j=1,npts do
				max_correlation=0;
				for k=1,#sel do
					if correlations[(ind[j]-1)*npts+sel[k]]==nil then 
						tmp=mi(scores[{ind[j],{},{}}]:cuda(),scores[{sel[k],{},{}}]:cuda());
						ops=ops+1;
						correlations[(ind[j]-1)*npts+sel[k]]=tmp;
						correlations[(sel[k]-1)*npts+ind[j]]=tmp;
					end
					max_correlation=math.max(correlations[(ind[j]-1)*npts+sel[k]],max_correlation);
				end
				if nu[j]-max_correlation>best_inc then
					best_inc=nu[j]-max_correlation;
					candidate=ind[j];
				end
				if j<npts and best_inc>nu[j+1] then
					break;
				end
			end
			--print(string.format('Turn %d, selecting %d, original %f, gain %f, correlation %f',i,candidate,curiosity[candidate],best_inc,curiosity[candidate]-best_inc));
			table.insert(sel,candidate);
			curiosity_clone[candidate]=-1e5;
		end
		print(string.format('ops %d',ops));
		sel=torch.LongTensor(sel);
	elseif params.method=='confidence' then
		--per class, highest confidence
		if nsel%params.noutput~=0 then
			error('Cannot perform per-class selection');
		end
		local tmp=logmeanexp(scores,2):reshape(npts,params.noutput);
		for c=1,params.noutput do
			local nu,ind=torch.sort(tmp[{{},c}],1,true);
			local n=0;
			local t=1;
			while n<nsel/params.noutput do
				if c==1 and t==1 and mask:eq(ind[t]):sum()==0 then
					n=n+1;
					table.insert(sel,ind[t]);
				elseif torch.Tensor(sel):eq(ind[t]):sum()==0 and mask:eq(ind[t]):sum()==0 then
					n=n+1;
					table.insert(sel,ind[t]);
				end
				t=t+1;
			end
		end
		sel=torch.LongTensor(sel);
	elseif params.method=='mi' then
		tmp=curiosity:clone();
		tmp:index(1,mask):fill(-1e5);
		local nu,ind=torch.sort(tmp,1,true);
		i=1;
		while #sel<params.inc do
			if mask:eq(ind[i]):sum()==0 then
				if i>1 and torch.LongTensor(sel):eq(ind[i]):sum()==0 then
					table.insert(sel,ind[i]);
				elseif i==1 then
					table.insert(sel,ind[i]);
				end
			end
			i=i+1;
		end
		sel=torch.LongTensor(sel);
	elseif params.method=='entropy' then
		tmp=entropy:clone();
		tmp:index(1,mask):fill(-1e5);
		local nu,ind=torch.sort(tmp,1,true);
		i=1;
		while #sel<params.inc do
			if mask:eq(ind[i]):sum()==0 then
				if i>1 and torch.LongTensor(sel):eq(ind[i]):sum()==0 then
					table.insert(sel,ind[i]);
				elseif i==1 then
					table.insert(sel,ind[i]);
				end
			end
			i=i+1;
		end
		sel=torch.LongTensor(sel);
	elseif params.method=='random' then
		tmp=torch.rand(npts);
		tmp:index(1,mask):fill(-1e5);
		local nu,ind=torch.sort(tmp,1,true);
		i=1;
		while #sel<params.inc do
			if mask:eq(ind[i]):sum()==0 then
				if i>1 and torch.LongTensor(sel):eq(ind[i]):sum()==0 then
					table.insert(sel,ind[i]);
				elseif i==1 then
					table.insert(sel,ind[i]);
				end
			end
			i=i+1;
		end
		sel=torch.LongTensor(sel);
	else
		error('Unrecognized sample selection strategy.')
	end
	
	miab=torch.DoubleTensor(params.inc,params.inc);
	for i=1,params.inc do
	for j=1,params.inc do
		miab[i][j]=mi(scores[{sel[i],{},{}}]:cuda(),scores[{sel[j],{},{}}]:cuda());
	end
	end
	est_mi=mst(miab);
	Log(string.format('\tchosen %f',curiosity:index(1,sel):mean()));
	Log(string.format('\tchosen-corr %f',curiosity:index(1,sel):mean()-est_mi/params.inc));
	Log(string.format('\trandom %f',curiosity:mean()));
	
	mask=torch.cat(mask,sel,1);
	torch.save(paths.concat(basedir,'mask',string.format('mask%d.t7',iter+1)),mask);
	Log(string.format('\tt_mask %f',timer:time().real));
end
