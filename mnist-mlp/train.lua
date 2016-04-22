cmd = torch.CmdLine();
cmd:text('Actively learn an MLP');
cmd:text('Strategy')
cmd:option('-method','mi','Active learning strategy. {random, mi, mi-decorr, entropy, confidence, confidence-2}');
cmd:option('-init',50,'Initial samples');
cmd:option('-inc',10,'How many examples to select each round');
cmd:option('-maxIter',200,'How many rounds');
cmd:option('-ntrials',30,'How many dropout samples');
cmd:option('-seed',1234,'random seed');
cmd:text('Dataset')
cmd:option('-data','../../dataset/mnist/data_train.t7','Dataset for training');
cmd:text('Model parameters');
cmd:option('-nlayers',3,'RNN layers');
cmd:option('-nh',1200,'RNN size');
cmd:option('-noutput',10,'RNN layers');
cmd:text('Optimization parameters');
cmd:option('-batch',256,'Batch size (Adjust base on GRAM and dataset size)');
cmd:option('-lr',1e-4,'Learning rate');
cmd:option('-decay',150,'Learning rate decay in epochs');
cmd:option('-l2',0.01,'L2 regularizer');
cmd:option('-epochs',300,'Epochs');
params=cmd:parse(arg);


require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'
require '../utils/optim_updates'

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
data=torch.load(params.data);
params.ninput=data['data']:size(2);
print('Building network');
net=require('../utils/MLP')(params.nlayers,params.ninput,params.nh,params.noutput,true);
criterion=nn.CrossEntropyCriterion():cuda();
logsoftmax=nn.LogSoftMax():cuda();
print('Initializing session');
paths.mkdir('sessions')
Session=require('../utils/session_manager');
session=Session:init('./sessions');
basedir=session:new(params);
paths.mkdir(paths.concat(basedir,'model'));
paths.mkdir(paths.concat(basedir,'mask'));

--Rewrite dropout forward for testing
function nn.Dropout:updateOutput(input)
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
	return self.output
end
--Data provider
function data:batch_train(batch_size,mask)
	local fv=torch.DoubleTensor(batch_size,self.data:size(2));
	local label=torch.DoubleTensor(batch_size);
	for i=1,batch_size do
		--randomly select an item
		local id=mask[torch.random(mask:size(1))];		
		label[i]=self.label[id];
		fv[i]=self.data[id];
	end
	return fv:cuda(),label:cuda();
end
function data:batch_eval(s,e)
	return self.data[{{s,e}}]:cuda(),self.label[{{s,e}}]:cuda();
end
--Network forward backward
function ForwardBackward(batch_size,mask)
	net.dw:zero();
	local fv,label=data:batch_train(batch_size,mask);
	local output=net.deploy:forward(fv);
	local f=criterion:forward(output,label);
	local doutput=criterion:backward(output,label);
	net.deploy:backward(fv,doutput);
	net.dw:add(params.l2/mask:size(1),net.w);
	return f;
end
--Network forward for evaluation
function Forward(s,e,t)
	local fv,label=data:batch_eval(s,e);
	torch.manualSeed(seeds[t][1]);
	cutorch.manualSeedAll(seeds[t][1]);
	local output=net.deploy:forward(fv);
	return logsoftmax:forward(output):clone();
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
npts=data.data:size(1);
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
	local opt={};
	opt.maxIter=niter_per_epoch*params.epochs;
	opt.learningRate=params.lr;
	opt.running_avg=0;
	opt.decay=math.exp(math.log(0.1)/params.decay/niter_per_epoch);
	net.deploy:training();
	--optimization loop
	for i=1,opt.maxIter do
		local f=ForwardBackward(params.batch,mask);
		rmsprop(net.w,net.dw,opt);
		opt.learningRate=opt.learningRate*opt.decay;
		if i==1 then
			opt.running_avg=f;
		else
			opt.running_avg=opt.running_avg*0.95+f*0.05;
		end
	end
	torch.save(paths.concat(basedir,'model',string.format('model%d.t7',iter)),{net=net.net,trainloss=opt.running_avg});
	Log(string.format('\tt_model %f',timer:time().real));
	
	--compute score
	net.deploy:evaluate();
	for i=1,npts,params.batch do
		r=math.min(i+params.batch-1,npts);
		for t=1,params.ntrials do
			scores[{{i,r},t,{}}]=Forward(i,r,t):double();
		end
	end
	_,pred=torch.max(logmeanexp(scores,2):reshape(npts,params.noutput),2);
	acc=pred:eq(data.label:long()):sum()/pred:size(1);
	Log(string.format('\tt_score %f',timer:time().real));
	Log(string.format('\ttrainloss %f',opt.running_avg));
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
