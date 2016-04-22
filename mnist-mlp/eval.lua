cmd = torch.CmdLine();
cmd:text('Testing for a session');
cmd:text('Strategy')
cmd:option('-session','','The folder name of the session');
cmd:option('-ntrials',30,'How many dropout samples');
cmd:option('-seed',1234,'random seed');
cmd:text('Dataset')
cmd:option('-data','../../dataset/mnist/data_test.t7','Dataset for testing');
params=cmd:parse(arg);


require 'nn'
require 'cutorch'
require 'cunn' 
require 'nngraph'


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

function data:batch_eval(s,e)
	return self.data[{{s,e}}]:cuda(),self.label[{{s,e}}]:cuda();
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



--Compute scores
npts=data.data:size(1);
scores=torch.DoubleTensor(npts,params.ntrials,params_train.noutput):fill(0);
local timer = torch.Timer();
for iter=1,params_train.maxIter do
	Log(string.format('Iter %d',iter));
	--load network
	local net={};
	net=torch.load(paths.concat(basedir,'model',string.format('model%d.t7',iter)));
	--Network forward for evaluation
	function Forward(s,e,t)
		local fv,label=data:batch_eval(s,e);
		torch.manualSeed(seeds[t][1]);
		cutorch.manualSeedAll(seeds[t][1]);
		local output=net.net:forward(fv);
		return logsoftmax:forward(output):clone();
	end
	--compute score
	net.net:training();
	for i=1,npts,params_train.batch do
		r=math.min(i+params_train.batch-1,npts);
		for t=1,params.ntrials do
			scores[{{i,r},t,{}}]=Forward(i,r,t):double();
		end
	end
	_,pred=torch.max(logmeanexp(scores,2):reshape(npts,params_train.noutput),2);
	acc=pred:eq(data.label:long()):sum()/pred:size(1);
	Log(string.format('\tt_score %f',timer:time().real));
	Log(string.format('\tacc %f',acc));
end
