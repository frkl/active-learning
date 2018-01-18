--NN weight generators
--PM:new(number_of_params,initial_weights)

require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
local PM={};
local function unit_bn(n)
	local noise=nn.Identity()();
	local vec=nn.Identity()();
	local std=nn.Square()(nn.Narrow(1,1,n)(vec));
	local logstd=nn.Log()(std);
	--local logstd=nn.AddConstant(-3)(nn.MulConstant(1)(nn.Narrow(1,1,n)(vec)));
	--local std=nn.Exp()(logstd);
	local mean=nn.Narrow(1,n+1,n)(vec);
	local odd=nn.CMulTable(){noise,std};
	local result=nn.CAddTable(){odd,mean};
	local log_ent=nn.MulConstant(-1)(nn.Mean(1)(logstd));
	local log_prior=nn.MulConstant(-1/2)(nn.Mean(1)(nn.Square()(result)));
	local kl=nn.CSubTable(){log_ent,log_prior};
	return nn.gModule({noise,vec},{result,kl});
end
local function unit_bernoulli(n)
	local vec=nn.Identity()();
	local result=nn.Dropout(0.5)(vec);
	return nn.gModule({vec},{result});
end
--Creates an object
function PM:new(n,w)
	local net={};
	local netmeta={};
	netmeta.__index = PM;
	setmetatable(net,netmeta);
	--nets
	net.n=n;
	if w then
		net.w=w:clone():cuda();
	else
		net.w=torch.CudaTensor(n):uniform(-0.01,0.01);
	end
	net.dw=net.w:clone();
	net.net=unit_bernoulli(n):cuda();
	return net;
end
function PM:training()
	self.net:training();
end
function PM:evaluate()
	self.net:evaluate();
end
function PM:zero_gradient()
	self.dw:zero();
end
function PM:forward()
	local wt=self.net:forward(self.w);
	return wt;
end
function PM:backward(dw)
	local dwt=self.net:backward(self.w,dw);
	self.dw:add(dwt);
end

return PM;