--Wraps a network to expose the parameters and gradients as vectors.
--Also create a "deploy" copy that stores the intermediate variables, so the original network is clean and small for torch.save.

local Net={};

--Set everything to training mode
function Net:training()
	self.net:training();
	if self.n then
		for i=1,self.n do self.deploy[i]:training(); end
	else
		self.deploy:training();
	end
end

--Set everything to testing mode
function Net:evaluate()
	self.net:evaluate();
	if self.n then
		for i=1,self.n do self.deploy[i]:evaluate(); end
	else
		self.deploy:evaluate();
	end
end

--Constructor
--net: An instance of the original network
--gpu: Using cuda or not
--Returns a Net object
--    Net.w are the parameters
--    Net.dw are the gradient parameters
--    Net.net is a copy of the original network
--    Net.deploy is a copy of .net sharing the same parameters and gradient parameters as .net.
function Net:wrap(net,gpu)
	local net_obj={};
	local netmeta={};
	netmeta.__index = Net;
	setmetatable(net_obj,netmeta);
	if gpu then
		net_obj.net=net:cuda();
	else
		net_obj.net=net;
	end
	net_obj.w,net_obj.dw=net_obj.net:getParameters();
	net_obj.deploy=net_obj.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
	return net_obj;
end


--Constructor where there's need to create n copies (more than 1) of the deploy network.
--For example when the same LookupTable is shared in multiple places. 
--net: An instance of the original network
--n: How many deploy copies to use
--gpu: Using cuda or not
--Returns a Net object
--    Net.w are the parameters
--    Net.dw are the gradient parameters
--    Net.net is a copy of the original network
--    Net.deploy is a table of copies of .net sharing the same parameters and gradient parameters as .net.
function Net:wrap_4Head(net,n,gpu)
	local net_obj={};
	local netmeta={};
	netmeta.__index = Net;
	setmetatable(net_obj,netmeta);
	if gpu then
		net_obj.net=net:cuda();
	else
		net_obj.net=net;
	end
	net_obj.w,net_obj.dw=net_obj.net:getParameters();
	net_obj.deploy={};
	net_obj.n=n;
	for i=1,n do
		net_obj.deploy[i]=net_obj.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
	end
	return net_obj;
end

return Net;