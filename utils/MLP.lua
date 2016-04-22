require 'nn'
local function mlp(nlayers,ninput,nh,noutput,gpu)
	gpu=gpu or false;
	local net=nn.Sequential();
	for i=1,nlayers-1 do
		if i==1 then
			net=net:add(nn.Linear(ninput,nh)):add(nn.ReLU()):add(nn.Dropout(0.5));
		else
			net=net:add(nn.Linear(nh,nh)):add(nn.ReLU()):add(nn.Dropout(0.5));
		end
	end
	if nlayers==1 then
		net=net:add(nn.Linear(ninput,noutput));
	else
		net=net:add(nn.Linear(nh,noutput));
	end
	if gpu then
		net=net:cuda();
	end
	local net_s={net=net};
	net_s.w,net_s.dw=net_s.net:getParameters();
	net_s.deploy=net_s.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std');
	return net_s;
end
return mlp