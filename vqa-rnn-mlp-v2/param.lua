--Tensor counterpart of net.lua. 
--Wraps a single tensor for w and dw. So we can work with a unified interface.
local Param={};

function Param:wrap(w,gpu)
	local net_obj={};
	local netmeta={};
	netmeta.__index = Param;
	setmetatable(net_obj,netmeta);
	if gpu then
		net_obj.w=w:cuda();
	else
		net_obj.w=w:clone();
	end
	net_obj.dw=net_obj.w:clone();
	return net_obj;
end


return Param;