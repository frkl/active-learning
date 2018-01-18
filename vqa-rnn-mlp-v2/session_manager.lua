--Provides some ability to manage runs
--  Creates a separate folder for each run
--  Store the parameters as a json file so it can be retrieved later
--  It is a work-in-progress partial implementation but useful

local cjson=require('cjson');
local torch=require('torch');
local paths=require('paths');
local dirname_length=12;
local Session={};
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
local function writeAll(file,data)
    local f = io.open(file, "w");
	if f==nil then
		error({msg='Failed to open file',file=file});
	end
    f:write(data);
    f:close();
end
local function saveJson(fname,t)
	return writeAll(fname,cjson.encode(t));
end
--hash a table of strings, numbers and nil
local function hash(t)
	return 0;
end
local function random_string(l,prefix)
	prefix=prefix or '';
	local s=prefix;
	local candidates='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
	for i=1,l do
		local ind=math.random(string.len(candidates));
		s=s..candidates:sub(ind,ind);
	end
	return s;
end
function Session:init(root)
	local session={};
	setmetatable(session, self);
    self.__index = self;
	session.root=root;
	return session;
end
--Create a new session
--Params: a table of parameters for this session. Only supports parameters that can be stored as json.
function Session:new(params)
	local dirname='';
	--Create a random directory
	while true do
		dirname=random_string(dirname_length,'session_');
		if paths.mkdir(paths.concat(self.root,dirname)) then
			break;
		end
	end
	--Write the parameters into the directory
	saveJson(paths.concat(self.root,dirname,'_session_config.json'),params);
	--Return the path to the directory
	return paths.concat(self.root,dirname);
end
--Find the dirname for a particular session. TODO
function Session:find(params)
	return paths.concat(self.root,'');
end
function Session:list()
	local dirs={};
	local params={};
	for dir in paths.iterdirs(self.root) do
		local param={};
		local root=self.root;
		if pcall(function () param=loadJson(paths.concat(root,dir,'_session_config.json')) end) then
			table.insert(dirs,dir);
			table.insert(params,param);
		else
			print(string.format('Warning: Could not find session in %s',dir));
		end
	end
	return {dirs=dirs,params=params};
end

--Load the configuration for a particular session by name.
function Session:get_params(session_name)
	return loadJson(paths.concat(self.root,session_name,'_session_config.json'));
end
return Session;
