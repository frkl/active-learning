cjson=require('cjson');
--------UTIL FUNCTIONS---------------
--Takes a table and return a table which swaps key and value
function reverse_table(tbl)
	local rev = {};
	print(ipairs(tbl));
	for k, v in ipairs(tbl) do
		rev[v] = k;
	end
	return rev;
end
--Takes an index (y=x[index]) and return its inverse (x=y[index])
function inverse_mapping(ind)
	local a,b=torch.sort(ind,false)
	return b
end


function readAll(file)
    local f = io.open(file, "r")
    local content = f:read("*all")
    f:close()
    return content
end
function loadJson(fname)
	local t=readAll(fname)
	return cjson.decode(t)
end
function writeAll(file,data)
    local f = io.open(file, "w")
    f:write(data)
    f:close() 
end
function saveJson(fname,t)
	return writeAll(fname,cjson.encode(t))
end

--Join and split Tensors
--Originally https://github.com/torch/nn/blob/master/JoinTable.lua nn.JoinTable:updateOutput()
function join_vector(tensor_table,dimension)
	if dimension==nil then
		dimension=1;
	end
	local size=torch.LongStorage();
	for i=1,#tensor_table do
		local currentOutput = tensor_table[i];
		if i == 1 then
			size:resize(currentOutput:dim()):copy(currentOutput:size());
		else
			size[dimension] = size[dimension] + currentOutput:size(dimension);
		end
	end
	local output=tensor_table[1]:clone();
	output:resize(size);
	
	local offset = 1;
	for i=1,#tensor_table do
		local currentOutput = tensor_table[i];
		output:narrow(dimension, offset, currentOutput:size(dimension)):copy(currentOutput);
		offset = offset + currentOutput:size(dimension);
	end
	return output;
end
function split_vector(w,sizes)
	local tensor_table={};
	local offset=1;
	local n;
	if type(sizes)=="table" then
		n=#sizes;
	else
		n=sizes:size(1);
	end
	for i=1,n do
		table.insert(tensor_table,w[{{offset,offset+sizes[i]-1}}]);
		offset=offset+sizes[i];
	end
	return tensor_table;
end

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
-------SEQUENCE PROCESSING HELPER FUNCTIONS-----------
function onehot(ind,vocabulary_size)
	local n=ind:size(1);
	local v=torch.DoubleTensor(n,vocabulary_size):fill(0);
	v:scatter(2,ind:view(-1,1),1);
	return v;
end
function onehot_cuda(ind,vocabulary_size)
	local n=ind:size(1);
	local v=torch.CudaTensor(n,vocabulary_size):fill(0);
	v:scatter(2,ind:view(-1,1),1);
	return v;
end
function right_align(seq,lengths)
	local v=seq:clone():fill(0);
	local N=seq:size(2);
	for i=1,seq:size(1) do
		v[i][{{N-lengths[i]+1,N}}]=seq[i][{{1,lengths[i]}}];
	end
	collectgarbage();
	return v;
end
function sequence_length(seq)
	local v=seq:gt(0):sum(2):view(-1):long();
	return v;
end
--generate bow representation for sequences
function bag_of_words(seq,length,cutoff)
	--cutoff: not using the entire vocabulary, but only words 1-cutoff
	local n=seq:size(1);
	local bow=torch.ByteTensor(n,cutoff):fill(0);
	for i=1,n do
		for j=1,length[i] do
			if seq[i][j]>0 and seq[i][j]<=cutoff then
				bow[i][seq[i][j]]=bow[i][seq[i][j]]+1;
			end
		end
	end
	return bow;
end

-------RNN UTIL FUNCTIONS-----------
--Repeat an RNN block multiple times for testing
function dupe_rnn(net,times)
	local net_arr={};
	for i=1,times do
		--print(string.format('duping...%d',i))
		net_arr[i]=net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std');
	end
	collectgarbage();
	return net_arr;
end
--sort encoding onehot left aligned --TBD
--sort encoding onehot right aligned
--Inputs: sentences, lengths, vocabulary size
--Outputs: all words as onehot embedding matrix, how many sequences each RNN cell has to deal with, I reorder all sequences by length, so there's a sort index that does original => sorted and an inverse sorted => original.
function sort_encoding_onehot_right_align(batch_word_right_align,batch_length,vocabulary_size)
	--batch_word_right_align: batch_size x MAX_LENGTH matrix, words are right aligned.
	--batch_length: batch_size x 1 matrix depicting actual length
	local batch_length_sorted,sort_index=torch.sort(batch_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local D=batch_word_right_align:size(2);
	local L=batch_length_sorted[1];
	local batch_word_right_align_t=batch_word_right_align:index(1,sort_index):cuda():t()[{{D-L+1,D}}];
	local words=torch.LongTensor(torch.sum(batch_length)):cuda();
	local batch_sizes=torch.LongTensor(L);
	local cnt=0;
	for i=1,L do
		local ind=batch_length_sorted:ge(L-i+1);
		local n=torch.sum(ind);
		words[{{cnt+1,cnt+n}}]=batch_word_right_align_t[i][{{1,n}}];
		batch_sizes[i]=n;
		cnt=cnt+n;
	end
	return {onehot=onehot_cuda(words:cuda(),vocabulary_size),batch_sizes=batch_sizes,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
end
--don't need to make onehot encodings in this one. just return words
--Inputs: sentences, lengths, vocabulary size
--Outputs: all words as ids, how many sequences each RNN cell has to deal with, I reorder all sequences by length, so there's a sort index that does original => sorted and an inverse sorted => original.
function sort_encoding_lookup_right_align(batch_word_right_align,batch_length)
	--batch_word_right_align: batch_size x MAX_LENGTH matrix, words are right aligned.
	--batch_length: batch_size x 1 matrix depicting actual length
	local batch_length_sorted,sort_index=torch.sort(batch_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local D=batch_word_right_align:size(2);
	local L=batch_length_sorted[1];
	local batch_word_right_align_t=batch_word_right_align:index(1,sort_index):cuda():t()[{{D-L+1,D}}];
	local words=torch.LongTensor(torch.sum(batch_length)):cuda();
	local batch_sizes=torch.LongTensor(L);
	local cnt=0;
	for i=1,L do
		local ind=batch_length_sorted:ge(L-i+1);
		local n=torch.sum(ind);
		words[{{cnt+1,cnt+n}}]=batch_word_right_align_t[i][{{1,n}}];
		batch_sizes[i]=n;
		cnt=cnt+n;
	end
	return {words=words:cuda(),batch_sizes=batch_sizes,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
end


--sort encoding left aligned, longest first
function sort_encoding(batch_encoding,batch_length)
	--sort and binning processing--
	local buffer_size=batch_encoding:size(2);
	local batch_length_sorted,sort_index=torch.sort(batch_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local batch_encoding_permute=batch_encoding:permute(2,1,3);
	
	local batch_encoding_sorted={};
	local size_per_buffer=torch.zeros(batch_length_sorted[1]);
	for i=1,batch_length_sorted[1] do
		batch_encoding_sorted[i]=batch_encoding_permute[i]:index(1,sort_index[batch_length_sorted:ge(i)]):cuda();
		size_per_buffer[i]=torch.sum(batch_length_sorted:ge(i));
	end
	return batch_encoding_sorted,size_per_buffer,sort_index,sort_index_inverse;
end
--sort encoding right aligned, longest first
function sort_encoding_right_align(batch_encoding,batch_length)
	--sort and binning processing--
	local buffer_size=batch_encoding:size(2);
	local batch_length_sorted,sort_index=torch.sort(batch_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local batch_encoding_permute=batch_encoding:permute(2,1,3);
	
	local batch_encoding_sorted={};
	local size_per_buffer=torch.zeros(batch_length_sorted[1]);
	for i=1,batch_length_sorted[1] do
		batch_encoding_sorted[i]=batch_encoding_permute[buffer_size-batch_length_sorted[1]+i]:index(1,sort_index[batch_length_sorted:ge(batch_length_sorted[1]-i+1)]):cuda();
		size_per_buffer[i]=torch.sum(batch_length_sorted:ge(batch_length_sorted[1]-i+1));
	end
	return {batch_encoding_sorted,size_per_buffer,sort_index,sort_index_inverse};
end
--sort encoding left aligned, longest first, along with the id for rnn output
function sort_encoding_duo_id(batch_encoding,batch_encoding_out,batch_length)
	--sort and binning processing--
	local buffer_size=batch_encoding:size(2);
	local batch_length_sorted,sort_index=torch.sort(batch_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local batch_encoding_permute=batch_encoding:permute(2,1,3);
	local batch_encoding_out_permute=batch_encoding_out:permute(2,1);
	
	local batch_encoding_sorted={};
	local batch_encoding_out_sorted={};
	local size_per_buffer=torch.zeros(batch_length_sorted[1]);
	for i=1,batch_length_sorted[1] do
		batch_encoding_sorted[i]=batch_encoding_permute[i]:index(1,sort_index[batch_length_sorted:ge(i)]):cuda();
		batch_encoding_out_sorted[i]=batch_encoding_out_permute[i]:index(1,sort_index[batch_length_sorted:ge(i)]):cuda();
		size_per_buffer[i]=torch.sum(batch_length_sorted:ge(i));
	end
	return batch_encoding_sorted,batch_encoding_out_sorted,size_per_buffer,sort_index,sort_index_inverse;
end
--sort encoding right aligned, longest first, along with the id for rnn output
function sort_encoding_right_align_duo_id(batch_encoding,batch_encoding_out,batch_length)
	--sort and binning processing--
	local buffer_size=batch_encoding:size(2);
	local batch_length_sorted,sort_index=torch.sort(batch_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local batch_encoding_permute=batch_encoding:permute(2,1,3);
	local batch_encoding_out_permute=batch_encoding_out:permute(2,1);
	
	local batch_encoding_sorted={};
	local batch_encoding_out_sorted={};
	local size_per_buffer=torch.zeros(batch_length_sorted[1]);
	for i=1,batch_length_sorted[1] do
		batch_encoding_sorted[i]=batch_encoding_permute[buffer_size-batch_length_sorted[1]+i]:index(1,sort_index[batch_length_sorted:ge(batch_length_sorted[1]-i+1)]):cuda();
		batch_encoding_out_sorted[i]=batch_encoding_out_permute[buffer_size-batch_length_sorted[1]+i]:index(1,sort_index[batch_length_sorted:ge(batch_length_sorted[1]-i+1)]):cuda();
		size_per_buffer[i]=torch.sum(batch_length_sorted:ge(batch_length_sorted[1]-i+1));
	end
	return batch_encoding_sorted,batch_encoding_out_sorted,size_per_buffer,sort_index,sort_index_inverse;
end

--rnn forward, tries to handle most cases
function rnn_forward(net_buffer,init_state,inputs,sizes)
	local sizes_0=sizes:clone():fill(0);
	sizes_0[{{2,sizes:size(1)}}]=sizes[{{1,sizes:size(1)-1}}];
	local offsets_start=torch.cumsum(sizes_0)+1;
	local offsets_end=torch.cumsum(sizes);
	
	local N=sizes:size(1);
	local states={init_state[{{1,sizes[1]},{}}]};
	local outputs={};
	for i=1,N do
		local tmp;
		if i==1 or sizes[i]==sizes[i-1] then
			tmp=net_buffer[i]:forward({states[i],inputs[{{offsets_start[i],offsets_end[i]}}]});
		elseif sizes[i]>sizes[i-1] then
			--right align
			local padding=init_state[{{1,sizes[i]},{}}];
			padding[{{1,sizes[i-1]},{}}]=states[i];
			states[i]=padding;
			tmp=net_buffer[i]:forward({padding,inputs[{{offsets_start[i],offsets_end[i]}}]});
		elseif sizes[i]<sizes[i-1] then
			--left align
			tmp=net_buffer[i]:forward({states[i][{{1,sizes[i]}}],inputs[{{offsets_start[i],offsets_end[i]}}]});
		end
		table.insert(states,tmp[1]);
		table.insert(outputs,tmp[2]);
	end
	return states,outputs;
end

--rnn forward, with each unit sharing the same seed
function rnn_forward_dropout(net_buffer,init_state,inputs,sizes,seed)
	local sizes_0=sizes:clone():fill(0);
	sizes_0[{{2,sizes:size(1)}}]=sizes[{{1,sizes:size(1)-1}}];
	local offsets_start=torch.cumsum(sizes_0)+1;
	local offsets_end=torch.cumsum(sizes);
	
	local N=sizes:size(1);
	local states={init_state[{{1,sizes[1]},{}}]};
	local outputs={};
	for i=1,N do
		--set seed here so all units share the same seeds, hopefully resulting in the same dropout masks
		torch.manualSeed(seed);
		cutorch.manualSeedAll(seed);
		local tmp;
		if i==1 or sizes[i]==sizes[i-1] then
			tmp=net_buffer[i]:forward({states[i],inputs[{{offsets_start[i],offsets_end[i]}}]});
		elseif sizes[i]>sizes[i-1] then
			--right align
			local padding=init_state[{{1,sizes[i]},{}}];
			padding[{{1,sizes[i-1]},{}}]=states[i];
			states[i]=padding;
			tmp=net_buffer[i]:forward({padding,inputs[{{offsets_start[i],offsets_end[i]}}]});
		elseif sizes[i]<sizes[i-1] then
			--left align
			tmp=net_buffer[i]:forward({states[i][{{1,sizes[i]}}],inputs[{{offsets_start[i],offsets_end[i]}}]});
		end
		table.insert(states,tmp[1]);
		table.insert(outputs,tmp[2]);
	end
	return states,outputs;
end
--rnn backward
function rnn_backward(net_buffer,dend_state,doutputs,states,inputs,sizes)
	local sizes_0=sizes:clone():fill(0);
	sizes_0[{{2,sizes:size(1)}}]=sizes[{{1,sizes:size(1)-1}}];
	local offsets_start=torch.cumsum(sizes_0)+1;
	local offsets_end=torch.cumsum(sizes);
	
	if type(doutputs)=="table" then
		--Has output gradients
		local N=sizes:size(1);
		local dstate={[N+1]=dend_state[{{1,sizes[N]},{}}]};
		local dinput_embedding=inputs:clone():fill(0);
		for i=N,1,-1 do
			local tmp;
			if i==1 or sizes[i]==sizes[i-1] then
				tmp=net_buffer[i]:backward({states[i],inputs[{{offsets_start[i],offsets_end[i]}}]},{dstate[i+1],doutputs[i]});
				dstate[i]=tmp[1];
			elseif sizes[i]>sizes[i-1] then
				--right align
				tmp=net_buffer[i]:backward({states[i],inputs[{{offsets_start[i],offsets_end[i]}}]},{dstate[i+1],doutputs[i]});
				dstate[i]=tmp[1][{{1,sizes[i-1]},{}}];
			elseif sizes[i]<sizes[i-1] then
				--left align
				--compute a larger dstate that matches i-1
				tmp=net_buffer[i]:backward({states[i][{{1,sizes[i]}}],inputs[{{offsets_start[i],offsets_end[i]}}]},{dstate[i+1],doutputs[i]});
				local padding=dend_state[{{1,sizes[i-1]},{}}];
				padding[{{1,sizes[i]},{}}]=tmp[1];
				dstate[i]=padding;
			end
			dinput_embedding[{{offsets_start[i],offsets_end[i]}}]=tmp[2];
		end
		return dstate,dinput_embedding;
	else
		--Just dummy output gradients
		local N=sizes:size(1);
		local dstate={[N+1]=dend_state[{{1,sizes[N]},{}}]};
		local dinput_embedding=inputs:clone():fill(0);
		for i=N,1,-1 do
			local tmp;
			if i==1 or sizes[i]==sizes[i-1] then
				tmp=net_buffer[i]:backward({states[i],inputs[{{offsets_start[i],offsets_end[i]}}]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
				dstate[i]=tmp[1];
			elseif sizes[i]>sizes[i-1] then
				--right align
				tmp=net_buffer[i]:backward({states[i],inputs[{{offsets_start[i],offsets_end[i]}}]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
				dstate[i]=tmp[1][{{1,sizes[i-1]},{}}];
			elseif sizes[i]<sizes[i-1] then
				--left align
				--compute a larger dstate that matches i-1
				tmp=net_buffer[i]:backward({states[i][{{1,sizes[i]}}],inputs[{{offsets_start[i],offsets_end[i]}}]},{dstate[i+1],torch.repeatTensor(doutputs,sizes[i],1)});
				local padding=dend_state[{{1,sizes[i-1]},{}}];
				padding[{{1,sizes[i]},{}}]=tmp[1];
				dstate[i]=padding;
			end
			dinput_embedding[{{offsets_start[i],offsets_end[i]}}]=tmp[2];
		end
		return dstate,dinput_embedding;
	end	
end