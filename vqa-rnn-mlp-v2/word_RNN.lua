--[[
Variable length RNN
*  Designed for word->sentence encoding
*  Do word embeddings in a single pass
*  Only forwards words that are there by length-sorting and adjusting batch size, makes it faster and more memory-efficient than all the way

Usage:
* Initialize a network
            ```
            encoder_net=RNN:new(RNN.unit.lstm(nhword,nh,nhoutput,nlayers,0.5),rnn_length,true);
            ```
* Save a network
            ```
            unit=encoder_net.net;
            torch.save('save.t7',unit);
            ```
* Initialize from checkpoint
            ```
            encoder_net=RNN:new(unit,rnn_length,true);
            ```
* Forward
            ```
            words_sorted=sort_by_length_left_aligned(words,true);
            word_embeddings=embedding_net:forward(words_sorted.words);
            sentence_embeddings,rnn_outputs=encoder_net:forward(initial_states,word_embeddings,words_sorted.batch_sizes);
            sentence_embeddings=sentence_embeddings:index(1,words_sorted.map_to_sequence);
            ```
* Backward
            ```
            dsentence_embeddings=dsentence_embeddings:index(1,words_sorted.map_to_rnn);
            dinitial_states,dword_embeddings=encoder_net:backward(initial_states,word_embeddings,words_sorted.batch_sizes,dsentence_embeddings,drnn_outputs);
            embedding_net:backward(words_sorted.words,dword_embeddings);
            encoder_net.dw:clamp(-5,5);

            rmsprop(encoder_net.w,encoder_net.dw,opt_encoder);
            rmsprop(embedding_net.w,embedding_net.dw,opt_embedding);
            ```
--]]

require 'nn'
require 'nngraph'
local RNN={};

--Some util functions
--Takes an index (y=x[index]) and return its inverse mapping (x=y[index])
function inverse_mapping(ind)
	local a,b=torch.sort(ind,false);
	return b
end
--Given seq, an N sentence x L length matrix of words, count how many words are in each sentence.
--words are [1..n]
function sequence_length(seq)
	local v=seq:gt(0):long():sum(2):view(-1):long();
	return v;
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
function split_vector(w,sizes,dimension)
	if dimension==nil then
		dimension=1;
	end
	local tensor_table={};
	local offset=1;
	local n;
	if type(sizes)=="table" then
		n=#sizes;
	else
		n=sizes:size(1);
	end
	for i=1,n do
		table.insert(tensor_table,w:narrow(dimension,offset,sizes[i]));
		offset=offset+sizes[i];
	end
	return tensor_table;
end
--Turn a left-aligned N sentence x L length matrix of words to right aligned.
--Allocates new memory
function right_align(seq)
	local lengths=sequence_length(seq);
	local v=seq:clone():fill(0);
	local L=seq:size(2);
	for i=1,seq:size(1) do
		v[i][{{L-lengths[i]+1,L}}]=seq[i][{{1,lengths[i]}}];
	end
	return v;
end


--1) Sort sentences by their lengths in descending order
--2) Slice the sentences into batches of words for RNN input at each time step. 
--3) Combine the sentence slices into 1 array of words.
--Inputs:  seq: right-aligned sentences
--Outputs: 1) words: words at each time step, concatenated at dimension 1, so they can be fed into a LookupTable in 1 shot for efficiency.
--         2) batch_sizes: batch size at each time step. Starts small and ends at batch_size for right-aligned sentences.
--         3) map_to_rnn: mapping from the sentence order of seq to sentence order of the sorted sentences
--         4) map_to_sequence: mapping from the sorted sentences to the original sentence order of seq.
function sort_by_length_right_aligned(seq,gpu,seq_length)
	gpu=gpu or false;
	seq_length=seq_length or sequence_length(seq);
	local seq_length_sorted,sort_index=torch.sort(seq_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local seq_sorted=seq:index(1,sort_index);
	local MAX_LENGTH=seq:size(2);
	local L=seq_length_sorted[1];
	if L==0 then
		--We got a batch of empty sentences. There's no words, no batch sizes, so there's nothing to do.
		return {words=nil,batch_sizes=nil,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
	end
	local words=torch.LongTensor(seq_length:sum());
	local batch_sizes=torch.LongTensor(L);
	local offset=0;
	for i=1,L do
		local ind=seq_length_sorted:ge(L-i+1):long();
		local n=torch.sum(ind);
		batch_sizes[i]=n;
		words[{{offset+1,offset+n}}]=seq_sorted[{{1,n},MAX_LENGTH-L+i}];
		offset=offset+n;
	end
	if gpu then
		words=words:cuda();
		sort_index=sort_index:cuda();
		sort_index_inverse=sort_index_inverse:cuda();
	end
	return {words=words,batch_sizes=batch_sizes,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
end
--Inputs:  seq: left-aligned sentences
--Outputs: 1) words: words at each time step, concatenated at dimension 1, so they can be fed into a LookupTable in 1 shot for efficiency.
--         2) batch_sizes: batch size at each time step. Starts at batch_size and ends small for left-aligned sentences.
--         3) map_to_rnn: mapping from the sentence order of seq to sentence order of the sorted sentences
--         4) map_to_sequence: mapping from the sorted sentences to the original sentence order of seq.
function sort_by_length_left_aligned(seq,gpu,seq_length)
	gpu=gpu or false;
	seq_length=seq_length or sequence_length(seq);
	local seq_length_sorted,sort_index=torch.sort(seq_length,true);
	local sort_index_inverse=inverse_mapping(sort_index);
	local seq_sorted=seq:index(1,sort_index);
	local L=seq_length_sorted[1];
	if L==0 then
		--We got a batch of empty sentences. There's no words, no batch sizes, so there's nothing to do.
		return {words=nil,batch_sizes=nil,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
	end
	local words=torch.LongTensor(seq_length:sum());
	local batch_sizes=torch.LongTensor(L);
	local offset=0;
	for i=1,L do
		local ind=seq_length_sorted:ge(i):long();
		local n=torch.sum(ind);
		batch_sizes[i]=n;
		words[{{offset+1,offset+n}}]=seq_sorted[{{1,n},i}];
		offset=offset+n;
	end
	if gpu then
		words=words:cuda();
		sort_index=sort_index:cuda();
		sort_index_inverse=sort_index_inverse:cuda();
	end
	return {words=words,batch_sizes=batch_sizes,map_to_rnn=sort_index,map_to_sequence=sort_index_inverse};
end

--Variable length RNN forward. Applies to left- or right- aligned sequences, sorted by length in descending order
--Inputs:  1) init_state: initial states for all sequences.
--         2) input: input at each time step. It can be either a table of inputs at each time step or concatenated across time steps.
--         3) sizes: The batch sizes at each time step. Not required if input is a table.
--Outputs: 1) final_state: final states for all sequences
--         2) outputs: a table of outputs at each time step
function RNN:forward(init_state,input,sizes)
	--First normalize the input to a table
	local input_table;
	if input==nil or type(input)=='table' and input[1]==nil then
		--Welp, there's no input for whatever reason.
		--In that case it's a pass through of init_state and there's no output
		return init_state,{};
	elseif type(input)=='table' then
		--figure out sizes automatically
		sizes=torch.LongTensor(#input):fill(0);
		for i=1,#input do
			sizes[i]=input[i]:size(1);
		end
		input_table=input;
	else
		--create a table of inputs
		input_table=split_vector(input,sizes);
	end
	--Loop through timesteps and do forwards
	local outputs={};
	local final_state=init_state:clone();
	local right_aligned=true;
	local state=init_state[{{1,sizes[1]}}];
	for t=1,sizes:size(1) do
		local tmp;
		--check whether the batch size is shrinking to detect left/right aligned sequences.
		if t==1 or sizes[t]==sizes[t-1] then
			--batch size doesn't change, just do forward.
			if self.cell_inputs[t]==nil then
				self.cell_inputs[t]=state:clone();
			else
				self.cell_inputs[t]:resizeAs(state);
				self.cell_inputs[t][{{}}]=state;
			end
			tmp=self.deploy[t]:forward({self.cell_inputs[t],input_table[t]});
		elseif sizes[t]>sizes[t-1] then
			--batch size becomes larger. Sequence is right aligned.
			--enlarge state so it fits the new batch size
			if self.cell_inputs[t]==nil then
				self.cell_inputs[t]=init_state[{{1,sizes[t]}}]:clone():zero();
			else
				self.cell_inputs[t]:resizeAs(init_state[{{1,sizes[t]}}]);
			end
			self.cell_inputs[t][{{1,sizes[t-1]}}]=state[{}];
			self.cell_inputs[t][{{sizes[t-1]+1,sizes[t]}}]=init_state[{{sizes[t-1]+1,sizes[t]}}];
			--forward
			tmp=self.deploy[t]:forward({self.cell_inputs[t],input_table[t]});
		elseif sizes[t]<sizes[t-1] then
			--batch size becomes smaller. Sequence is left aligned.
			--shrink state so it fits the new batch size
			right_aligned=false;
			if self.cell_inputs[t]==nil then
				self.cell_inputs[t]=state[{{1,sizes[t]}}]:clone();
			else
				self.cell_inputs[t]:resizeAs(state[{{1,sizes[t]}}]);
				self.cell_inputs[t][{{}}]=state[{{1,sizes[t]}}];
			end
			--copy the left overs to final state
			final_state[{{sizes[t]+1,sizes[t-1]}}]=state[{{sizes[t]+1,sizes[t-1]}}];
			--forward
			tmp=self.deploy[t]:forward({self.cell_inputs[t],input_table[t]});
		end
		state=tmp[1];
		table.insert(outputs,tmp[2]);
	end
	final_state[{{1,sizes[sizes:size(1)]}}]=self.deploy[sizes:size(1)].output[1];
	return final_state,outputs;
end
--Variable length RNN backward. Applies to left- or right- aligned sequences, sorted by length in descending order
--Inputs:  1) init_state: initial states for all sequences.
--         2) input: input at each time step. It can be either a table of inputs at each time step or concatenated across time steps.
--         3) sizes: The batch sizes at each time step. Can be nil if input is a table.
--         4) dfinal_state: gradient to the final state
--         5) doutputs: a table of gradients to the outputs at each time step
--Outputs: 1) dinit_state: final states for all sequences
--         2) dinput: derivative to the inputs, concatenated over time steps.
function RNN:backward(init_state,input,sizes,dfinal_state,doutputs)
	--First normalize the input to a table
	local input_table;
	if input==nil or type(input)=='table' and input[1]==nil then
		--Welp, there's no input for whatever reason.
		--The network is just pass through, so there's no need to backprop.
		return dfinal_state,nil;
	elseif type(input)=='table' then
		--figure out sizes automatically
		sizes=torch.LongTensor(#input):fill(0);
		for i=1,#input do
			sizes[i]=input[i]:size(1);
		end
		input_table=input;
	else
		--create a table of inputs
		input_table=split_vector(input,sizes);
	end
	--Then normalize the doutputs to a table
	local doutputs_table;
	local doutputs_exp;
	if type(doutputs)=='table' then
		doutputs_table=doutputs;
	else
		--make a table by repeating dummy output gradients
		doutputs_exp=torch.repeatTensor(doutputs,sizes:max(),1);
		doutputs_table={};
		for t=1,sizes:size(1) do
			doutputs_table[t]=doutputs_exp[{{1,sizes[t]}}];
		end
	end
	
	--Loop through timesteps and do backwards	
	local N=sizes:size(1);
	local dstate=dfinal_state[{{1,sizes[N]}}]:clone();
	local dinit_state=dfinal_state:clone();
	--local dinput_embedding=input:clone():fill(0);
	local dinput_embedding={};
	local left_aligned=true;
	local offset=sizes:sum();	
	for t=N,1,-1 do
		local tmp;
		--check whether the batch size is shrinking to detect right/left aligned sequences.
		if t==1 or sizes[t]==sizes[t-1] then
			--It's not shrinking, so just do backward
			tmp=self.deploy[t]:backward({self.cell_inputs[t],input_table[t]},{dstate,doutputs_table[t]});
			dstate[{{}}]=tmp[1];
		elseif sizes[t]>sizes[t-1] then
			--It's shrinking, the sequence is right aligned.
			left_aligned=false;
			tmp=self.deploy[t]:backward({self.cell_inputs[t],input_table[t]},{dstate,doutputs_table[t]});
			--We need to shrink dstate to fit the previous cell
			dstate:resizeAs(tmp[1][{{1,sizes[t-1]}}]);
			dstate[{{}}]=tmp[1][{{1,sizes[t-1]}}];
			--And copy the left overs to the initial state
			dinit_state[{{sizes[t-1]+1,sizes[t]}}]=tmp[1][{{sizes[t-1]+1,sizes[t]}}];
		elseif sizes[t]<sizes[t-1] then
			--It's expanding, the sequence is left aligned.
			tmp=self.deploy[t]:backward({self.cell_inputs[t],input_table[t]},{dstate,doutputs_table[t]});
			--We need to expand dstate to fit the previous cell
			dstate:resizeAs(dfinal_state[{{1,sizes[t-1]}}]);
			dstate[{{1,sizes[t]}}]=tmp[1][{}];
			dstate[{{sizes[t]+1,sizes[t-1]}}]=dfinal_state[{{sizes[t]+1,sizes[t-1]}}];
		end
		--Slot the input embedding gradients
		--dinput_embedding[{{offset-sizes[t]+1,offset}}]=tmp[2];
		dinput_embedding[t]=tmp[2]:clone();
		offset=offset-sizes[t];
	end
	dinit_state[{{1,sizes[1]}}]=dstate;
	return dinit_state,join_vector(dinput_embedding);
end

function RNN:new(unit,n,gpu)
	--unit: {state,input}->{state,output}
	--gpu: yes/no
	gpu=gpu or false;
	local net={};
	local netmeta={};
	netmeta.__index = RNN
	setmetatable(net,netmeta);
	--stuff
	if gpu then
	net.net=unit:cuda();
	else
	net.net=unit;
	end
	net.w,net.dw=net.net:getParameters();
	net.n=n;
	
	net.deploy={};
	for i=1,n do
		net.deploy[i]=net.net:clone('weight','bias','gradWeight','gradBias','running_mean','running_std','running_var');
	end
	collectgarbage();
	net.cell_inputs={};
	return net;
end
function RNN:training()
	self.net:training();
	for i=1,self.n do
		self.deploy[i]:training();
	end
end
function RNN:evaluate()
	self.net:evaluate();
	for i=1,self.n do
		self.deploy[i]:evaluate();
	end
end
function RNN:clearState()
	self.net:clearState();
	for i=1,self.n do
		self.deploy[i]:clearState();
	end
end
RNN.unit={};
--{c+h,input}->{c+h,output}
function RNN.unit.lstm(nhinput,nh,noutput,n,dropout)
	dropout = dropout or 0 
	local h_prev=nn.Identity()(); -- batch x (2nh), combination of past cell state and hidden state
	local input=nn.Identity()(); -- batch x nhword, input embeddings
	local prev_c={};
	local prev_h={};
	for i=1,n do
		prev_c[i]=nn.Narrow(2,2*(i-1)*nh+1,nh)(h_prev);
		prev_h[i]=nn.Narrow(2,(2*i-1)*nh+1,nh)(h_prev);
	end
	local c={};
	local h={};
	local mixed={};
	for i=1,n do
		local x;
		local input_size;
		if i==1 then
			x=input;
			input_size=nhinput;
		else
			x=h[i-1];
			input_size=nh;
		end
		local controls=nn.Linear(input_size+nh,4*nh)(nn.JoinTable(1,1)({x,prev_h[i]}));
		--local sigmoid_chunk=nn.Sigmoid()(nn.Narrow(2,1,3*nh)(controls));
		local data_chunk=nn.Tanh()(nn.Narrow(2,3*nh+1,nh)(controls));
		local in_gate=nn.Sigmoid()(nn.Narrow(2,1,nh)(controls));
		local out_gate=nn.Sigmoid()(nn.Narrow(2,nh+1,nh)(controls));
		local forget_gate=nn.Sigmoid()(nn.Narrow(2,2*nh+1,nh)(controls));
		c[i]=nn.CAddTable()({nn.CMulTable()({forget_gate,prev_c[i]}),nn.CMulTable()({in_gate,data_chunk})});
		h[i]=nn.CMulTable()({out_gate,nn.Tanh()(c[i])});
		table.insert(mixed,c[i]);
		table.insert(mixed,h[i]);
	end
	local h_current=nn.JoinTable(1,1)(mixed);
	local output=nn.Linear(nh,noutput)(nn.Dropout(dropout)(h[n]));
	return nn.gModule({h_prev,input},{h_current,output});
end

function RNN.unit.gru(nhinput,nh,noutput,n,dropout)
	dropout = dropout or 0 
	local h_prev=nn.Identity()(); -- batch x (2nh), combination of past cell state and hidden state
	local input=nn.Identity()(); -- batch x nhword, input embeddings
	local prev_h={};
	for i=1,n do
		prev_h[i]=nn.Narrow(2,(i-1)*nh+1,nh)(h_prev);
	end
	local h={};
	local mixed={};
	for i=1,n do
		local x;
		local input_size;
		if i==1 then
			x=input;
			input_size=nhinput;
		else
			x=h[i-1];
			input_size=nh;
		end
		local tx=nn.Linear(input_size,3*nh)(x);
		local th=nn.Linear(nh,2*nh)(prev_h[i]);
		local sigmoid_chunk=nn.Sigmoid()(nn.CAddTable()({nn.Narrow(2,1,2*nh)(tx),th}));
		local output_gate=nn.Narrow(2,1,nh)(sigmoid_chunk);
		local forget_gate=nn.Narrow(2,nh+1,nh)(sigmoid_chunk);
		local data_chunk=nn.Tanh()(nn.CAddTable()({nn.Narrow(2,2*nh+1,nh)(tx),nn.Linear(nh,nh)(nn.CMulTable()({output_gate,prev_h[i]}))}));
		h[i]=nn.CAddTable()({nn.CMulTable()({forget_gate,data_chunk}),nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(forget_gate)),prev_h[i]})});
	end
	local h_current=nn.JoinTable(1,1)(h);
	local output=nn.Linear(nh,noutput)(nn.Dropout(dropout)(h[n]));
	return nn.gModule({h_prev,input},{h_current,output});
end

--GRU from a very old Andrej Karpathy's Char-RNN commit with a wrapper around c's and h's
function RNN.unit.gru_old(input_size,rnn_size,noutput,n,dropout)
  dropout = dropout or 0 
--my wrapper
	local h_old=nn.Identity()();
	local input=nn.Identity()();
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, input) -- x
  for L = 1,n do
    table.insert(inputs, nn.Narrow(2, (L-1)*rnn_size+1, rnn_size)(h_old)) 
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do

    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, noutput)(top_h)
	
	local h_new;
	if n>1 then
		h_new=nn.JoinTable(1,1)(outputs);
	else
		h_new=outputs[1];
	end
	local outs=proj;

  return nn.gModule({h_old,input},{h_new,outs}) 
end




return RNN