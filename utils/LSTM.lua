require 'nn'
require 'nngraph'

LSTM={};

--Author: Andrej Karpathy https://github.com/karpathy
--Project: Char-RNN https://github.com/karpathy/char-rnn
--Slightly modified by Xiao Lin for generic input (not just one hot vectors) and output (removing logsoftmax).

function LSTM.lstm(input_size,rnn_size,noutput,n,dropout)
  dropout = dropout or 0 
--my wrapper
	local h_old=nn.Identity()();
	local input=nn.Identity()();
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, input) -- x
  for L = 1,n do
    table.insert(inputs, nn.Narrow(2, 2*(L-1)*rnn_size+1, rnn_size)(h_old)) -- prev_c[L]
    table.insert(inputs, nn.Narrow(2, 2*(L-1)*rnn_size+rnn_size+1, rnn_size)(h_old)) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, noutput)(top_h)

 
	local h_new=nn.JoinTable(1,1)(outputs);
	local outs=proj;

  return nn.gModule({h_old,input},{h_new,outs})
end


--col dropout on all linear parameters
function LSTM.lstm_dropout(input_size,rnn_size,noutput,n,dropout)
  dropout = dropout or 0 
--my wrapper
	local h_old=nn.Identity()();
	local input=nn.Identity()();
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, input) -- x
  for L = 1,n do
    table.insert(inputs, nn.Narrow(2, 2*(L-1)*rnn_size+1, rnn_size)(h_old)) -- prev_c[L]
    table.insert(inputs, nn.Narrow(2, 2*(L-1)*rnn_size+rnn_size+1, rnn_size)(h_old)) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(nn.Dropout(dropout)(x))
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(nn.Dropout(dropout)(prev_h))
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  local proj = nn.Linear(rnn_size, noutput)(nn.Dropout(dropout)(top_h))

 
	local h_new=nn.JoinTable(1,1)(outputs);
	local outs=proj;

  return nn.gModule({h_old,input},{h_new,outs})
end

--Add sparsity
--Cell state are larger, may hold many detailed states
--Hidden states are may hold factors and are sparse
function LSTM.lstm_sparse(input_size,cell_size,hidden_size,noutput,n,dropout)
	dropout = dropout or 0 
	--my wrapper
	local h_old=nn.Identity()();
	local input=nn.Identity()();
	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, input) -- x
	for L = 1,n do
		table.insert(inputs, nn.Narrow(2, (L-1)*(cell_size+hidden_size)+1, cell_size)(h_old)) -- prev_c[L]
		table.insert(inputs, nn.Narrow(2, (L-1)*(cell_size+hidden_size)+cell_size+1, hidden_size)(h_old)) -- prev_h[L]
	end

	local x, input_size_L
	local outputs = {}
	for L = 1,n do
		-- c,h from previos timesteps
		local prev_h = inputs[L*2+1]
		local prev_c = inputs[L*2]
		-- the input to this layer
		if L == 1 then 
			x = inputs[1]
			input_size_L = input_size
		else 
			x = outputs[(L-1)*2] 
			if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
			input_size_L = hidden_size
		end
		-- evaluate the input sums at once for efficiency
		local i2h = nn.Linear(input_size_L, 3 * cell_size + hidden_size)(x)
		local h2h = nn.Linear(hidden_size, 3 * cell_size + hidden_size)(prev_h)
		local all_input_sums = nn.CAddTable()({i2h, h2h})
		-- decode the gates
		local sigmoid_chunk = nn.Narrow(2, 1, 2 * cell_size+hidden_size)(all_input_sums)
		sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
		local in_gate = nn.Narrow(2, 1, cell_size)(sigmoid_chunk)
		local forget_gate = nn.Narrow(2, cell_size + 1, cell_size)(sigmoid_chunk)
		local out_gate = nn.Narrow(2, 2 * cell_size + 1, hidden_size)(sigmoid_chunk)
		-- decode the write inputs
		local in_transform = nn.Narrow(2, 2 * cell_size + hidden_size + 1, cell_size)(all_input_sums)
		in_transform = nn.Tanh()(in_transform)
		-- perform the LSTM update
		local next_c           = nn.CAddTable()({
			nn.CMulTable()({forget_gate, prev_c}),
			nn.CMulTable()({in_gate,     in_transform})
		  })
		-- gated cells form the output
		local output;
		if dropout>0 then
			output=nn.Tanh()(nn.Linear(cell_size,hidden_size)(nn.Dropout(dropout)(next_c)));
		else
			output=nn.Tanh()(nn.Linear(cell_size,hidden_size)(next_c));
		end
		local next_h = nn.CMulTable()({out_gate,output})
		table.insert(outputs, next_c)
		table.insert(outputs, next_h)
	end

	-- set up the decoder
	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
	local proj = nn.Linear(hidden_size, noutput)(top_h)

	--local top_c = outputs[#outputs-1]
	--if dropout > 0 then top_c = nn.Dropout(dropout)(top_c) end
	--local proj = nn.Linear(cell_size, noutput)(top_c)

	local h_new=nn.JoinTable(1,1)(outputs);
	local outs=proj;

	return nn.gModule({h_old,input},{h_new,outs})
end
--GRU from Andrej Karpathy's Char-RNN
function LSTM.gru(input_size,rnn_size,noutput,n,dropout)
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

return LSTM