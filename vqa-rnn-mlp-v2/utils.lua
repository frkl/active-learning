--Input: token_table, table of tokens
--{
--{'what','is','the','weather','today','?'}
--{'what','is','it','?'}
--}
--Output: tokens, tensor
--[
--[1,2,3,4,5,6]
--[1,2,7,6,0,0]
--]
--Dictionary: table of words. 1~#dictionary for words in dictionary. #dictionary+1=UNK
--word_cap: maximum number of words in dictionary
--length_cap: maximum sentence length, set to avoid blowing up memory
function encode_sents(token_table,dictionary,word_cap,length_cap)
	local sz={#token_table,0};
	if dictionary==nil then
		--Build a token dictionary 
		local word_freq={};
		for i=1,#token_table do
			for j=1,#token_table[i] do
				if word_freq[token_table[i][j]]==nil then
					word_freq[token_table[i][j]]=0;
				end
				word_freq[token_table[i][j]]=word_freq[token_table[i][j]]+1;
			end
		end
		local tmp={};
		for w,f in pairs(word_freq) do
			table.insert(tmp,{f,w});
		end
		local function cmp(a,b) 
			return a[1]>b[1] 
		end
		table.sort(tmp,cmp);
		dictionary={};
		local c=0;
		for i,w in ipairs(tmp) do
			if word_cap==nil or c<word_cap then
				table.insert(dictionary,w[2]);
			end
			c=c+1;
		end
	end
	collectgarbage();
	--Figure out how much storage is required.
	for i=1,#token_table do
		sz[2]=math.max(sz[2],#token_table[i]);
	end
	if length_cap~=nil then
		sz[2]=math.min(sz[2],length_cap);
	end
	print(string.format('Allocating %d x %d',sz[1],sz[2]));
	
	--Map to dictionary
	local word2id={};
	for i,w in ipairs(dictionary) do
		word2id[w]=i;
	end
	local tokens=torch.LongTensor(sz[1],sz[2]):fill(0);
	for i=1,#token_table do
		for j=1,#token_table[i] do
			if length_cap==nil or j<=length_cap then
				if word2id[token_table[i][j]]==nil then
					--UNK
					tokens[{i,j}]=#dictionary+1;
				else
					tokens[{i,j}]=word2id[token_table[i][j]];
				end
			end
		end
	end
	return tokens,dictionary;
end

--Same as above, for 3D tensors.
function encode_sents3(token_table,dictionary,word_cap,length_cap)
	local sz={#token_table,0,0};
	if dictionary==nil then
		--Build a token dictionary 
		local word_freq={};
		for i=1,#token_table do
			for j=1,#token_table[i] do
				for k=1,#token_table[i][j] do
					if word_freq[token_table[i][j][k]]==nil then
						word_freq[token_table[i][j][k]]=0;
					end
					word_freq[token_table[i][j][k]]=word_freq[token_table[i][j][k]]+1;
				end
			end
		end
		local tmp={};
		for w,f in pairs(word_freq) do
			table.insert(tmp,{f,w});
		end
		local function cmp(a,b) 
			return a[1]>b[1] 
		end
		table.sort(tmp,cmp);
		dictionary={};
		local c=0;
		for i,w in ipairs(tmp) do
			if word_cap==nil or c<word_cap then
				table.insert(dictionary,w[2]);
			end
			c=c+1;
		end
	end
	collectgarbage();
	--Figure out how much storage is required.
	for i=1,#token_table do
		sz[2]=math.max(sz[2],#token_table[i]);
		for j=1,#token_table[i] do
			sz[3]=math.max(sz[3],#token_table[i][j]);
		end
	end
	if length_cap~=nil then
		sz[3]=math.min(sz[3],length_cap);
	end
	print(string.format('Allocating %d x %d x %d',sz[1],sz[2],sz[3]));
	
	--Map to dictionary
	local word2id={};
	for i,w in ipairs(dictionary) do
		word2id[w]=i;
	end
	local tokens=torch.LongTensor(sz[1],sz[2],sz[3]):fill(0);
	for i=1,#token_table do
		for j=1,#token_table[i] do
			for k=1,#token_table[i][j] do
				if length_cap==nil or k<=length_cap then
					if word2id[token_table[i][j][k]]==nil then
						--UNK
						tokens[{i,j,k}]=#dictionary+1;
					else
						tokens[{i,j,k}]=word2id[token_table[i][j][k]];
					end
				end
			end
		end
	end
	return tokens,dictionary;
end


--3D token tensor to bag of words representation
function bow_sents3(token_table,dictionary,nhbow)
	local sz={#token_table,0};
	if dictionary==nil then
		--Build a token dictionary 
		local word_freq={};
		for i=1,#token_table do
			for j=1,#token_table[i] do
				for k=1,#token_table[i][j] do
					if word_freq[token_table[i][j][k]]==nil then
						word_freq[token_table[i][j][k]]=0;
					end
					word_freq[token_table[i][j][k]]=word_freq[token_table[i][j][k]]+1;
				end
			end
		end
		local tmp={};
		for w,f in pairs(word_freq) do
			table.insert(tmp,{f,w});
		end
		local function cmp(a,b) 
			return a[1]>b[1] 
		end
		table.sort(tmp,cmp);
		dictionary={};
		local c=0;
		for i,w in ipairs(tmp) do
			table.insert(dictionary,w[2]);
		end
	end
	collectgarbage();
	--Figure out how much storage is required.
	for i=1,#token_table do
		sz[2]=math.max(sz[2],#token_table[i]);
	end
	print(string.format('Allocating %d x %d x %d',sz[1],sz[2],nhbow));
	
	--Map to dictionary
	local word2id={};
	for i,w in ipairs(dictionary) do
		word2id[w]=i;
	end
	local tokens=torch.LongTensor(sz[1],sz[2],nhbow):fill(0);
	local indicator=torch.LongTensor(sz[1],sz[2]):fill(0);
	print('Computing bow');
	for i=1,#token_table do
		for j=1,#token_table[i] do
			indicator[{i,j}]=1;
			for k=1,#token_table[i][j] do
				local id=word2id[token_table[i][j][k]];
				if id~=nil and id<=nhbow then
					tokens[{i,j,id}]=tokens[{i,j,id}]+1;
				end
			end
		end
		if i%1000==0 then
			--collectgarbage();
		end
	end
	return tokens,dictionary,indicator;
end