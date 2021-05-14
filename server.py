print("-------------------------")
print("GodAI Transformers Server")
print("-------------------------")
print(" ")

import asyncio
import websockets
import sys
import json
import gc
from threading import Thread

import torch
import random
import ast
from transformers import AutoTokenizer, GPTNeoForCausalLM

from transformers import GPTNeoModel
from transformers.modeling_outputs import BaseModelOutputWithPast

class Request:
	GENERATE = 0
	TOKENIZE = 1
	DETOKENIZE = 2
	CHECK_LENGTH = 3
	QUIT = 4

device = ""
model = None
tokenizer = None
number_of_parts = 32
started = False
max_context = 1024

def top_k_top_p_filtering(logits, top_k = 0, top_p = 0.0, value = -float("Inf")):
	""" Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
		Args:
			logits: logits distribution shape (batch size x vocabulary size)
			top_k > 0: keep only top k tokens with highest probability (top-k filtering).
			top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
				Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
		From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
	"""
	top_k = min(top_k, logits.size(-1))  # Safety check
	if top_k > 0:
		#Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = value

	if top_p > 0.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending = True)
		cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim = -1), dim = -1)

		#Remove tokens with cumulative probability above the threshold
		sorted_indices_to_remove = cumulative_probs > top_p
		#Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		#Scatter sorted tensors to original indexing
		indices_to_remove = sorted_indices_to_remove.scatter(
			dim = -1, index = sorted_indices, src = sorted_indices_to_remove
		)
		logits[indices_to_remove] = value

	return logits

def token_encode(text: str):
	tokens = tokenizer.encode(text, add_special_tokens = False, add_prefix_space = True)
	return tokens

def token_decode(tokens):
	text = tokenizer.decode(tokens, clean_up_tokenization_spaces = True, skip_special_tokens = True)
	return text

async def hello(clientsocket, path):
	print("Client Connected")
	
	global started
	global device
	if not started:
		global model
		global tokenizer
		
		await clientsocket.send("Waiting for model")
		buf = await clientsocket.recv()
		data = buf.decode()
		data = ast.literal_eval(data)

		await clientsocket.send("Loading Model {}".format(data["model"]))
		
		print("Starting model {} ({}-bit)".format(data["model"], data["bit_mode"]))
		tokenizer = AutoTokenizer.from_pretrained(data["model"])
		model = GPTNeoForCausalLM.from_pretrained(data["model"])
		
		gpu = torch.cuda.is_available()

		
		global max_context
		max_context = model.config.n_positions
		
		started = True
		
        model.eval().half().to("cpu")
        model.transformer.wte.to("cuda")
        model.transformer.wpe.to("cuda")
        model.transformer.ln_f.to("cuda")
        model.lm_head.to("cuda")
        torch.cuda.empty_cache()
        for param in model.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.transformer.wpe.parameters():
            param.requires_grad = False
        for i in range(32):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = False
        for param in model.transformer.ln_f.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        setattr(model.transformer,"extrastorage",None)
        model.transformer.extrastorage = copy.deepcopy(model.transformer.h)
        smalltensor = torch.tensor(0).to("cuda")
        for j in range(32):
            for param1 in model.transformer.h[j].parameters():
                param1.data = smalltensor
        gc.collect()
        torch.cuda.empty_cache()
        model.transformer.extrastorage.to("cpu")
        for i in range(32):
            for param in model.transformer.extrastorage[i].parameters():
                param.requires_grad = False
                param.data.pin_memory()
        gc.collect()
        torch.cuda.empty_cache()
        if number_of_parts == 2:
            for j in range(16,32):
                for param1,param2 in zip(model.transformer.h[j].parameters(),model.transformer.extrastorage[j].parameters()):
                    param1.data = param2.data.to("cuda", non_blocking=True)
                model.transformer.h[j].to("cuda", non_blocking=True)  
            print("number_of_parts = 4" )
            
        if number_of_parts == 4:
            for j in range(24,32):
                for param1,param2 in zip(model.transformer.h[j].parameters(),model.transformer.extrastorage[j].parameters()):
                    param1.data = param2.data.to("cuda", non_blocking=True)
                model.transformer.h[j].to("cuda", non_blocking=True)  
            print("number_of_parts = 4" )
            
        if number_of_parts == 32:
            for param1,param2 in zip(model.transformer.h[31].parameters(),model.transformer.extrastorage[31].parameters()):
                param1.data = param2.data.to("cuda", non_blocking=True)
            model.transformer.h[31].to("cuda", non_blocking=True)  
            print("number_of_parts = 32" )
        
        
		print("Model loaded!")

	while True:
		print("---\nAwaiting for input")
		await clientsocket.send("Awaiting for input")
		buf = await clientsocket.recv()

		#print(buf)

		#Decode client data
		data = buf.decode()
		data = ast.literal_eval(data)

		top_p_first = True
		ignore_tokens = None
		replace_tokens = token_encode(["<|endoftext|>"])
		replaced_by = [198] #New line - don't judge me
		
		request = data["request"]

		if request == Request.QUIT:
			sys.exit(1785)
			
		elif request == Request.CHECK_LENGTH:
			print("Checking length")

			#Should return values: Untrimmed token length, Trimmed token length, Untrimmed text length, Trimmed text length, Fringe token length, Fringe text length
			tokens = token_encode(data["text"])

			trimmed = tokens[-max_context:]
			trimmed_text = token_decode(trimmed)

			fringe = tokens[-max_context + max_length:]
			fringe_text = token_decode(fringe)

			data["max_context"] = max_context
			data["token_untrimmed_size"] = len(tokens)
			data["token_trimmed_size"] = len(trimmed)
			data["token_fringe_size"] = len(fringe)
			data["text_trimmed"] = trimmed_text
			data["text_fringe"] = fringe_text

		elif request == Request.GENERATE:
			#Client variables
			max_length = data["max_length"]
			temperature = data["temperature"]
			top_k = data["top_k"]
			top_p = data["top_p"]
			repetition_penalty = data["repetition_penalty"]
			
			print("Generating Text")
			await clientsocket.send("Generating Text")

			context = token_encode(data["text"])
			context_tokens = context
			
			context = torch.tensor(context, dtype = torch.long, device = "cuda")
			generated = context
			next_token = context[-max_context + max_length:]

			print("Context length: {}".format(len(next_token)))
			if len(next_token) != len(context):
				print("Trimmed Context: {} -> {}".format(len(context), len(next_token)))
				await clientsocket.send("Trimmed Context: {} -> {}".format(len(context), len(next_token)))
				
			data["token_trimmed_size"] = len(next_token)
			data["token_untrimmed_size"] = len(context)

			use_past = True
			pasts = None
			generated_amount = 0

			with torch.no_grad():
				for step in range(max_length):
					#Use cached previous generation
					if not use_past:
						input_ids_next = generated
						pasts = None
					else:
						input_ids_next = next_token
					
					#Generate choices
					input_ids_next.to("cuda")
					logits, next_pasts = model(input_ids = input_ids_next, past_key_values = pasts)
					logits.to("cuda")
					
					logits = logits[-1, :].float()

					if top_p_first:
						logits = top_k_top_p_filtering(logits, top_k = top_k, top_p = top_p)
					
					logits = logits / (temperature if temperature > 0 else 1.0)
					
					#Repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
					for k in set(generated.tolist()):
						logits[k] /= repetition_penalty

					if not top_p_first:
						logits = top_k_top_p_filtering(logits, top_k = top_k, top_p = top_p)
					
					new_token = None
					
					if temperature == 0:
						new_token = torch.argmax(logits, dim = -1).unsqueeze(-1)
					else:
						new_token = torch.multinomial(
							torch.nn.functional.softmax(logits, dim = -1), num_samples = 1
						)
					
					pasts = next_pasts
					
					if new_token in replace_tokens:
						#print("{} replaced by {}".format(new_token[0], replaced_by[0]))
						new_token[0] = replaced_by[0]
						
					next_token = new_token
					#print(next_token)
					
					#decoded = token_decode(next_token)
					#print("{} is {}".format(next_token, decoded))
					
					generated = torch.cat((generated, next_token), dim = -1)
					generated_amount += 1
					
					if generated_amount % 25 == 0:
						print(("Generated {}/{}".format(generated_amount, max_length)))
						#await clientsocket.send("Generated {}/{}".format(generated_amount, max_length))
						
					#if generated_amount % 5 == 0:
					#	await clientsocket.send("Progress {}".format(generated_amount))

					#Decode into plain text
					o = generated[len(context_tokens):].tolist()
					generated.text = tokenizer.decode(
						o, clean_up_tokenization_spaces = True, skip_special_tokens = False
					)

					# if (stop_tokens is not None) and (step > 4) and (next_token[0] in stop_tokens):
						# print(
							# "Stopping generation as we found stop tokens. One of `%s`, in '%s'. token generated `%s`",
							# stop_tokens,
							# next_token,
							# j,
						# )
						# break

			#print(generated.text)
			
			data["generated_text"] = generated.text

			#clear_lines(clines)
			print("Text generated: {}".format(generated.text))

		print("Sending output")
		data["result"] = 0
		output = json.dumps(data)

		#print(output)

		await clientsocket.send(output)

def new_forward(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    global number_of_parts
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:
        assert batch_size > 0, "batch_size has to be defined and > 0"
        global_attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        global_attention_mask = global_attention_mask[:, None, None, :]

        # Since global_attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        global_attention_mask = global_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        global_attention_mask = (1.0 - global_attention_mask) * -10000.0
    else:
        global_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x num_headss x N x N
    # head_mask has shape n_layer x batch x num_headss x N x N
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if number_of_parts == 2:
            if i == 0:
                cudastreams = {}
                for j in range(0,16):
                    cudastreams[j] = torch.cuda.Stream()
                    for param1,param2 in zip(self.h[j].parameters(),self.h[j+16].parameters()):
                        param1.data = param2.data
                        
                for j in range(0,16):
                    with torch.cuda.stream(cudastreams[j]):
                        for param1,param2 in zip(self.h[j].parameters(),self.extrastorage[j].parameters()):
                            param1.data.copy_(param2.data, non_blocking=True)
                        self.h[j].to("cuda", non_blocking=True)
                        
                torch.cuda.synchronize()
                del cudastreams
                
            if i == 16:
                cudastreams = {}
                for j in range(16,32):
                    cudastreams[j] = torch.cuda.Stream()
                    for param1,param2 in zip(self.h[j].parameters(),self.h[j-16].parameters()):
                        param1.data = param2.data
                        
                for j in range(16,32):  
                    with torch.cuda.stream(cudastreams[j]):
                        for param1,param2 in zip(self.h[j].parameters(),self.extrastorage[j].parameters()):
                            param1.data.copy_(param2.data, non_blocking=True)
                            pass
                        self.h[j].to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                del cudastreams
                
        if number_of_parts == 4:
            if i == 0:
                cudastreams = {}
                for j in range(0,8):
                    cudastreams[j] = torch.cuda.Stream()
                    for param1,param2 in zip(self.h[j].parameters(),self.h[j+24].parameters()):
                        param1.data = param2.data
                for j in range(0,8):
                    with torch.cuda.stream(cudastreams[j]):
                        for param1,param2 in zip(self.h[j].parameters(),self.extrastorage[j].parameters()):
                            param1.data.copy_(param2.data, non_blocking=True)
                        self.h[j].to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                del cudastreams
                
            if i == 8:
                cudastreams = {}
                for j in range(8,16):
                    cudastreams[j] = torch.cuda.Stream()
                    for param1,param2 in zip(self.h[j].parameters(),self.h[j-8].parameters()):
                        param1.data = param2.data
                for j in range(8,16):
                    with torch.cuda.stream(cudastreams[j]):
                        for param1,param2 in zip(self.h[j].parameters(),self.extrastorage[j].parameters()):
                            param1.data.copy_(param2.data, non_blocking=True)
                        self.h[j].to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                del cudastreams
                    
            if i == 16:
                cudastreams = {}
                for j in range(16,24):
                    cudastreams[j] = torch.cuda.Stream()
                    for param1,param2 in zip(self.h[j].parameters(),self.h[j-8].parameters()):
                        param1.data = param2.data
                for j in range(16,24):
                    with torch.cuda.stream(cudastreams[j]):
                        for param1,param2 in zip(self.h[j].parameters(),self.extrastorage[j].parameters()):
                            param1.data.copy_(param2.data, non_blocking=True)
                        model.transformer.h[j].to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                del cudastreams
                
            if i == 24:
                cudastreams = {}
                for j in range(24,32):
                    cudastreams[j] = torch.cuda.Stream()
                    for param1,param2 in zip(self.h[j].parameters(),self.h[j-8].parameters()):
                        param1.data = param2.data
                for j in range(24,32):
                    with torch.cuda.stream(cudastreams[j]):
                        for param1,param2 in zip(self.h[j].parameters(),self.extrastorage[j].parameters()):
                            param1.data.copy_(param2.data, non_blocking=True)
                        self.h[j].to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                del cudastreams
                
        if number_of_parts == 32:
            
            if i == 0:
                for param1,param2 in zip(self.h[i].parameters(),self.h[31].parameters()):
                    param1.data = param2.data
                    
                for param1,param2 in zip(self.h[0].parameters(),self.extrastorage[0].parameters()):
                    param1.data = param2.data.to("cuda", non_blocking=True)
                self.h[0].to("cuda", non_blocking=True)
                    
                    
            if i >= 1:
                for param1,param2 in zip(self.h[i].parameters(),self.h[i-1].parameters()):
                    param1.data = param2.data
                    
                for param1,param2 in zip(self.h[i].parameters(),self.extrastorage[i].parameters()):
                    param1.data.copy_(param2.data, non_blocking=True)
                self.h[i].to("cuda", non_blocking=True)
        
        attn_type = self.config.attention_layers[i]
        attn_mask = global_attention_mask if attn_type == "global" else attention_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "gradient_checkpointing", False) and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attn_mask,
                head_mask[i],
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attn_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
GPTNeoModel.forward = new_forward


start_server = websockets.serve(hello, "127.0.0.1", 22255)
print("Server started, waiting for client")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()