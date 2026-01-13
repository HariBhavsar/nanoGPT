"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import matplotlib.pyplot as plt
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
# start_ids = encode(start)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# # run generation
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             print(decode(y[0].tolist()))
#             print('---------------')

# Part A.1
# Total inference time and inference time per output token as a function of varying size of input tokens
import time
input_token_sizes = [1, 1000, 2000, 3000, 4000, 5000]  # Different sizes of input tokens to test
num_samples = 1 # Will take average over these many samples
max_new_tokens = [1, 1000, 2000, 3000, 4000, 5000]  # Different sizes of output tokens to test
input_size_to_total_time = {}
input_size_to_avg_time_per_token = {}
output_size_to_total_time = {}
output_size_to_avg_time_per_token = {}
for size in input_token_sizes:
    average_total_times = []
    average_average_time_per_token = []
    for max_new_token in max_new_tokens:
        # First generate a random input of the specified size
        start_ids = torch.randint(low=0, high=model.config.vocab_size, size=(size,), dtype=torch.long)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    y = model.generate(x, max_new_token, temperature=temperature, top_k=top_k)
        print("Generated output for input size", size, "and output size", max_new_token)
        end_time = time.time()
        total_time = end_time - start_time
        average_time_per_token = total_time / max_new_token
        average_total_times.append(total_time/num_samples)
        average_average_time_per_token.append(average_time_per_token/num_samples)
    input_size_to_total_time[size] = torch.mean(torch.tensor(average_total_times))
    input_size_to_avg_time_per_token[size] = torch.mean(torch.tensor(average_average_time_per_token))
    for max_new_token in max_new_tokens:
        if max_new_token not in output_size_to_total_time:
            output_size_to_total_time[max_new_token] = 0
        output_size_to_total_time[max_new_token] += average_total_times[max_new_tokens.index(max_new_token)]
        if max_new_token not in output_size_to_avg_time_per_token:
            output_size_to_avg_time_per_token[max_new_token] = 0
        output_size_to_avg_time_per_token[max_new_token] += average_average_time_per_token[max_new_tokens.index(max_new_token)]
for max_new_token in max_new_tokens:
    output_size_to_total_time[max_new_token] /= len(input_token_sizes)
    output_size_to_avg_time_per_token[max_new_token] /= len(input_token_sizes)

# Plot the results
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.plot(list(input_size_to_total_time.keys()), list(input_size_to_total_time.values()), 
        marker='o', markersize=10, linewidth=2.5, color='#2E86AB', label='Total Time')
ax.set_title('Total Inference Time vs Input Token Size', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Input Token Size', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Inference Time (seconds)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=11)
fig.tight_layout()
plt.savefig("total_inference_time_vs_input_token_size.pdf", dpi=300, bbox_inches='tight')

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.plot(list(input_size_to_avg_time_per_token.keys()), list(input_size_to_avg_time_per_token.values()), 
        marker='o', markersize=10, linewidth=2.5, color='#2E86AB', label='Average Time Per Token')
ax.set_title('Average Inference Time Per Token vs Input Token Size', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Input Token Size', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Inference Time Per Token (seconds)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=11)
fig.tight_layout()
plt.savefig("average_inference_time_per_token_vs_input_token_size.pdf", dpi=300, bbox_inches='tight')

# Plot the results
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.plot(list(output_size_to_total_time.keys()), list(output_size_to_total_time.values()), 
        marker='o', markersize=10, linewidth=2.5, color='#2E86AB', label='Total Time')
ax.set_title('Total Inference Time vs Output Token Size', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Output Token Size', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Inference Time (seconds)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=11)
fig.tight_layout()
plt.savefig("total_inference_time_vs_output_token_size.pdf", dpi=300, bbox_inches='tight')

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.plot(list(output_size_to_avg_time_per_token.keys()), list(output_size_to_avg_time_per_token.values()), 
        marker='o', markersize=10, linewidth=2.5, color='#2E86AB', label='Average Time Per Token')
ax.set_title('Average Inference Time Per Token vs Output Token Size', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Output Token Size', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Inference Time Per Token (seconds)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=11)
fig.tight_layout()
plt.savefig("average_inference_time_per_token_vs_output_token_size.pdf", dpi=300, bbox_inches='tight')




