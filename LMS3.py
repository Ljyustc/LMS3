import os
from argparse import ArgumentParser
def get_args():
    parser = ArgumentParser(description='LMS3')
    parser.add_argument('--cuda', type=str, dest='cuda_id', default=None)
    args = parser.parse_args()
    return args

args = get_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
import json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import numpy as np

layer_id = -1
def get_last_layer_attention_weights(model):
    # find W_q, W_k, W_v of last layer
    last_layer = model.model.layers[layer_id].self_attn
    # print(last_layer.hidden_size, last_layer.num_heads, last_layer.head_dim, last_layer.num_key_value_heads)
    # 4096 32 128 8
    W_q = last_layer.q_proj.weight
    W_k = last_layer.k_proj.weight
    W_v = last_layer.v_proj.weight
    return W_q, W_k, W_v

ori_model_path  = "model_path"
config_kwargs = {
    "trust_remote_code": True,
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "output_hidden_states": True
}
config = AutoConfig.from_pretrained(ori_model_path, **config_kwargs)
model = AutoModelForCausalLM.from_pretrained(
    ori_model_path,
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    revision='main'
)
# print(model)
tokenizer = AutoTokenizer.from_pretrained(ori_model_path, **config_kwargs)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

w_q, w_k, w_v = get_last_layer_attention_weights(model)  # W_k: 1024*4096
w_k1 = w_k.reshape(8, 128, 4096).repeat(1, 4, 1).reshape(4096, 4096).T # num_key_value_heads = 8 for Llama3

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data

def write_jsonl(res, outfile):
    f = open(outfile, 'w', encoding='utf-8')
    for d in res:
        f.writelines(json.dumps(d, ensure_ascii=False))
        f.writelines('\n')
    f.close()

def get_representation(text):
    # get representation of problem text
    inputs = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True,
                                   max_length=128, return_tensors='pt')
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # get last hidden states
    representation = outputs.hidden_states[layer_id][:, -1, :] 
    return representation

def calculate_distance(x_A, x_B, batch_size=32):
    # W_q: 4096 * 4096
    # W_k1: 4096 * 4096
    # x_A, x_B: num_A * 4096, num_B * 4096
    # calculate W_k^T * W_q * x_B
    sta_vec = x_B @ w_v.T
    sta = torch.norm(sta_vec, dim=-1).detach().cpu().numpy()

    # calculate LLM-oriented semantic similarity
    distances = []
    num_A, num_B = x_A.size(0), x_B.size(0)
    for i in range(0, num_A, batch_size):
        x_A_batch = x_A[i:i+batch_size]
        batch_distances = []
        for j in range(0, num_B, batch_size):
            x_B_batch = x_B[j:j+batch_size]
            new_vector_batch = w_k1.T @ (w_q @ x_B_batch.T)
            new_vector_batch = new_vector_batch.T
            distance_batch = np.linalg.norm(x_A_batch.unsqueeze(1).detach().cpu().numpy() - new_vector_batch.unsqueeze(0).detach().cpu().numpy(), 2, axis=-1)  # batch_size * batch_size
            batch_distances.append(distance_batch)
            del new_vector_batch, distance_batch
            torch.cuda.empty_cache()
        batch_distances = np.concatenate(batch_distances, axis=1)  # batch_size * num_B
        distances.append(batch_distances)
        del x_A_batch, batch_distances
        torch.cuda.empty_cache()
    # distance = torch.norm(x_A.unsqueeze(1) - new_vector.unsqueeze(0), p=2, dim=-1)  # num_A * num_B
    return np.concatenate(distances, axis=0) * sta, np.concatenate(distances, axis=0)

def batch_process(corpus, batch_size=16):
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        batch_embeddings = get_representation(batch)
        embeddings.append(batch_embeddings)
        del batch_embeddings
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)

def construct_example(train_data, test_data, q_key, dataset, k=1):
    train_question_list = [x[q_key] for x in train_data]
    test_question_list = [x[q_key] for x in test_data]
    
    corpus_embeddings = batch_process(train_question_list, batch_size=1)
    target_embeddings = batch_process(test_question_list, batch_size=1)

    similarities, dis_similarities = calculate_distance(target_embeddings, corpus_embeddings)
    del corpus_embeddings, target_embeddings
    torch.cuda.empty_cache()
    dis_sorted_indices = np.argsort(dis_similarities, axis=1)
    sorted_indices = np.argsort(similarities, axis=1)
    max_sim_index = sorted_indices[:, :k]
    return dis_sorted_indices, max_sim_index

def get_files_in_folder(folder_path, sim_type, few_shot, l):
    one_shot_prompt1 = "Here are some examples you can refer to."
    one_shot_prompt2 = "\n\n---Now it's your turn!\n- Take a deep breath\n- Think step by step\n- I will tip $200\n\nQ: "
    data = []
    for file_name in os.listdir(folder_path+"/test"):
        train_file_path = os.path.join(folder_path+"/train", file_name)
        train_data = []
        for data_name in os.listdir(train_file_path):
            data_path = os.path.join(train_file_path, data_name)
            d = load_json(data_path)
            train_data.append(d)
        
        test_file_path = os.path.join(folder_path+"/test", file_name)
        test_data = []
        for data_name in os.listdir(test_file_path):
            data_path = os.path.join(test_file_path, data_name)
            d = load_json(data_path)
            test_data.append(d)
        
        sorted_indices, max_sim_index = construct_example(train_data, test_data, q_key='problem', dataset='MATH_'+file_name, k=few_shot)
        
        file_path = os.path.join(folder_path+"/test", file_name)
        for data_name_id in range(len(os.listdir(file_path))):
            data_name = os.listdir(file_path)[data_name_id]
            data_path = os.path.join(file_path, data_name)
            d = load_json(data_path)
            demon_indexs = max_sim_index[data_name_id]
            sorted_indice = sorted_indices[data_name_id]
            prompt = ""
            for demon_index in demon_indexs:
                if demon_index in sorted_indice[:int(l*len(sorted_indice))]: # int(distance.size(1))
                    prompt += "\n\nQ: " + train_data[demon_index]['problem'] + "\nA: " +  train_data[demon_index]['solution']
            if prompt != "":
                prompt = one_shot_prompt1 + prompt + one_shot_prompt2
            else:
                prompt = "Let's think step by step: "
            data.append({"id":file_name.replace(".json","")+"@"+data_name+"@one_shot_"+sim_type, "query":prompt+d['problem']+"\nA: "})
    return data

def gsm8k_data(path, sim_type, few_shot, l):
    one_shot_prompt1 = "Here are some examples you can refer to."
    one_shot_prompt2 = "\n\n---Now it's your turn!\n- Take a deep breath\n- Think step by step\n- I will tip $200\n\nQ: "
    data = []
    
    train_data = load_jsonl(path+"/train.jsonl")
    test_data = load_jsonl(path+"/test.jsonl")
    sorted_indices, max_sim_index = construct_example(train_data, test_data, q_key='question', dataset='GSM8K', k=few_shot)
     
    for i in range(len(test_data)):
        d = test_data[i]
        demon_indexs = max_sim_index[i]
        sorted_indice = sorted_indices[i]
        prompt = ""
        for demon_index in demon_indexs:
            if demon_index in sorted_indice[:int(l*len(sorted_indice))]: # int(distance.size(1))
                prompt += "\n\nQ: " + train_data[demon_index]['question'] + "\nA: " +  train_data[demon_index]['answer']
        if prompt != "":
            prompt = one_shot_prompt1 + prompt + one_shot_prompt2
        else:
            prompt = "Let's think step by step: "
        data.append({"id":"GSM8K_"+str(i)+"@one_shot_"+sim_type, "query":prompt+d['question']+"\nA: "})
    return data

def mawps_data(train_path, test_path, sim_type, few_shot, l):
    one_shot_prompt1 = "Here are some examples you can refer to."
    one_shot_prompt2 = "\n\n---Now it's your turn!\n- Take a deep breath\n- Think step by step\n- I will tip $200\n\nQ: "
    data = []
    
    train_data = load_json(train_path)
    test_data = load_json(test_path)
    sorted_indices, max_sim_index = construct_example(train_data, test_data, q_key='question', dataset='MAWPS', k=few_shot)
    
    for i in range(len(test_data)):
        d = test_data[i]
        demon_indexs = max_sim_index[i]
        sorted_indice = sorted_indices[i]
        prompt = ""
        for demon_index in demon_indexs:
            if demon_index in sorted_indice[:int(l*len(sorted_indice))]: # int(distance.size(1))
                prompt += "\n\nQ: " + train_data[demon_index]['question'] + "\nA: " + str(train_data[demon_index]['answer'])
        if prompt != "":
            prompt = one_shot_prompt1 + prompt + one_shot_prompt2
        else:
            prompt = "Let's think step by step: "
        data.append({"id":"MAWPS_"+str(i)+"@one_shot_"+sim_type, "query":prompt+d['question']+"\nA: "})
    return data

few_shot = 1
l = 0.01
few_shot_name = 'one_shot'
sim_type = 'LMS3'
output_file = "root_path/outputs.json"

math_crawl_data = get_files_in_folder("root_path/dataset/MATH/MATH", sim_type.lower(), few_shot, l)
gsm8k_crawl_data = gsm8k_data("root_path/dataset/GSM8K/grade-school-math-master/grade_school_math/data", sim_type.lower(), few_shot, l)
mawps_crawl_data = mawps_data("root_path/dataset/MAWPS/train.json", "root_path/dataset/MAWPS/dev.json", sim_type.lower(), few_shot, l)

crawl_data = math_crawl_data+mawps_crawl_data+gsm8k_crawl_data
write_jsonl(crawl_data, output_file)