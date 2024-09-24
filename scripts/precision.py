from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from react import *
import torch

# 'teknium/OpenHermes-2.5-Mistral-7B'
access_token = "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"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
print("begin test")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=access_token,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
print('tokenizer loaded')

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # {'': 'cuda:0'},
    quantization_config=double_quant_config,
    use_flash_attention_2=False,
    low_cpu_mem_usage=True,
    token=access_token,
)

embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("model loaded")


def cosine_similarity(x, y):
    return np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)


questions = [
    "When do you want the simulation to start ?",
    "When do you want the simulation to end ?",
    "How many electric vehicles do you own ?",
    "Where do you live ?",
    "When do you come back from work ?",
    "When do you leave your house ?",
    "What is your house minimum comfort temperature ?",
    "What is your house maximum comfort temperature ?",
]
best_answers = [
    "I want the simulation to start on the $VAL.",
    "I want the simulation to end on the $VAL.",
    "I own $VAL electric vehicles.",
    "I live in $VAL.",
    "I come back at $VAL.",
    "I leave my house at $VAL.",
    "My house minimum comfort temperature is $VAL degrees celsius.",
    "My house maximum comfort temperature is $VAL degrees celsius.",]

scores = {'easy': [[] for i in range(8)],
          'medium': [[] for i in range(8)],
          'hard': [[] for i in range(8)]}
answers = {'easy': [[] for i in range(8)],
           'medium': [[] for i in range(8)],
           'hard': [[] for i in range(8)]}

for i in range(20):
    print(i)
    for mode in ['easy', 'medium', 'hard']:
        user = User(model, tokenizer, path=None, mode=mode)
        user_params = [user.date_start, user.date_end, user.ev, user.city,
                       user.arrival_time, user.leaving_time, user.tmin, user.tmax]
        for k, question in enumerate(questions):
            param = user_params[k]
            if k < 2:
                param = param.strftime("%Y/%m/%d")
            user_answer = user(question)
            answer = Template(best_answers[k]).safe_substitute(VAL=param)
            answers[mode][k].append(user_answer)
            scores[mode][k].append(cosine_similarity(embedding.encode(
                question + '\n' + answer), embedding.encode(question + '\n' + user_answer)))

mean_scores = {'Easy (E)': [],
               'Medium (M)': [],
               'Hard (H)': []}

difficulty_levels = ["easy", "medium", "hard"]
for i, mode in enumerate(difficulty_levels):
    for k in range(8):
        mean_scores[list(mean_scores.keys())[i]].append(
            100*np.mean(scores[mode][k]))

labels = ['date start', 'date end', 'EV', 'city',
          'start time', 'end time', 'Tmin', 'Tmax']
difficulty_levels = ["easy", "medium", "hard"]
df = pd.DataFrame(mean_scores)
df = df.rename(index={i: labels[i] for i in range(len(labels))})
df.to_csv('../data/cosine')

f, ax = plt.subplots()
h = sns.heatmap(df, cmap="Greens", cbar=True, ax=ax,
                annot=False, linewidths=0.1, linecolor='black')
h.set_yticklabels(labels, fontsize=15)
h.set_xticklabels(mean_scores.keys(), fontsize=15)
cbar = ax.collections[0].colorbar
cbar.set_label('Cosine similarity', rotation=270, fontsize=15, labelpad=16)
plt.savefig("../img/cosine_similarity_test.pdf",
            format='pdf', bbox_inches='tight')

with open("../data/scores_precisions", 'w', encoding='utf-8') as fscores:
    json.dump(scores, fscores, ensure_ascii=False, indent=4)
    fscores.close()
with open("../data/answers_precisions", 'w', encoding='utf-8') as fscores:
    json.dump(answers, fscores, ensure_ascii=False, indent=4)
    fscores.close()
