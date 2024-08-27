# %% [markdown]
# # <ins>M</ins>emory <ins>A</ins>ugmented <ins>Ma</ins>mba test notebook

# %%
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# %%
import json
import pyarrow.parquet as pq
import random
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--resume_epoch", type=int, default=-1)
parser.add_argument("--optim8bit", action="store_true")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

#%%
args = parser.parse_args()


# %% [markdown]
# ## Setup Datasets

# %%
# Load the memory corpus
MEMORY_PATH = "./data/mama_toy_memory.json"
DATA_PATH = "./data/mama_toy_chat.jsonl"
NOROBOTS_PATH = "./data/norobots_train.parquet"
TASK_SUMMARIZATION_PATH = "./data/instruction_summary_convo_dataset.jsonl"
DISTRACTOR_PATH = "./data/distract.txt"

# %%
memory_corpus = []
with open(MEMORY_PATH, "r") as f:
    memory_corpus = json.load(f)
len(memory_corpus)

# %%
toy_data = []
with open(DATA_PATH, "r") as f:
    for line in f:
        toy_data.append(json.loads(line))
len(toy_data)

# %%
norobots_pq = pq.read_table(
    NOROBOTS_PATH,
    columns=["messages", "category"],
)
norobots_pq = norobots_pq.to_pandas()
norobots_pq.head()

# %%
norobots_pq = norobots_pq[
    (norobots_pq["category"] == "Summarize")
    | (norobots_pq["category"] == "Rewrite")
    | (norobots_pq["category"] == "Generation")
]
norobots_pq.head()

# %%
len(norobots_pq)

# %%
norobots_all = norobots_pq["messages"].tolist()
norobots_chat = norobots_all[:-100]
norobots_memory = norobots_all[-100:]

# %%


# %%
for x in norobots_all:
    if x[-1]["role"] != "assistant":
        print(x)
        break

# %%
distractor_memory = []
with open(DISTRACTOR_PATH, "r") as f:
    for line in f:
        if len(line.strip()) == 0:
            continue
        distractor_memory.append(line)

# %%
# Let's load up the instruction summarization data
task_summarization_data = []
with open(TASK_SUMMARIZATION_PATH, "r") as f:
    for line in f:
        task_summarization_data.append(json.loads(line))
len(task_summarization_data)

# %%
# Let's sample some to add to the memory
task_summarization_memory = task_summarization_data[:30]
task_summarization_data = task_summarization_data[30:]
len(task_summarization_memory)

# %% [markdown]
# ## Setup Model Arch

# %%
# Setup the model archi
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
from monarch_i2i import MonarchI2i

# %%
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


# %%
class Mama(nn.Module):
    """Memory-based Retrieval + Mama Chat model"""

    def __init__(self, retrieval_path=None, model_path=None):
        super(Mama, self).__init__()
        self.retriever = MonarchI2i()
        if retrieval_path:
            self.retriever.load_state_dict(
                torch.load(retrieval_path, map_location="cpu")
            )
        self.generator = MambaLMHeadModel.from_pretrained(
            "state-spaces/mamba-2.8b", dtype=torch.bfloat16
        )
        self.retriever.train()
        self.generator.train()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def retrieve(self, query_r, embedded_corpus, k=3):
        """Retrieve k most similar items from corpus"""
        # Embed the query
        query = self.retriever.model.forward(query_r)
        # Use dot product to find the most similar
        scores_with_idx = []
        for i, item in enumerate(embedded_corpus):
            emb = item[0]
            scores_with_idx.append((self.cos(query, emb), i))
        # Sort by score
        scores_with_idx.sort(reverse=True, key=lambda x: x[0])
        # Return the top k
        return scores_with_idx[:k]

    def generate(self, query_g, embedded_corpus, memory_indices, **kwargs):
        """Generate a response from the query and the retrieved memory"""
        # Retrieve the memory
        memory = [embedded_corpus[i[1]][1] for i in memory_indices]
        # Input is memory + query
        input_ids = torch.cat(memory + [query_g], dim=1)
        # Generate the response
        response = self.generator(input_ids)
        return_augmented_input_ids = kwargs.get("return_augmented_input_ids", False)
        if return_augmented_input_ids:
            return response, input_ids
        return response

    def forward(self, query_r, query_g, embedded_corpus, k=3):
        """Forward pass"""
        # Retrieve the memory
        memory_indices = self.retrieve(query_r, embedded_corpus, k)
        # Generate the response
        response = self.generate(query_g, embedded_corpus, memory_indices)
        return response


# %%
mama = Mama(retrieval_path="./monarch_768_retrieval.pt")
mama

# %%
from transformers import AutoTokenizer

# %%
r_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
g_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


# %%
# Create the embedded corpus
def tokenize_memory_corpus(memory_corpus):
    corpus_r_tokens = []
    corpus_g_tokens = []

    for item in memory_corpus:
        r_tokens = r_tokenizer(item, return_tensors="pt", max_length=512)
        g_tokens = g_tokenizer(
            f"<|memory|>{item}{g_tokenizer.eos_token}",
            return_tensors="pt",
            max_length=512,
        )
        corpus_r_tokens.append(r_tokens)
        corpus_g_tokens.append(g_tokens)
    return corpus_r_tokens, corpus_g_tokens


corpus_r_tokens, corpus_g_tokens = tokenize_memory_corpus(memory_corpus)

# %%


# %%
mama.retriever.model.forward


# %%
def embed_corpus(corpus_r_tokens, corpus_g_tokens, device="cpu"):
    with torch.no_grad():
        embedded_corpus = []
        for r, g in zip(corpus_r_tokens, corpus_g_tokens):
            r_emb = mama.retriever.model.forward(r["input_ids"].to(device))
            embedded_corpus.append((r_emb, g["input_ids"].to(device)))
    return embedded_corpus


embedded_corpus = embed_corpus(corpus_r_tokens, corpus_g_tokens, device="cpu")

# %%
ORIGINAL_NAMES = ["Kaneema", "Minh", "Django", "Gumbs"]
REPLACEMENT_NAMES = [
    "Thomson",
    "Jerry",
    "Alice",
    "Rachel",
    "Ganeesh",
    "Adam",
    "Nic",
    "Veronica",
    "Sam",
    "Samantha",
    "Joe",
    "Donald",
    "Peter",
    "Paul",
    "Jorge",
] + ORIGINAL_NAMES
ORIGINAL_CITIES = ["New York", "Cape Town", "Los Angeles"]
REPLACEMENT_CITIES = [
    "London",
    "Paris",
    "Berlin",
    "Moscow",
    "Lagos",
    "Cairo",
    "Abuja",
] + ORIGINAL_CITIES
ORIGINAL_AGES = ["36", "27", "35"]
ORIGINAL_COLORS = ["Blue"]
REPLACEMENT_COLORS = [
    "Red",
    "Green",
    "Yellow",
    "Purple",
    "Black",
    "White",
] + ORIGINAL_COLORS


# %%
from tqdm import tqdm


# %%
def preprocess(
    conversations, r_tokenizer, g_tokenizer, conversation_template, max_tokens
):
    """
    Preprocess the data by tokenizing.
    """
    all_input_ids_r = []
    all_input_ids_g = []
    all_label_ids = []
    r_tokenizer.use_default_system_prompt = False
    r_tokenizer.eos_token = g_tokenizer.eos_token
    g_tokenizer.use_default_system_prompt = False

    print("Tokenizing dataset...")
    for conv in tqdm(conversations):
        current_conv = conv["messages"]
        tokenized_responses = []
        for msg in current_conv:
            if msg["role"] == "assistant":
                tokenized_responses.append(
                    g_tokenizer.encode(msg["content"], add_special_tokens=False)
                )

        tokenized_conv_r = r_tokenizer.apply_chat_template(
            current_conv,
            chat_template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'user:\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ 'system:\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ 'assistant:\n'  + message['content'] }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'assistant:' }}\n{% endif %}\n{% endfor %}",
            max_length=max_tokens,
            truncation=True,
        )
        tokenized_conv_g = g_tokenizer.apply_chat_template(
            current_conv,
            chat_template=conversation_template,
            max_length=max_tokens,
            truncation=True,
        )
        tokenized_labels = g_tokenizer.apply_chat_template(
            [current_conv[-1]],
            chat_template=conversation_template,
            max_length=max_tokens,
            truncation=True,
        )
        all_input_ids_g.append(torch.LongTensor(tokenized_conv_g))
        all_input_ids_r.append(torch.LongTensor(tokenized_conv_r))
        all_label_ids.append(torch.LongTensor(tokenized_labels))
    return {
        "input_ids_r": all_input_ids_r,
        "input_ids_g": all_input_ids_g,
        "label_ids": all_label_ids,
    }


# %%
mama_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
print(mama_template)

# %%
import re
import random


def randomize_dataset(device="cpu", epoch=0):
    """Replace the names, cities and ages in the dataset"""
    kaneema_to = random.choice(REPLACEMENT_NAMES)
    gumbs_to = random.choice(REPLACEMENT_NAMES)
    minh_to = random.choice(REPLACEMENT_NAMES)
    django_to = random.choice(REPLACEMENT_NAMES)
    new_york_to = random.choice(REPLACEMENT_CITIES)
    cape_town_to = random.choice(REPLACEMENT_CITIES)
    los_angeles_to = random.choice(REPLACEMENT_CITIES)
    age1_to = str(random.randint(19, 60))
    age2_to = str(random.randint(18, int(age1_to) - 1))
    age3_to = str(random.randint(18, 60))
    color_to = random.choice(REPLACEMENT_COLORS)

    randomized_toy_data = []
    randomized_memory_corpus = []

    for conv in toy_data:
        new_conv = []
        for msg in conv["messages"]:
            new_msg = {}
            new_msg["role"] = msg["role"]
            new_msg["content"] = msg["content"]
            new_msg["content"] = new_msg["content"].replace("Kaneema", kaneema_to)
            new_msg["content"] = new_msg["content"].replace("Gumbs", gumbs_to)
            new_msg["content"] = new_msg["content"].replace("Minh", minh_to)
            new_msg["content"] = new_msg["content"].replace("Django", django_to)
            new_msg["content"] = new_msg["content"].replace("New York", new_york_to)
            new_msg["content"] = new_msg["content"].replace("Cape Town", cape_town_to)
            new_msg["content"] = new_msg["content"].replace(
                "Los Angeles", los_angeles_to
            )
            new_msg["content"] = new_msg["content"].replace("36", age1_to)
            new_msg["content"] = new_msg["content"].replace("27", age2_to)
            new_msg["content"] = new_msg["content"].replace("35", age3_to)
            pattern = re.compile("blue", re.IGNORECASE)
            new_msg["content"] = pattern.sub(color_to, new_msg["content"])

            new_conv.append(new_msg)

        randomized_toy_data.append({"messages": new_conv})
    for x in memory_corpus:
        new_x = x
        new_x = new_x.replace("Kaneema", kaneema_to)
        new_x = new_x.replace("Gumbs", gumbs_to)
        new_x = new_x.replace("Minh", minh_to)
        new_x = new_x.replace("Django", django_to)
        new_x = new_x.replace("New York", new_york_to)
        new_x = new_x.replace("Cape Town", cape_town_to)
        new_x = new_x.replace("Los Angeles", los_angeles_to)
        new_x = new_x.replace("36", age1_to)
        new_x = new_x.replace("27", age2_to)
        new_x = new_x.replace("35", age3_to)
        pattern = re.compile("blue", re.IGNORECASE)
        new_x = pattern.sub(color_to, new_x)
        randomized_memory_corpus.append(new_x)

    if epoch > 20:
        # Now we add the norobots data and the task summarization data
        for conv in norobots_chat:
            randomized_toy_data.append(
                {
                    "messages": conv,
                }
            )
        for x in norobots_memory:
            mem = "\n".join(f"{x['role']}: {x['content']}" for x in x)
            randomized_memory_corpus.append(mem)

        for conv in task_summarization_data:
            randomized_toy_data.append(
                {
                    "messages": conv,
                }
            )
        for x in task_summarization_memory:
            mem = "\n".join(f"{x['role']}: {x['content']}" for x in x)
            randomized_memory_corpus.append(mem)

        for x in distractor_memory:
            randomized_memory_corpus.append(x)

    # Update the embedded_corpus
    global corpus_r_tokens, corpus_g_tokens
    corpus_r_tokens, corpus_g_tokens = tokenize_memory_corpus(randomized_memory_corpus)
    global embedded_corpus
    embedded_corpus = embed_corpus(corpus_r_tokens, corpus_g_tokens, device=device)

    # Update the dataset
    global toy_data_preprocessed
    toy_data_preprocessed = preprocess(
        randomized_toy_data,
        r_tokenizer,
        g_tokenizer,
        mama_template,
        max_tokens=1024,
    )
    return toy_data_preprocessed, embedded_corpus


# %%
randomize_dataset()

# %%
toy_data_preprocessed = preprocess(
    toy_data, r_tokenizer, g_tokenizer, mama_template, 2048 * 4
)

# %%
toy_data_preprocessed["input_ids_g"][0]

# %%
g_tokenizer.decode(toy_data_preprocessed["input_ids_g"][0])

# %%
g_tokenizer.eos_token

# %%
mama = mama.cuda()

# %%
out = mama.generator.forward(
    toy_data_preprocessed["input_ids_g"][0].unsqueeze(0).cuda()
)
out

# %%
g_tokenizer.decode(out.logits[0].argmax(dim=-1))

# %%
# move embedded corpus to GPU
embedded_corpus = [(r_emb.cuda(), g_emb.cuda()) for r_emb, g_emb in embedded_corpus]

# %%
out2 = mama.forward(
    query_r=toy_data_preprocessed["input_ids_r"][0].unsqueeze(0).cuda(),
    query_g=toy_data_preprocessed["input_ids_g"][0].unsqueeze(0).cuda(),
    embedded_corpus=embedded_corpus,
)

# %%
out2

# %%
g_tokenizer.decode(out2.logits[0].argmax(dim=-1))

# %%
toy_data[0]

# %%
out3 = mama.retrieve(
    query_r=toy_data_preprocessed["input_ids_r"][0].unsqueeze(0).cuda(),
    embedded_corpus=embedded_corpus,
)

# %%
out3

# %%
memory_corpus[0]

# %%
import torch.optim as optim

# %%
mama.train()
mama.cuda()

# %%
EPOCHS = 30
BATCH_SIZE = 1

# %%
optimizer1 = optim.Adam(mama.generator.parameters(), lr=5e-5)
optimizer2 = optim.Adam(mama.retriever.parameters(), lr=3e-6, weight_decay=5e-6)

if args.optim8bit:
    import bitsandbytes as bnb
    optimizer1 = bnb.optim.Adam8bit(mama.generator.parameters(), lr=5e-5)
    # optimizer2 = bnb.optim.Adam8bit(mama.retriever.parameters(), lr=3e-6, weight_decay=5e-6)

# %%
toy_data_preprocessed

# %%
import numpy as np

# %%
exp = torch.exp

# %%
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# %%

START_EPOCH = args.resume_epoch + 1 if args.resume_epoch > 0 else 0
if args.resume_epoch >= 0:
    mama.load_state_dict(torch.load(f"./mama_in_progress_epoch_{args.resume_epoch}.pt"))
    toy_data_preprocessed, embedded_corpus = randomize_dataset(
        device="cuda", epoch=START_EPOCH
    )

# %%
# We finetune in an alternating pattern, first generator, then retriever
for epoch in tqdm(range(START_EPOCH, EPOCHS)):
    indices = np.arange(len(toy_data_preprocessed["input_ids_r"]))
    np.random.shuffle(indices)
    print(f"Epoch {epoch}")
    print("-----------------------------------")
    for i in range(0, len(indices), BATCH_SIZE):
        index_slice = indices[i : i + BATCH_SIZE]
        batch_x_r = [toy_data_preprocessed["input_ids_r"][i] for i in index_slice]
        batch_x_g = [toy_data_preprocessed["input_ids_g"][i] for i in index_slice]
        batch_x_g_len = torch.tensor([len(i) for i in batch_x_g])

        batch_y_g = [toy_data_preprocessed["label_ids"][i] for i in index_slice]
        batch_y_g_len = torch.tensor([len(i) for i in batch_y_g])

        batch_x_r = torch.nn.utils.rnn.pad_sequence(
            batch_x_r, batch_first=True, padding_value=0
        ).cuda()
        batch_x_g = torch.nn.utils.rnn.pad_sequence(
            batch_x_g, batch_first=True, padding_value=0
        ).cuda()
        batch_y_g = torch.nn.utils.rnn.pad_sequence(
            batch_y_g, batch_first=True, padding_value=0
        ).cuda()
        print("index_slice")
        print(index_slice)

        # Optimize Generator
        top_k = mama.retrieve(batch_x_r, embedded_corpus, k=3)
        print("top_k")
        print(top_k)

        out, augmented_input_ids = mama.generate(
            query_g=batch_x_g,
            embedded_corpus=embedded_corpus,
            memory_indices=top_k,
            return_augmented_input_ids=True,
        )
        logits = out.logits
        labels = augmented_input_ids[:, 1:].cuda().contiguous()
        labels_r = batch_y_g[:, 6:].cuda().contiguous()
        # labels = batch_x_g[:, 1:].cuda().contiguous()
        shift_offset = (
            5
            + (
                logits.shape[1] - ((batch_x_g.shape[1] - batch_x_g_len) + batch_y_g_len)
            ).cuda()
        )
        shift_logits_r = logits[:, shift_offset:-1, :].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        # shift_offset = (logits.shape[1] - ((batch_x_g.shape[1] - batch_x_g_len))).cuda()
        print("shift_logits_shape")
        print(shift_logits.shape)
        print("-----------")
        # Decode logits
        decoded_logits = g_tokenizer.decode(logits[0].argmax(dim=-1))
        print("decoded_logits")
        print(decoded_logits)

        # Decode labels
        decoded_labels = g_tokenizer.decode(labels[0])
        print("decoded_labels")
        print(decoded_labels)
        print("---------------")

        # Sample 3 random memories
        rand_memory_indices = np.random.choice(len(embedded_corpus), 3)
        rand_memory_indices = [(None, i) for i in rand_memory_indices]
        print(rand_memory_indices)

        out2 = mama.generate(batch_x_g, embedded_corpus, rand_memory_indices)
        shift_offset2 = (
            5
            + (
                out2.logits.shape[1]
                - ((batch_x_g.shape[1] - batch_x_g_len) + batch_y_g_len)
            ).cuda()
        )
        shift_logits2 = out2.logits[:, shift_offset2:-1, :].contiguous()

        # Decode logits
        decoded_logits2 = g_tokenizer.decode(out2.logits[0].argmax(dim=-1))
        print("decoded_logits2")
        print(decoded_logits2)
        print("-------------")

        print("labels shape")
        print(labels.shape)
        print(labels)

        generator_loss_fn = torch.nn.CrossEntropyLoss()
        loss = generator_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )
        if args.gradient_accumulation_steps == 1:
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(mama.generator.parameters(), 1.0)
            optimizer1.step()
        else:
            loss.backward(retain_graph=True)
            if i % args.gradient_accumulation_steps == 0:
                optimizer1.step()
                optimizer1.zero_grad()
        # Print loss
        generator_loss = loss.item()
        print(loss.item())
        # Optimize Retriever

        # Get embedding for the query
        query_r_emb = mama.retriever.model.forward(batch_x_r)

        print("shift_logits_r shape")
        print(shift_logits_r.shape)

        print("shift_logits2 shape")
        print(shift_logits2.shape)

        # Calculate s_retrieval, the probability of the generated response given the memory
        s_retrieval = (
            F.log_softmax(shift_logits_r, dim=2)
            .gather(dim=2, index=labels_r.unsqueeze(2))
            .squeeze(2)
            .sum(dim=1)
        )
        # Calculate s_random, the probability of the generated response given a random memory
        s_random = (
            F.log_softmax(shift_logits2, dim=2)
            .gather(dim=2, index=labels_r.unsqueeze(2))
            .squeeze(2)
            .sum(dim=1)
        )

        print("---------- Retriever ------------")

        print("s_retrieval shape")
        print(s_retrieval)

        print("s_random shape")
        print(s_random)

        print("s_retrieval > s_random")
        print(s_retrieval > s_random)

        # s_r1 = exp(s_retrieval  ) / (exp(s_retrieval ) + exp(s_random ))
        # s_r2 = exp(s_random  ) / (exp(s_retrieval ) + exp(s_random ))

        print("s_r1 shape")
        # print(s_r1.shape)

        print("s_r2 shape")
        # print(s_r2.shape)

        # s_r = torch.stack([s_r1, s_r2], dim=1)
        s_r = F.softmax(torch.stack([s_retrieval, s_random], dim=1), dim=1)

        print(top_k[0][1])

        # Calculate the average embedding of the items in the memory
        avg_memory_emb_r = torch.stack(
            [
                mama.retriever.model.forward(corpus_r_tokens[i[1]]["input_ids"].cuda())
                for i in top_k
            ]
        ).mean(dim=0)
        # mama.retriever.cpu()
        # avg_memory_emb_r = torch.stack(
        #     [
        #         mama.retriever.model.forward(corpus_r_tokens[i[1]]["input_ids"])
        #         for i in top_k
        #     ]
        # ).mean(dim=0)

        # Calculate the average embedding of t
        # he items in the random memory
        avg_random_emb_r = torch.stack(
            [
                mama.retriever.model.forward(corpus_r_tokens[i[1]]["input_ids"].cuda())
                for i in rand_memory_indices
            ]
        ).mean(dim=0)

        # Calculate the cosine similarity between the query and the average memory embedding
        a_memory = cos(query_r_emb, avg_memory_emb_r)

        # Calculate the cosine similarity between the query and the average random memory embedding
        a_random = cos(query_r_emb, avg_random_emb_r)

        # a_r1 = exp(a_memory ) / (exp(a_memory ) + exp(a_random ))
        # a_r2 = exp(a_random ) / (exp(a_memory ) + exp(a_random ))

        # a_r = torch.stack([a_r1, a_r2], dim=1)
        a_r = torch.softmax(torch.stack([a_memory, a_random], dim=1), dim=1)

        print("a_r shape")
        print(a_r.shape)

        print("s_r shape")
        print(s_r.shape)

        print("a_r s_r")
        print(a_r)
        print(s_r)

        # Minimize the KL divergence between a_r and s_r
        if epoch > 10:
            kl_loss = torch.nn.KLDivLoss()
            loss = kl_loss(a_r.log(), s_r.detach())
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mama.retriever.parameters(), 1.0)
            optimizer2.step()
        # Print loss
        print("Losses")
        print("-----------------------------")
        print("Generator Loss")
        print(generator_loss)
        print("KL loss")
        print(loss.item())
    # Save the model
    torch.save(mama.state_dict(), f"./mama_in_progress_epoch_{epoch}.pt")
    # Randomize the dataset
    toy_data_preprocessed, embedded_corpus = randomize_dataset(
        device="cuda", epoch=epoch
    )


# %%
len(embedded_corpus)

# %%
top_k

# %%
corpus_r_tokens[2]

# %%


# %%
out3 = mama.forward(
    query_r=toy_data_preprocessed["input_ids_r"][0].unsqueeze(0).cuda(),
    query_g=toy_data_preprocessed["input_ids_g"][0].unsqueeze(0).cuda(),
    embedded_corpus=embedded_corpus,
)

# %%
torch.save(mama.state_dict(), "./mama_toy.pt")

# %%
out3

# %%
s = "'<|system|>\nAnswer the given question<|endoftext|>\n<|user|>\nHow old is Minh?<|endoftext|>\n"
s_tokens = g_tokenizer.encode(s, add_special_tokens=False, return_tensors="pt")
s_tokens_r = r_tokenizer.encode(s, add_special_tokens=False, return_tensors="pt")
s_tokens.shape

# %%
s_tokens.cuda()

# %%
out3s = mama.forward(
    query_r=s_tokens_r.cuda(), query_g=s_tokens.cuda(), embedded_corpus=embedded_corpus
)

# %%
topks = mama.retrieve(query_r=s_tokens_r.cuda(), embedded_corpus=embedded_corpus)
topks

# %%
out3s = mama.generate(
    query_g=s_tokens.cuda(), embedded_corpus=embedded_corpus, memory_indices=topks
)

# %%
g_tokenizer.decode(out3s.logits[0].argmax(dim=-1))

# %%
decoded_logits

# %%
rand_memory_indices = np.random.choice(len(memory_corpus), 3)
rand_memory_indices = [(None, i) for i in rand_memory_indices]

# %%
out4 = mama.generate(
    query_g=toy_data_preprocessed["input_ids_g"][0].unsqueeze(0).cuda(),
    embedded_corpus=embedded_corpus,
    memory_indices=rand_memory_indices,
)

# %%
g_tokenizer.decode(out4.logits[0].argmax(dim=-1))

# %%
topk = mama.retrieve(
    query_r=toy_data_preprocessed["input_ids_r"][0].unsqueeze(0).cuda(),
    embedded_corpus=embedded_corpus,
)

out5 = mama.generate(
    query_g=toy_data_preprocessed["input_ids_g"][0].unsqueeze(0).cuda(),
    embedded_corpus=embedded_corpus,
    memory_indices=topk,
)

# %%
g_tokenizer.decode(out5.logits[0].argmax(dim=-1))

# %%
g_tokenizer.encode("<|assistant|>36", return_tensors="pt")


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
count_parameters(mama)

# %%
len(toy_data_preprocessed["input_ids_g"])

# %%
