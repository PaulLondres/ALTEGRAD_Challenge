from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

# SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# DistilGPT2
gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = AutoModel.from_pretrained("distilgpt2")

# RoBERTa
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = AutoModel.from_pretrained("roberta-base")

def SBERT_embedding(txt_prompt):
    with torch.no_grad():
        embeddings = sbert_model.encode(txt_prompt, convert_to_tensor=True)
    return embeddings


def DistilGPT2_embedding(txt_prompt, batch_size=None):
    inputs = gpt2_tokenizer(txt_prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = gpt2_model(**inputs)

    embeddings = outputs.last_hidden_state[:, -1, :]  # Taille : (batch_size, 768)
    return embeddings.squeeze()


def RoBERTa_embedding(txt_prompt, batch_size=None):
    inputs = roberta_tokenizer(txt_prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Taille : (batch_size, 768)
    return embeddings.squeeze()


def get_conditioning_vector(stats, txt_prompt, conditioning_method):
    if conditioning_method == "regex_parsing":
        cond_vec = stats
    elif conditioning_method == "SBERT":
        cond_vec = SBERT_embedding(txt_prompt)
    elif conditioning_method == "DistilGPT2":
        cond_vec = DistilGPT2_embedding(txt_prompt)
    elif conditioning_method == "RoBERTa":
        cond_vec = RoBERTa_embedding(txt_prompt)
    else:
        raise KeyError(f"Conditioning method {conditioning_method} not supported, following supported methods : "
                       f"'regex_parsing', 'SBERT', 'DistilGPT2', 'RoBERTa']")
    return cond_vec

if __name__ == "__main__":
    text_prompt = "This graph comprises 28 nodes and 165 edges. The average degree is equal to 11.785714285714286 and there are 387 triangles in the graph. The global clustering coefficient and the graph's maximum k-core are 0.4742647058823529 and 8 respectively. The graph consists of 3 communities."
    conditioning_method = "RoBERTa"
    print("Computing conditioning vectors...")
    cond_vec = get_conditioning_vector(None, text_prompt, conditioning_method)
    print(cond_vec, cond_vec.shape)