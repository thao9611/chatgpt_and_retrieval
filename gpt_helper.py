import torch
from transformers import AutoTokenizer, AutoModel
import json
import hnswlib
import torch.nn.functional as F
from torch import Tensor
from typing import List
import openai
import logging
from time import sleep, time
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


openai.organization = None
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""

NO_WIKI_PROMPT_TEMPLATE = """
Answer the following question:

Question: ```{q}```

Your response:
"""

HAVE_WIKI_PROMPT_TEMPLATE = """
You will be provided with the following information:
1. A question delimited with triple backticks.
2. Addition information that is related to the question.

Perform the following tasks:
1. Understand the provided information.
2. Use the provided information and answer the question.

Question: ```{q}```
Addition information: ```{info}```

Your response:
"""


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Average pooling for vector sequence
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def construct_message(role: str, content: str) -> dict:
    """
    Build message in Open API format
    """
    if role not in ("system", "user", "assistant"):
        raise ValueError("Invalid role")
    return {"role": role, "content": content}


def get_qa(prompt):
    """
    Given a prompt string, get ChatGPT answer
    """
    msgs = []
    msgs.append(construct_message("system", "You are a question-answer model."))
    msgs.append(construct_message("user", prompt))
    completion = get_chat_completion(messages=msgs)
    try:
        answer = str(completion["choices"][0]["message"]["content"])
        return answer
    except Exception as err:
        print(f"Could not extract the answer from the completion: {str(err)}")
        return


def get_chat_completion(
    messages: dict,
    model: str = "gpt-3.5-turbo",
    max_retries: int = 3,
    debug: bool = False,
):
    """
    Send request to chat completion API
    """
    model_dict = {"engine": "gpt_openapi", "model": model}
    error_msg = None
    error_type = None
    if debug:
        print(
            f"Sending chat with message={messages}, model_dict={model_dict}..."
        )
    for _ in range(max_retries):
        try:
            completion = openai.ChatCompletion.create(
                temperature=0.0, messages=messages, **model_dict
            )
            return completion
        except Exception as err:
            error_msg = str(err)
            error_type = type(err).__name__
            sleep(3)
    print(
        f"Could not obtain the completion after {max_retries} retries: `{error_type} ::"
        f" {error_msg}`"
    )
    return


class GPTHelper:
    def __init__(
        self,
        vector_model: str = "intfloat/multilingual-e5-large",
        input_file: str = "data/wiki/wikipedia_id.json",
        index_file: str = "data/wiki/index.bin",
        rebuild_index: bool = False,
        max_index_count: int = 1000000,
        debug: bool = False,
        dim: int = 1024,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(vector_model)
        self.model = AutoModel.from_pretrained(vector_model)
        self.model.eval()
        self.id2text = {}
        self.debug = debug
        self.max_index_count = max_index_count
        self.index_file = index_file
        self.dim = dim
        self._read_input_file(input_file)
        if (not os.path.exists(index_file)) or (rebuild_index):
            self._init_hnsw_index(input_file)
        else:
            self._load_hnsw_index()

    def _load_hnsw_index(self):
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        print(f"\nLoading index from {self.index_file}...\n")
        self.index.load_index(self.index_file, max_elements=self.max_index_count)

    def _init_hnsw_index(self, input_file):
        embeddings = self._get_doc_embedding()
        ids = list(self.id2text.keys())
        start = time()
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(
            max_elements=self.max_index_count, ef_construction=200, M=16
        )
        self.index.add_items(embeddings, ids)
        self.index.set_ef(50)  # ef should always be > k
        print(f"Finish building ann index, took {time()-start:.2f}s")
        self.index.save_index(self.index_file)

    def _read_input_file(self, input_file):
        with open(input_file, "r") as f:
            for line in f.readlines():
                try:
                    example = json.loads(line.strip("\n"))
                    self.id2text[example["id"]] = (
                        example.get("title", "")
                        + self.tokenizer.sep_token
                        + example.get("doc", "")
                    )
                except Exception as err:
                    print(err, " :", line)
                    continue
                if len(self.id2text) >= self.max_index_count:
                    break
        print(f"Read {len(self.id2text)} sample from input file {input_file}")
        return

    def map_id_to_text(self, label: List[int]):
        """
        Map id back to doc string
        """
        return [self.id2text[l] for l in label]

    @torch.no_grad()
    def _get_batch_embedding(self, batch: List[str]):
        batch_dict = self.tokenizer(
            batch, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        # normalize embeddings
        return F.normalize(embeddings, p=2, dim=1)

    def _get_doc_embedding(self):
        """
        Get embedding for all docs
        """
        data_loader = DataLoader(list(self.id2text.values()), batch_size=8)
        embeddings = []
        for batch in tqdm(data_loader):
            embeddings += self._get_batch_embedding(batch)
        embeddings = [e.tolist() for e in embeddings]
        return embeddings

    def get_nn(self, text: List[str], topk: int = 1):
        """
        Get nearest neighbor of list of queries
        """
        embeddings = self._get_batch_embedding(text)
        labels, distances = self.index.knn_query(embeddings.detach().numpy(), k=topk)
        nb_texts = [self.map_id_to_text(label) for label in labels]
        if self.debug:
            for i, t in enumerate(text):
                print(
                    f"Query={t}, neighbor_id={labels[i]} ,neighbor={nb_texts[i]}, distances={distances[i]}"
                )
        return nb_texts, labels, distances

    def get_qa_no_wiki(self, text):
        """
        Get anwser without extra wiki info
        """
        return get_qa(NO_WIKI_PROMPT_TEMPLATE.format(q=text))

    def get_qa_with_wiki(self, text):
        """
        Get anwser with extra wiki info
        """
        info = self.get_nn([text])[0]
        print("Wiki info: ", info)
        return get_qa(HAVE_WIKI_PROMPT_TEMPLATE.format(q=text, info=info))


def compare(question: str):
    """
    Compare anwser with different setting
    """
    print("Question: ", question)
    print("ChatGPT: ", gpt_helper.get_qa_no_wiki(question))
    print("ChatGPT + wiki info: ", gpt_helper.get_qa_with_wiki(question))


if __name__ == "__main__":
    gpt_helper = GPTHelper()
    compare("can you explain seq2seq model?")
