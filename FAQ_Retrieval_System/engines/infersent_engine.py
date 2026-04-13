import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseEngine

class InferSent(nn.Module):
    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort.copy()).to(sent.device)
        sent = sent.index_select(0, idx_sort)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted.copy(), batch_first=True)
        sent_output = self.enc_lstm(sent_packed)[0]
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]
        idx_unsort = torch.from_numpy(idx_unsort.copy()).to(sent.device)
        sent_output = sent_output.index_select(0, idx_unsort)
        if self.pool_type == "max":
            sent_output = torch.max(sent_output, 1)[0]
        elif self.pool_type == "mean":
            sent_output = torch.mean(sent_output, 1)
        return sent_output

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences):
        word_dict = {}
        for sent in sentences:
            for word in sent.split():
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<p>'] = ''
        return word_dict

    def build_vocab(self, sentences):
        assert hasattr(self, 'w2v_path'), 'word vectors path not set'
        word_dict = self.get_word_dict(sentences)
        self.word_vec = {}
        with open(self.w2v_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split(' ', 1)
                if len(parts) < 2: continue
                word, vec = parts
                if word in word_dict:
                    self.word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found {0} words with word vectors out of {1} words'.format(len(self.word_vec), len(word_dict)))

    def get_batch(self, batch):
        embed = np.zeros((len(batch), max([len(x) for x in batch]), self.word_emb_dim))
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                word = batch[i][j]
                if word in self.word_vec:
                    embed[i, j, :] = self.word_vec[word]
        return torch.from_numpy(embed).float()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        all_embeddings = []
        for i in range(0, len(sentences), bsize):
            batch_texts = sentences[i:i + bsize]
            batch_tokens = [s.split() if tokenize else s for s in batch_texts]
            lengths = np.array([max(len(x), 1) for x in batch_tokens])
            embed = self.get_batch(batch_tokens)
            if self.is_cuda():
                embed = embed.cuda()
            with torch.no_grad():
                embeddings = self.forward((embed, lengths))
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

class InferSentEngine(BaseEngine):
    def __init__(self, faqs, version=1, base_dir=None):
        super().__init__(faqs)
        self.name = f"InferSent (v{version})"
        self.version = version
        self.model = None
        self.faq_embeddings = None
        self.base_dir = base_dir or os.getcwd()
        self.model_path = os.path.join(self.base_dir, f"infersent{version}.pkl")
        self.w2v_path = None
        
        # Check potential word vector paths
        w2v_filename = "glove.840B.300d.txt" if version == 1 else "crawl-300d-2M.vec"
        possible_paths = [
            os.path.join(self.base_dir, w2v_filename),
            os.path.join(self.base_dir, "glove.840B.300d", w2v_filename) if version == 1 else ""
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                self.w2v_path = path
                break

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Error: InferSent weights not found at {self.model_path}")
            return False
            
        if not self.w2v_path or not os.path.exists(self.w2v_path):
            print(f"Error: Word vectors for InferSent not found.")
            return False

        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': self.version}
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.set_w2v_path(self.w2v_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        return True

    def train(self):
        print(f"Initializing {self.name}...")
        if not self._load_model():
            raise FileNotFoundError("Missing InferSent weights or word vectors.")
        questions = [f["question"] for f in self.faqs]
        self.model.build_vocab(questions)
        print(f"Encoding {len(self.faqs)} FAQs with InferSent...")
        self.faq_embeddings = self.model.encode(questions, tokenize=True, verbose=True)

    def get_similarity(self, query):
        if self.model is None:
            raise RuntimeError("InferSent model not trained/loaded.")
        query_emb = self.model.encode([query], tokenize=True)
        return cosine_similarity(query_emb, self.faq_embeddings).flatten()
