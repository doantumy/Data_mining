# Load bibraries
import numpy as np
import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.utils.data import Dataset, random_split
from torchtext.data import Iterator, Field
from torchtext import data, datasets
import spacy
import json, re
from tensorboardX import SummaryWriter


print('Torch version: ',torch.__version__)
print('Torchtext version: ',torchtext.__version__)

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from tensorboardX import SummaryWriter
writer = SummaryWriter('tensorboard')

# Scaled Dot Product Attention
class ScaledDotProductAttention(nn.Module):
  # level = TensorLoggingLevels.attention # Logging level: 
  def __init__(self, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, key, value, mask=None):
    d_k = key.size()[-1]
    assert query.size(-1) == d_k
    scores = torch.matmul(query, key.transpose(-1,-2)) # [batch,?,?]
    # log_size(scores, "attention weight")
    if mask is not None:
      # Apply mask (padding) here with very small value -1e9
      scores = scores.masked_fill(mask==0,-1e9)
    p_att = F.softmax(scores /math.sqrt(d_k),dim=-1)
    p_att = self.dropout(p_att)
    # log_size(p_att, "softmax")
    output = torch.matmul(p_att, value)
    # log_size(output, "output")
    return output#, p_att

# Attention Head
class AttentionHead(nn.Module):
  # level = TensorLoggingLevels.attention_head
  def __init__(self, d_model, d_out, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.d_out = d_out
    self.query_linear = nn.Linear(d_model, d_out)
    self.key_linear = nn.Linear(d_model, d_out)
    self.value_linear = nn.Linear(d_model, d_out)
    self.attn = ScaledDotProductAttention(dropout)

  def forward(self, query, key, value, mask=None):
    query = self.query_linear(query)
    key = self.key_linear(key)
    value = self.value_linear(value)
    # log_size(query, "queries, keys, vals")
    return self.attn(query, key, value)

# Multihead Attention
class MultiheadAttention(nn.Module):
  # level = TensorLoggingLevels.attention_head
  def __init__(self, d_model, d_out, n_heads, dropout):
    super().__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_out = d_out
    # d_k = d_v = d_model / n_heads => d_model = d_k * n_heads
    assert d_model%n_heads == 0
    assert d_model == d_out * n_heads
    self.multiheads = nn.ModuleList([AttentionHead(d_model=d_model,d_out=d_out,dropout=dropout) for _ in range(n_heads)])
    self.linear = nn.Linear(d_out*n_heads, d_model)

  def forward(self, query, key, value, mask = None):
    #log_size(query, "queries, keys, vals")
    # Concat on the last dimension value => output will be [batch,sed,out*n_heads]
    multi_concat = torch.cat([head(query, key, value) for head in self.multiheads],dim=-1)
    #log_size(multi_concat, "concatenated output")
    # Linear must accept input of dimension out*n_heads and output dimension is d_model = 512 as in paper
    output = self.linear(multi_concat)
    #log_size(output, "projected output")
    return output 

# Layer Norm
class LayerNorm(nn.Module):
  def __init__(self,d_model, epsilon=1e-6):
    super().__init__()
    self.d_model = d_model
    self.beta = nn.Parameter(torch.zeros(d_model))
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.epsilon = epsilon

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return ((x - mean)/(std+self.epsilon))* self.gamma + self.beta # squareroot?

# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = nn.Dropout(dropout)
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)
  def forward(self, x):
    x = self.w_1(x)
    x = self.dropout(F.relu(x))
    x = self.w_2(x)
    return x

# Encoder Block
class EncoderBlock(nn.Module):
  # level = TensorLoggingLevels.enc_dec_block
  def __init__(self, d_model=512, d_out=64, d_ff=2048, n_heads=8, dropout=0.1):
    super().__init__()
    self.heads = MultiheadAttention(d_out * n_heads, d_out, n_heads, dropout)
    self.layernorm_1 = LayerNorm(d_model)
    self.feedforward = PositionwiseFeedForward(d_model,d_ff)
    self.layernorm_2 = LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    #log_size(x, "Encoder block input")
    att = self.heads(query=x, key=x, value=x, mask = None)
    #log_size(att, "Attention output")
    x = x + (self.dropout(self.layernorm_1(att)))
    pff = self.feedforward(x)
    #log_size(x, "Feedforward output")
    x = x + (self.dropout(self.layernorm_2(pff)))
    # log_size(x, "Encoder size output")
    return x

# Encoder
class Encoder(nn.Module):
  def __init__(self,d_model, d_out, d_ff, n_heads, n_blocks=6, dropout=0.1):
    super().__init__()
    self.encoderblocks = nn.ModuleList([EncoderBlock(d_model, d_out, d_ff, n_heads, dropout) for _ in range(n_blocks)])
  
  def forward(self, x: torch.FloatTensor, mask=None):
    # encode x for n_blocks times and apply mask (optional)
    for enc in self.encoderblocks:
      x = enc(x, mask)
    return x

# Decoder Block
class DecoderBlock(nn.Module):
  # level = TensorLoggingLevels.enc_dec_block
  def __init__(self, d_model=512, d_out=64, d_ff = 2048, n_heads=8, dropout=0.1):
    super().__init__()
    self.masked_att_heads = MultiheadAttention(d_model, d_out, n_heads, dropout)
    self.layernorm_1 = LayerNorm(d_model)
    self.att_heads = MultiheadAttention(d_model, d_out, n_heads, dropout)
    self.layernorm_2 = LayerNorm(d_model)
    self.feedforward = PositionwiseFeedForward(d_model,d_ff)
    self.layernorm_3 = LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
    self_att = self.masked_att_heads(query=x, key = x, value=x, mask=tgt_mask)
    x = x + self.dropout(self.layernorm_1(self_att))
    src_att = self.att_heads(query=x, key = enc_out, value=enc_out, mask=src_mask)
    x = x + self.dropout(self.layernorm_2(src_att))
    pff = self.feedforward(x)
    x = x + self.dropout(self.layernorm_3(pff))
    # log_size(x, 'decoder out')
    return x


# Decoder
class Decoder(nn.Module):
  def __init__(self, d_model, d_out, d_ff, n_heads, dropout, n_blocks=6) :
    super().__init__()
    self.decoderblocks = nn.ModuleList([DecoderBlock(d_model, d_out, d_ff, n_heads, dropout) for _ in range (n_blocks)])
    self.linear = nn.Linear(d_model, d_model)
  def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
    for decoder in self.decoderblocks:
      x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
    return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # level = TensorLoggingLevels.enc_dec_block
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # torch.arange returns a LongTensor, try torch.arange(0.0, d_model, 2) to force torch to return a FloatTensor instead.
        # It is possible that the torch exp and sin previously support LongTensor but maybe not anymore
        position = torch.arange(0., max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0., d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x):
        # log_size(self.weight[:, :x.size(1)] , "PositionalEncoding output")
        # log_size(x + self.weight[:, :x.size(1)] , "x + PositionalEncoding output")
        return self.weight[:, :x.size(1), :] # (1, Seq, Feature)

# Word Position Embeddings
class WordPositionEmbeddings(nn.Module):
  # level = TensorLoggingLevels.enc_dec_block
  def __init__(self, d_model=512, vocab_size=1000, dropout=0.1, max_len=5000):
    super().__init__()
    self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    self.position = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

  def forward(self, x: torch.LongTensor, mask=None) -> torch.FloatTensor:
    # print(x)
    # log_size(x,'wpos x')
    emb_out = self.embed(x)
    # log_size(emb_out, "emb_out")
    emb_pos = self.position(x)
    # log_size(emb_pos, "emb_pos")
    out = emb_out + emb_pos
    # log_size(out, "Embeddings out")
    return out


# FULL MODEL
class Generator(nn.Module):
  def __init__(self, d_model, vocab):
    super().__init__()
    self.linear = nn.Linear(d_model, vocab)

  def forward(self, x):
    return F.log_softmax(self.linear(x), dim=-1)

class EncoderDecoder(nn.Module):
  # level = TensorLoggingLevels.enc_dec_block
  def __init__(self, tgt_vocab=1000, src_vocab=1000, d_model=512, d_out=64, d_ff=2048, n_heads=8, n_blocks=6, dropout=0.1, max_len=5000):
    super().__init__()
    self.src_emb = WordPositionEmbeddings(d_model=d_model, vocab_size=src_vocab)
    self.tgt_emb = WordPositionEmbeddings(d_model=d_model, vocab_size=tgt_vocab)
    self.encoder = Encoder(d_model=d_model, d_out=d_out, d_ff=d_ff, n_heads=n_heads, n_blocks=n_blocks, dropout=dropout)
    self.decoder = Decoder(d_model=d_model, d_out=d_out, d_ff=d_ff, n_heads=n_heads, n_blocks=n_blocks, dropout=dropout)
    self.generator = Generator(d_model=d_model, vocab=tgt_vocab)

  def forward(self, src, tgt, src_mask, tgt_mask):
    encoder_out = self.encoder(x=self.src_emb(src), mask=src_mask)
    # log_size(encoder_out,'encoder_out')
    decoder_out = self.decoder(x=self.tgt_emb(tgt),enc_out=encoder_out, src_mask=src_mask, tgt_mask=tgt_mask)
    # log_size(decoder_out,'decoder_out')
    # output = self.generator(decoder_out)
    # log_size(output,'output')
    return decoder_out# output#, 

# TRAINING
# Batching and Masking
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(-2)
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return src_mask, tgt_mask

class Batch:
    def __init__(self, src, trg, src_mask, trg_mask, ntokens):
        self.src = src
        self.trg = trg
        self.src_mask = src_mask
        self.trg_mask = trg_mask
        self.ntokens = ntokens

# Optimizer
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model_size=512, 
                   factor=2, warmup=4000, 
                   optimizer=torch.optim.Adam(
                       model.parameters(), lr=0, 
                       betas=(0.9, 0.98), eps=1e-9))

# Label Smoothing
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# Loss
def loss_backprop(generator, criterion, out, targets, normalize):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []
    for i in range(out.size(1)):
        # print("out[:] ", out.size()) # [30, 9, 512]
        # print("out[:, i] ", out[:, i].size()) # [30, 512]
        out_column = Variable(out[:, i].data, requires_grad=True)
        # print("out_column ", out_column.size()) # == out[:, i].data == [30, 512]
        gen = generator(out_column)
        # print("gen.size()", gen.size()) # [30, 18]
        # print("targets[:, i]", targets[:, i].size()) # [30]
        # print("targets[:, i] ", targets[:, i]) 
        loss = criterion(gen, targets[:, i]) / normalize
        total += loss.item()#loss.data[0]
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)
    return total

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

train_file = 'train.json'
valid_file = 'val.json'
test_file = 'test.json'
SOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
SRC = Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
TGT = Field(tokenize=tokenize_en, pad_token=BLANK_WORD, init_token=SOS_WORD, eos_token=EOS_WORD)
fields = {'src':('src',SRC),'trg':('trg',TGT)}
train_data, val_data, test_data = data.TabularDataset.splits(
    format='json',
    path = 'data',
    train=train_file,
    validation=valid_file,
    test=test_file,
    fields=fields
)

MIN_FREQ = 1
SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
TGT.build_vocab(train_data.trg, min_freq=MIN_FREQ)

# Train Epoch
def train_epoch(train_iter, model, criterion, opt, transpose=False, SRC=SRC, TGT=TGT):
    model.train()
    best_loss = 1.0
    for i, batch in enumerate(train_iter):
        src, trg, src_mask, trg_mask = batch.src, batch.trg, batch.src_mask, batch.trg_mask
        # src, trg, src_mask, trg_mask = batch.src.cuda(), batch.trg.cuda(), batch.src_mask.cuda(), batch.trg_mask.cuda()
        # print('src ', src.size())
        out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        # print('Train out: ', out.size())
        # print("out = model.forward ", out.size()) # [30, 9, 512]
        # print("trg[:, 1:] ", trg[:, 1:].size()) # [30, 9]
        # print("trg[:, :] ", trg[:, :].size()) # [30, 10]
        # print("batch.trg ", batch.trg)

        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens) 
        if loss < best_loss:
          best_loss = loss
          print("Saved model at epoch: %d with best loss: %f"%(i, best_loss))
          torch.save(model.state_dict(), "./model/saved_model.pt")             
        model_opt.step()
        model_opt.optimizer.zero_grad()
        if i % 300 == 0:
          print("Epoch Step: %d Loss: %f" % (i, loss))
          # print('Train out: ', out)
          translate_output(model, batch, SRC, TGT)

# Validation epoch
def valid_epoch(valid_iter, model, criterion, transpose=False, SRC=SRC, TGT=TGT):
    model.eval()
    total = 0
    for i, batch in enumerate (valid_iter):
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens) 
        total += loss
        if i % 100 == 0:
          print('Validation out: ', out)
          print("         Batch %d Eval Loss: %f"%(i, loss))
          translate_output(model, batch, SRC, TGT)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.forward(src, Variable(ys), src_mask, Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
  
def test_model(model, test_iter, SRC, TGT):
    model.eval()
    for i, batch in enumerate(test_iter):
        print(batch)
        src = batch.src.transpose(0, 1)[:1]
        print(src)
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, 
                          max_len=50, start_symbol=TGT.vocab.stoi["<s>"])
        print(out)
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end =" ")
        print('=')

def translate_output(model,batch, SRC, TGT):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                          max_len=50, start_symbol=TGT.vocab.stoi["<s>"])
    print('out.size()', out.size())
    print("Translation:", end="\t")
    for i in range(1,out.size(1)):
        sym = TGT.vocab.itos[out[0,i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    src_mask, trg_mask = make_std_mask(src, trg, pad_idx)
    return Batch(src, trg, src_mask, trg_mask, (trg[1:] != pad_idx).data.sum())


# Create the model an load it onto our GPU.
model = EncoderDecoder(tgt_vocab=len(TGT.vocab), src_vocab=len(SRC.vocab), d_model=512, d_out=64, d_ff=1024, 
                       n_heads=8, n_blocks=6, dropout=0.5, max_len=300)
model_opt = get_std_opt(model)
model.cuda()
pad_idx = TGT.vocab.stoi["<blank>"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
BATCH_SIZE = 1024

train_iter = MyIterator(train_data, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val_data, batch_size=1, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)
test_iter = data.BucketIterator(test_data, batch_size=1, train=False, sort=False, repeat=False, device=device)

criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
for epoch in range(50):
    print('============== Epoch ',epoch,' ==============')
    train_epoch((rebatch(pad_idx, b) for b in train_iter), model, criterion, model_opt, writer)
    print("====== Start Validation ======")
    valid_epoch((rebatch(pad_idx, b) for b in valid_iter), model, criterion, writer)
    print("====== Testing ======")
    test_model(model,test_iter,SRC, TGT)
