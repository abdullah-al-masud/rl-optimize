import torch
import numpy as np


class PN_Actor(torch.nn.Module):
    
    def __init__(self, config, dtype=torch.float32):
        super(PN_Actor, self).__init__()
        
        self.config = config
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' in config.device else "cpu")
        self.T = 1
        
        self.build_model()
        
    def build_model(self,):
        
        init = torch.nn.init.xavier_uniform_
        
        # encoder structure
        self.embedder1 = torch.nn.Conv1d(self.config.dimension, self.config.hidden1, 1).to(device=self.device)
        self.lstm_enc = torch.nn.LSTMCell(self.config.hidden1, self.config.hidden2).to(device=self.device)
        self.lstm_enc_c0 = torch.nn.parameter.Parameter(init(torch.empty((1, self.config.hidden2), dtype=self.dtype, device=self.device)).repeat(self.config.batch_size, 1))
        self.lstm_enc_h0 = torch.nn.parameter.Parameter(init(torch.empty((1, self.config.hidden2), dtype=self.dtype, device=self.device)).repeat(self.config.batch_size, 1))
        
        # decoder structure
        self.lstm_dec = torch.nn.LSTMCell(self.config.hidden1, self.config.hidden2).to(device=self.device)
        self.dec_input0 = torch.nn.parameter.Parameter(init(torch.empty((1, self.config.hidden1), dtype=self.dtype, device=self.device)).repeat(self.config.batch_size, 1))
        self.lstm_dec_c0 = torch.nn.parameter.Parameter(init(torch.empty((1, self.config.hidden2), dtype=self.dtype, device=self.device)).repeat(self.config.batch_size, 1))
        
        # glimpse and pointing parameters
        self.Wref_g = torch.nn.parameter.Parameter(init(torch.empty((self.config.hidden2, self.config.hidden2), dtype=self.dtype, device=self.device)))
        self.Wq_g = torch.nn.parameter.Parameter(init(torch.empty((self.config.hidden2, self.config.hidden2), dtype=self.dtype, device=self.device)))
        self.v_g = torch.nn.parameter.Parameter(init(torch.empty((self.config.hidden2, 1), dtype=self.dtype, device=self.device)))
        
        self.Wref = torch.nn.parameter.Parameter(init(torch.empty((self.config.hidden2, self.config.hidden2), dtype=self.dtype, device=self.device)))
        self.Wq = torch.nn.parameter.Parameter(init(torch.empty((self.config.hidden2, self.config.hidden2), dtype=self.dtype, device=self.device)))
        self.v = torch.nn.parameter.Parameter(init(torch.empty((self.config.hidden2, 1), dtype=self.dtype, device=self.device)))
        
        # other initializations
        self.zero_mask = torch.zeros(self.config.batch_size, self.config.problem_size).to(dtype=self.dtype, device=self.device)
        self.neg_inf = 1e8
        self.range_index = torch.tensor(list(range(self.config.batch_size)), device=self.device).long()
        self.start_index_tensor = (torch.ones(self.config.batch_size) * self.config.start_index).to(device=self.device).long()
        
    def encoder(self, x):
        embed = self.embedder1(x.transpose(1, 2)).transpose(1, 2)
        # print('embed.shape:', embed.shape, embed.min())
        enc = []
        h = self.lstm_enc_h0
        c = self.lstm_enc_c0
        for i in range(x.shape[1]):
            h, c = self.lstm_enc(embed[:, i, :], (h, c))
            enc.append(h)
        enc = torch.stack(enc).transpose(0, 1)
        # print('enc.shape:', enc.shape, enc.min())
        return enc, h, c, embed
    
    
    def glimpse_pointing(self, enc, embed, wenc, wenc_g, q, mask):
        # glimpse
        g = q
        for i in range(self.config.n_glimpse):
            u_g = torch.matmul(torch.tanh(wenc_g + torch.matmul(g, self.Wq_g).unsqueeze(axis=1)), self.v_g).squeeze()
            u_g = torch.nn.functional.softmax(self.config.c * torch.tanh(u_g) - mask * self.neg_inf, dim=1).unsqueeze(axis=2)
            # print('u_g.shape:', u_g.shape, u_g.min())
            g = (enc * u_g).sum(axis=1) + q
            # print('g.shape:', g.shape, g.min())
        
        # pointing
        u = torch.matmul(torch.tanh(wenc + torch.matmul(g, self.Wq).unsqueeze(axis=1)), self.v).squeeze()
        u = torch.nn.functional.softmax(self.config.c * torch.tanh(u) - mask * self.neg_inf, dim=1)
        # print('u.shape:', u.shape)
        
        # next action
        if self.config.greedy:
            # greedy method
            next_index = u.argmax(dim=1)
        else:
            # sampling
            m = torch.distributions.categorical.Categorical(probs=u)
            next_index = m.sample()

        # updating mask
        log_u = u[self.range_index, next_index]
        next_input = embed[self.range_index, next_index, :]
        mask += torch.nn.functional.one_hot(next_index, self.config.problem_size)

        return mask, log_u, next_index, next_input
    
    
    def decoder(self, enc, enc_h, embed):
        # decoder lstm initialization
        h = enc_h
        c = self.lstm_dec_c0
        mask = self.zero_mask.clone()
        log_prob, indices = [], []
        
        # not using start index 
        start = 0
        x = self.dec_input0
        
        # using start index
#         indices.append(self.start_index_tensor)
#         x = embed[:, self.config.start_index, :]
#         mask[:, self.config.start_index] = 1
#         start = 1

        # decoder poiting partial calculation
        wenc = torch.matmul(enc, self.Wref)
        wenc_g = torch.matmul(enc, self.Wref_g)
        # print('wenc.shape:', wenc.shape, wenc.min(), '; wenc_g.shape:', wenc_g.shape, wenc_g.min())
        # decoder rollout loop
        for i in range(start, self.config.problem_size):
            h, c = self.lstm_dec(x, (h, c))
            # print('h.shape:', h.shape, h.min())
            
            # glimpse and pointing
            mask, log_u, index, x = self.glimpse_pointing(enc, embed, wenc, wenc_g, h, mask)
            log_prob.append(log_u)
            indices.append(index)
            # input()
        indices.append(indices[0])
        indices = torch.stack(indices).T
        log_prob = torch.log(torch.stack(log_prob)).sum(dim=0)
        
        return indices, log_prob
    
    
    def forward(self, x):
        enc, h, c, embed = self.encoder(x)
        indices, log_prob = self.decoder(enc, h, embed)
        self.log_prob = log_prob
        self.indices = indices
        
        return indices


class PN_Critic(torch.nn.Module):
    
    def __init__(self, config, dtype=torch.float32):
        super(PN_Critic, self).__init__()
        
        self.config = config
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' in config.device else "cpu")
        
        self.build_model()
        
    def build_model(self,):
        
        init = torch.nn.init.xavier_uniform_
        
        # encoder structure
        self.embedder1 = torch.nn.Conv1d(self.config.dimension, self.config.hidden1, 1).to(device=self.device)
        
        self.lstm_enc = torch.nn.LSTMCell(self.config.hidden1, self.config.hidden2).to(device=self.device)
        self.lstm_enc_c0 = torch.nn.parameter.Parameter(init(torch.empty((1, self.config.hidden2), dtype=self.dtype, device=self.device)).repeat(self.config.batch_size, 1))
        self.lstm_enc_h0 = torch.nn.parameter.Parameter(init(torch.empty((1, self.config.hidden2), dtype=self.dtype, device=self.device)).repeat(self.config.batch_size, 1))
        
        # decoder structure
        self.fcn1 = torch.nn.Linear(self.config.hidden2, self.config.hidden2).to(self.device)
        self.nl1 = torch.nn.ReLU()
        self.fcn2 = torch.nn.Linear(self.config.hidden2, 1).to(self.device)
        
    def encoder(self, x):
        embed = self.embedder1(x.transpose(1, 2)).transpose(1, 2)
        # print('embed.shape:', embed.shape)
        enc = []
        h = self.lstm_enc_h0
        c = self.lstm_enc_c0
        for i in range(x.shape[1]):
            h, c = self.lstm_enc(embed[:, i, :], (h, c))
            enc.append(h)
        enc = torch.stack(enc).transpose(0, 1)
        # print('enc.shape:', enc.shape)
        return enc
    
    def decoder(self, enc):
        dec = self.fcn1(enc.sum(dim=1))
        dec = self.nl1(dec)
        v = self.fcn2(dec).squeeze()
        return v
    
    def forward(self, x):
        enc = self.encoder(x)
        v = self.decoder(enc)
        
        return v


