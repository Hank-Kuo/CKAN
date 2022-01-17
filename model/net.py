import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CKAN(nn.Module):
    def __init__(self, args):
        super(CKAN, self).__init__()
        self._parse_args(args)
        self._init_weight()

    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.criterion = nn.BCELoss()
    
        
        
    def _init_weight(self):
        # init embedding
        self.entity_emb = nn.Embedding(self.n_entity+1, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation+1, self.dim)
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
        )

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, items, labels, user_triple_set, item_triple_set):       
        user_embeddings = []
        
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        # [batch_size, dim]
        user_embeddings.append(user_emb_0.mean(dim=1))
        
        for i in range(self.n_hop):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_i)
            
        item_embeddings = []
        
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)
        
        for i in range(self.n_hop):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)
            
        scores = self.predict(user_embeddings, item_embeddings)
        base_loss = self.criterion(scores, labels.float())
        return_dict = dict(scores=scores, base_loss=base_loss, loss=base_loss)

        return return_dict, []
        
    
    
    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
        
        for i in range(1, len(user_embeddings)):
            e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
        for i in range(1, len(item_embeddings)):
            e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
            
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    
    
    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i



class HOP0_CKAN(nn.Module):
    def __init__(self, args):
        super(HOP0_CKAN, self).__init__()
        self._parse_args(args)
        self._init_weight()

    def _parse_args(self, args):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.criterion = nn.BCELoss()
    
        
        
    def _init_weight(self):
        # init embedding
        self.entity_emb = nn.Embedding(self.n_entity+1, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation+1, self.dim)
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
        )

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, items, labels, user_triple_set, item_triple_set):       
        user_embeddings = []
        
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        # [batch_size, dim]
        user_embeddings.append(user_emb_0.mean(dim=1))
        
     
        item_embeddings = []
        
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)
        
     
            
        scores = self.predict(user_embeddings, item_embeddings)
        base_loss = self.criterion(scores, labels.float())
        return_dict = dict(scores=scores, base_loss=base_loss, loss=base_loss)

        return return_dict, []
        
    
    
    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
        
        for i in range(1, len(user_embeddings)):
            e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
        for i in range(1, len(item_embeddings)):
            e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
            
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    
    
    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i