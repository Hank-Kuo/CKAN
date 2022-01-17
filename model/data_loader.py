import torch
from torch.utils import data as torch_data

class Dataset(torch_data.Dataset):
    def __init__(self, params, data, user_triple_set, item_triple_set):
        self.params = params
        self.data = data
        self.user_triple_set = user_triple_set
        self.item_triple_set = item_triple_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.data[index, 0]
        items = self.data[index, 1]
        labels = self.data[index, 2]
        
        user_h, user_r, user_t = [], [], []
        item_h, item_r, item_t = [], [], []
        
        for i in range(self.params.n_hop):
            user_h.append(torch.LongTensor(self.user_triple_set[user][i][0]).to(self.params.device))
            user_r.append(torch.LongTensor(self.user_triple_set[user][i][1]).to(self.params.device))
            user_t.append(torch.LongTensor(self.user_triple_set[user][i][2]).to(self.params.device))

            item_h.append(torch.LongTensor(self.item_triple_set[items][i][0]).to(self.params.device))
            item_r.append(torch.LongTensor(self.item_triple_set[items][i][1]).to(self.params.device))
            item_t.append(torch.LongTensor(self.item_triple_set[items][i][2]).to(self.params.device))

        return user, items, labels, [user_h, user_r, user_t], [item_h, item_r, item_t]
