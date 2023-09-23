import torch.nn as nn
import torch
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):

    def __init__(self, embed_dim, q_dim):
        super(SingleHeadAttention, self).__init__()
        self.Wq = nn.Linear(embed_dim, q_dim)
        self.Wk = nn.Linear(embed_dim, q_dim)    # when using dot product for attention q_dim == k_dim
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, src_seq, target_seq):
        """
            - src_seq    - (N, S, E)  -> E = embed_dim, S = Src Sequence length, N = batch_size 
            - target_seq - (N, L, E)  -> E = embed_dim, L = Target Sequence length, N = batch_size
        """

        # Project the source sequence into key-space:
        keys = self.Wk(src_seq)              # (N, S, q_dim)

        # Project the target sequence into the query-space:
        queries = self.Wq(target_seq)        # (N, L, q_dim)

        # Project the source sequence into value-space:
        values = self.Wv(src_seq)            # (N, S, embed_dim)

        # Compute dot-product attention between all queries and all keys:
        attention = torch.bmm(queries, keys.permute(0, 2, 1))    # (N, L, S)

        # Normalize the attention scores along the S dimension:
        normalized_attention = F.softmax(attention, dim=-1)

        # Sum up the values based on the attention weights to get the output
        output = torch.bmm(normalized_attention, values)    # (N, L, embed_dim)

        return output
