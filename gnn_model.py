import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim//2, output_dim, bias=True)
        )

    def forward(self, x):
        return self.mlp(x)

class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Aggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class UserModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rating_emb):
        super(UserModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rating_emb = rating_emb

        self.g_v = MLP(2*self.emb_dim, self.emb_dim)
        
        self.user_item_attn = MLP(2*self.emb_dim, 1)
        self.aggr_items = Aggregator(self.emb_dim, self.emb_dim)

        self.user_user_attn = MLP(2*self.emb_dim, 1)
        self.aggr_neighbors = Aggregator(self.emb_dim, self.emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10

    def forward(self, uids, u_item_pad, u_user_pad, u_user_item_pad):

        q_a = self.item_emb(u_item_pad[:,:,0])
        u_item_er = self.rating_emb(u_item_pad[:,:,1])
        x_ia = self.g_v(torch.cat([q_a, u_item_er], dim=2).view(-1, 2*self.emb_dim)).view(q_a.size())
        mask_u = torch.where(u_item_pad[:,:,0]>0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        p_i = mask_u.unsqueeze(2).expand_as(x_ia) * self.user_emb(uids).unsqueeze(1).expand_as(x_ia)
        alpha = self.user_item_attn(torch.cat([x_ia, p_i], dim=2).view(-1, 2*self.emb_dim)).view(mask_u.size())
        alpha = torch.exp(alpha)*mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)
        h_iI = self.aggr_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * x_ia, 1))


        q_a_s = self.item_emb(u_user_item_pad[:,:,:,0])
        u_user_item_er = self.rating_emb(u_user_item_pad[:,:,:,1])
        x_ia_s = self.g_v(torch.cat([q_a_s, u_user_item_er], dim=2).view(-1, 2*self.emb_dim)).view(q_a_s.size())
        mask_s = torch.where(u_user_item_pad[:,:,:,0]>0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        p_i_s = mask_s.unsqueeze(3).expand_as(x_ia_s) * self.user_emb(u_user_pad).unsqueeze(2).expand_as(x_ia_s)
        alpha_s = self.user_item_attn(torch.cat([x_ia_s, p_i_s], dim=3).view(-1, 2*self.emb_dim)).view(mask_s.size())
        alpha_s = torch.exp(alpha_s)*mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)
        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * x_ia_s, 2)
        h_oI = self.aggr_items(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())
        
        beta = self.user_user_attn(torch.cat([h_oI, self.user_emb(u_user_pad)], dim = 2).view(-1, 2 * self.emb_dim)).view(u_user_pad.size())
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iS = self.aggr_neighbors(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))

        h_i = self.mlp(torch.cat([h_iI, h_iS], dim = 1))

        return h_i


class ItemModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rating_emb):
        super(ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rating_emb = rating_emb

        self.g_u = MLP(2*self.emb_dim, self.emb_dim)

        self.item_users_attn = MLP(2*self.emb_dim, 1)
        self.aggr_users = Aggregator(self.emb_dim, self.emb_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10
    
    def forward(self, iids, i_user_pad):

        p_t = self.user_emb(i_user_pad[:,:,0])
        i_user_er = self.rating_emb(i_user_pad[:,:,1])
        mask_i = torch.where(i_user_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        f_jt = self.g_u(torch.cat([p_t, i_user_er], dim = 2).view(-1, 2 * self.emb_dim)).view(p_t.size())
        q_j = mask_i.unsqueeze(2).expand_as(f_jt) * self.item_emb(iids).unsqueeze(1).expand_as(f_jt)
        mu_jt = self.item_users_attn(torch.cat([f_jt, q_j], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_i.size())
        mu_jt = torch.exp(mu_jt) * mask_i
        mu_jt = mu_jt / (torch.sum(mu_jt, 1).unsqueeze(1).expand_as(mu_jt) + self.eps)
        
        z_j = self.aggr_users(torch.sum(mu_jt.unsqueeze(2).expand_as(f_jt) * f_jt, 1))

        return z_j
        
    
class GraphRec(nn.Module):
    def __init__(self, n_users, n_items, n_ratings, emb_dim = 64):
        super(GraphRec, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.emb_dim = emb_dim

        self.user_emb = nn.Embedding(self.n_users, self.emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim, padding_idx=0)
        self.rating_emb = nn.Embedding(self.n_ratings, self.emb_dim, padding_idx=0)

        self.user_model = UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rating_emb)
        self.item_model = ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rating_emb)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1)
        )

    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad):

        h_i = self.user_model(uids, u_item_pad, u_user_pad, u_user_item_pad)
        z_j = self.item_model(iids, i_user_pad)

        r_ij = self.mlp(torch.cat([h_i, z_j], dim=1))

        return r_ij