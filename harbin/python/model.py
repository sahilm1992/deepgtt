
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#use_cuda = False
#device = torch.device("cuda" if use_cuda else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout, use_selu=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.leaky_relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc2(h1)

class MLP2(nn.Module):
    """
    MLP with two output layers
    """
    def __init__(self, input_size, hidden_size, output_size,
                 dropout, use_selu=False):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, output_size)
        self.fc22 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

def logwsumexp(x, w):
    """
    log weighted sum (along dim 1) exp, i.e., log(sum(w * exp(x), 1)).

    Input:
      x (n, m): exponents
      w (n, m): weights
    Output:
      y (n,)
    """
    maxv, _ = torch.max(x, dim=1, keepdim=True)
    y = torch.log(torch.sum(torch.exp(x - maxv) * w, dim=1, keepdim=True)) + maxv
    return y.squeeze(1)

class ProbPi(nn.Module):

    def __init__(self, hidden_size, dim_pi, dropout, use_selu):
        super(ProbPi, self).__init__()
        self.f = MLP(2, hidden_size, dim_pi, dropout, use_selu) 

    def forward(self, x, temp, hard=False):
        """
        Input:
          x (batch_size * 2)
          temp (non-negative scalar temperature)
          hard (true for strictly one-hot)
        Output:
          close to one-hot K-dimensional vector
        """
        y = self.f(x)   # these should be class probabilities
        y = F.softmax(y, dim=-1)
        logy = torch.log(y + 1e-20)     # ye nan ho jaayega if I do not remove the above line
        pi = F.gumbel_softmax(logy, temp, hard)    # should be 1xK
        # CHECK y should be unnormalized log probabilities-- kya mai khud log karu? par negative values ka log is NaN
        return(pi, y)

class ProbX(nn.Module):
    def __init__(self, dim_pi, hidden_size, dropout, use_selu):
        super(ProbX, self).__init__()
        self.f = MLP(dim_pi, hidden_size, dim_pi, dropout, use_selu)
        self.M = nn.Linear(dim_pi, 2)
        self.logS = nn.Linear(dim_pi, 2)       # DOUBT - how do I enforce that this should be non-negative
        ###VVV - I'm assuming S is logvar

    def reparameterize(self, mu, logvar):
        # https://pytorch.org/docs/stable/generated/torch.randn_like.html#torch.randn_like -- don't we want to set requires_grad
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)     # std is only to know the dim of the tensor
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, pi):
        pi_ = self.f(pi)
        mu = self.M(pi_)
        logvar = self.logS(pi_)
        return(self.reparameterize(mu, logvar), mu, logvar)
        # return(self.reparameterize(self.M(pi), self.logS(pi)))
        # cov_matrix = torch.diag(self.S(pi))
        # m = Normal(self.M(pi), cov_matrix)
        # return(m.sample())      # CHECK - reparameterisation not required here
        

# class ProbRho(nn.Module):
#     """
#     s1 (14): road type
#     s2 (7): number of lanes
#     s3 (2): one way or not
#     """
#     def __init__(self, num_u, dim_u, dict_u, lengths,
#                        num_s1, dim_s1, dict_s1,
#                        num_s2, dim_s2, dict_s2,
#                        num_s3, dim_s3, dict_s3,
#                        hidden_size, dim_rho,
#                        dropout, use_selu, device):
#         super(ProbRho, self).__init__()
#         self.lengths = torch.tensor(lengths, dtype=torch.float32, device=device)
#         self.dict_u = dict_u
#         self.dict_s1 = dict_s1
#         self.dict_s2 = dict_s2
#         self.dict_s3 = dict_s3
#         self.embedding_u = nn.Embedding(num_u, dim_u)
#         self.embedding_s1 = nn.Embedding(num_s1, dim_s1)
#         self.embedding_s2 = nn.Embedding(num_s2, dim_s2)
#         self.embedding_s3 = nn.Embedding(num_s3, dim_s3)
#         self.device = device
#         self.f = MLP2(dim_u+dim_s1+dim_s2+dim_s3,
#                       hidden_size, dim_rho, dropout, use_selu)

#     def roads2u(self, roads):
#         """
#         road id to word id (u)
#         """
#         return self.roads_s_i(roads, self.dict_u)

#     def roads_s_i(self, roads, dict_s):
#         """
#         road id to feature id

#         This function should be called in cpu
#         ---
#         Input:
#         roads (batch_size * seq_len): road ids
#         dict_s (dict): the mapping from road id to feature id
#         Output:
#         A tensor like roads
#         """
#         return roads.clone().apply_(lambda k: dict_s[k])

#     def roads_length(self, roads, ratios=None):
#         """
#         roads (batch_size, seq_len): road id to road length
#         ratios (batch_size, seq_len): The ratio of each road segment
#         """
#         if ratios is not None:
#             return self.lengths[roads] * ratios.to(self.device)
#         else:
#             return self.lengths[roads]

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add(mu)
#         else:
#             return mu

#     def forward(self, roads):
#         """
#         roads (batch_size * seq_len)
#         """
#         u  = self.embedding_u(self.roads2u(roads).to(self.device))
#         s1 = self.embedding_s1(self.roads_s_i(roads, self.dict_s1).to(self.device))
#         s2 = self.embedding_s2(self.roads_s_i(roads, self.dict_s2).to(self.device))
#         s3 = self.embedding_s3(self.roads_s_i(roads, self.dict_s3).to(self.device))
#         x  = torch.cat([u, s1, s2, s3], dim=2)
#         mu, logvar = self.f(x)
#         return self.reparameterize(mu, logvar)

class ProbTraffic(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """
    def __init__(self, n_in, hidden_size, dim_c, dropout, use_selu):
        super(ProbTraffic, self).__init__()
        conv_layers = [
            nn.Conv2d(n_in, 32, (5, 5), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(7)
        ]
        self.f1 = nn.Sequential(*conv_layers)       # CNN
        self.f2 = MLP2(128*2*2, hidden_size, dim_c, dropout, use_selu)  # MLPs

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        x = self.f1(T)
        mu, logvar = self.f2(x.view(x.size(0), -1))
        return self.reparameterize(mu, logvar), mu, logvar

class Probr_i_plus_1(nn.Module):
    def __init__(self, dim_pi, dim_h, dim_c,  dim_n_r, dim_n_x,
                       hidden_size, dropout, use_selu, MAX_N_R, NUM_ROADS, MASK_MATRIX):
        super(Probr_i_plus_1, self).__init__()


        self.Wpi = nn.Linear(dim_pi, dim_n_x )

        self.alpha_t = nn.Linear(dim_n_r,NUM_ROADS)
        self.beta_t = nn.Linear(dim_n_x, NUM_ROADS)
        self.gamma_t = nn.Linear(dim_c, NUM_ROADS)


        self.embedding = nn.Embedding(NUM_ROADS, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.f = MLP(NUM_ROADS, hidden_size, NUM_ROADS, dropout, use_selu)



        self.softmax = nn.Softmax(dim=1)
        self.MASK_MATRIX = MASK_MATRIX

    def initHidden(self, hidden_size):
        return torch.zeros(1, 1, hidden_size)

    def own_softmax(self, x, mask_vector):

        means = torch.mean(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x - means)
        mask_x_exp = mask_vector* x_exp
        #x_exp_MASK = x_exp

        x_exp_sum_MASKED = torch.sum(mask_x_exp, 1, keepdim=True)

        return mask_vector / x_exp_sum_MASKED

    def forward(self, c, pi, r_1_i_road_id, hidden):
        """
        rho (batch_size, seq_len, dim_rho)
        c (1, dim_c): the traffic state vector sampling from ProbTraffic
        w (batch_size, seq_len): the normalized road lengths
        l (batch_size, ): route or path lengths
        """
        ## (batch_size, seq_len, dim_rho+dim_c)

        r_1_i_road_id = torch.LongTensor([r_1_i_road_id])
        embedded = self.embedding(r_1_i_road_id).view(1, 1, -1)
        gru_output = embedded
        gru_output, hidden = self.gru(gru_output, hidden)

        Wpi = self.Wpi(pi)
        # Wpi = Wpi.t()

        alphah = self.alpha_t(hidden.view(-1,128))
        betaWpi = self.beta_t(Wpi)
        gammtc = self.gamma_t(c)
        aggregated_input = torch.add(alphah, betaWpi)

        aggregated_input = torch.add(aggregated_input, gammtc)

        road_i_plus_1_prob = self.f(aggregated_input)

        mask_vector = self.MASK_MATRIX[r_1_i_road_id]
        # road_i_plus_1_prob = road_i_plus_1_prob * mask_vector# self.own_softmax(road_i_plus_1_prob, mask_vector)

        zero_indices = (mask_vector==0).nonzero(as_tuple=True)[1]

        road_i_plus_1_prob[0][zero_indices]= -10000000000

        return road_i_plus_1_prob, hidden


# class ProbTravelTime(nn.Module):
#     def __init__(self, dim_rho, dim_c,
#                        hidden_size, dropout, use_selu):
#         super(ProbTravelTime, self).__init__()
#         self.f = MLP2(dim_rho+dim_c, hidden_size, 1, dropout, use_selu)

#     def forward(self, rho, c, w, l):
#         """
#         rho (batch_size, seq_len, dim_rho)
#         c (1, dim_c): the traffic state vector sampling from ProbTraffic
#         w (batch_size, seq_len): the normalized road lengths
#         l (batch_size, ): route or path lengths
#         """
#         ## (batch_size, seq_len, dim_rho+dim_c)
#         x = torch.cat([rho, c.expand(*rho.shape[:-1], -1)], 2)
#         ## (batch_size, seq_len, 1)
#         logm, logv = self.f(x)
#         ## (batch_size, seq_len)
#         logm, logv = logm.squeeze(2), logv.squeeze(2)
#         #m, v = torch.exp(logm), torch.exp(logv)
#         ## (batch_size, )
#         #m_agg, v_agg = torch.sum(m * w, 1), torch.sum(v * w.pow(2), 1)
#         ## parameters of IG distribution
#         ## (batch_size, )
#         #logμ = torch.log(l) - torch.log(m_agg)
#         #logλ = 3*logμ - torch.log(v_agg) - 2*torch.log(l)
#         ## (batch_size, )
#         logm_agg = logwsumexp(logm, w)
#         logv_agg = logwsumexp(logv, w.pow(2))
#         logl = torch.log(l)
#         ## parameters of IG distribution
#         ## (batch_size, )
#         logμ = logl - logm_agg
#         logλ = logl - 3*logm_agg - logv_agg
#         return logμ, logλ

# class TTime(nn.Module):
#     def __init__(self, num_u, dim_u, dict_u,
#                        num_s1, dim_s1, dict_s1,
#                        num_s2, dim_s2, dict_s2,
#                        num_s3, dim_s3, dict_s3,
#                        dim_rho, dim_c, lengths,
#                        hidden_size1, hidden_size2, hidden_size3,
#                        dropout, use_selu, device):
#         super(TTime, self).__init__()
#         self.probrho = ProbRho(num_u, dim_u, dict_u, lengths,
#                                num_s1, dim_s1, dict_s1,
#                                num_s2, dim_s2, dict_s2,
#                                num_s3, dim_s3, dict_s3,
#                                hidden_size1, dim_rho,
#                                dropout, use_selu, device)
#         self.probtraffic = ProbTraffic(1, hidden_size2, dim_c,
#                                        dropout, use_selu)
#         self.probttime = ProbTravelTime(dim_rho, dim_c, hidden_size3,
#                                         dropout, use_selu)

#     def forward(self, roads, ratios,T):
#         road_lens = self.probrho.roads_length(roads, ratios)
#         l = road_lens.sum(dim=1) # the whole trip lengths
#         w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
#         rho = self.probrho(roads)
#         c, mu_c, logvar_c = self.probtraffic(T.to(device))
#         logμ, logλ = self.probttime(rho, c, w, l)
#         return logμ, logλ, mu_c, logvar_c
