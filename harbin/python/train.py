
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from data_utils import DataLoader
from model import *
import argparse, os, time
# import db_utils
import data_utils
from collections import namedtuple
from args import *
import pickle

##  python ~/travel-model/travel-modelling/harbin/python/train.py -trainpath /home/xiucheng/travel-model/data-7min/traindata -validpath /home/xiucheng/travel-model/data-7min/validdata -kl_decay 0.0 -use_selu -random_emit

args = make_args()
print(args)

device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(12)
##############################################################################

def log_pdf(logμ, logλ, t):
    """
    log pdf of IG distribution.
    Input:
      logμ, logλ (batch_size, ): the parameters of IG
      t (batch_size, ): the travel time
    ------
    Output:
      logpdf (batch_size, )
    """
    eps = 1e-9
    μ = torch.exp(logμ)
    expt = -0.5 * torch.exp(logλ)*torch.pow(t-μ,2) / (μ.pow(2)*t+eps)
    logz = 0.5*logλ - 1.5*torch.log(t)
    return expt+logz

def log_prob_loss(logμ, logλ, t):
    """
    logμ, logλ (batch_size, ): the parameters of IG
    t (batch_size, ): the travel time
    ---
    Return the average loss of the log probability of a batch of data
    """
    logpdf = log_pdf(logμ, logλ, t)
    return torch.mean(-logpdf)

def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

def adjust_lr(optimizer, epoch):
    lr = args.lr * (args.lr_decay ** (epoch//3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def log_Piplus1(pdist, true_index):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pdist, torch.tensor([true_index]))
    return loss

def loss_X(x, M, logS):
    return 0.5 * torch.sum(logS + torch.pow((x-M),2)/(torch.exp(logS)))

def loss_gumbel(y, dim_pi):
    log_ratio = torch.log(y * dim_pi + 1e-20)
    KLD = torch.sum(y * log_ratio, dim=-1).mean()
    return(KLD)

#############################################################################

## Preparing the data
trainfiles = list(filter(lambda x:x.endswith(".h5"),
                         sorted(os.listdir(args.trainpath))))
validfiles = list(filter(lambda x:x.endswith(".h5"),
                         sorted(os.listdir(args.validpath))))
train_dataloader = DataLoader(args.trainpath)
print("Loading the training data...")
train_dataloader.read_files(trainfiles)
# valid_dataloader = DataLoader(args.validpath)
# print("Loading the validation data...")
# valid_dataloader.read_files(validfiles)
train_slot_size = np.array(list(map(lambda s:s.ntrips, train_dataloader.slotdata_pool)))
train_num_iterations = int(np.ceil(train_slot_size/args.batch_size).sum())
print("There are {} trips in the training dataset".format(train_slot_size.sum()))
print("Number of iterations for an epoch: {}".format(train_num_iterations))
# valid_slot_size = np.array(list(map(lambda s:s.ntrips, valid_dataloader.slotdata_pool)))
# valid_num_iterations = int(np.ceil(valid_slot_size/args.batch_size).sum())

MAX_NBRS = -1
NUM_ROADS = max(data_utils.unique_roads)+1
MASK_MATRIX = torch.zeros((NUM_ROADS,NUM_ROADS),dtype=torch.long)

for road in data_utils.dict_index_of_road_segment_in_neighborhood.keys():
    road_nbrs = list(data_utils.dict_index_of_road_segment_in_neighborhood[road].keys())

    if(len(road_nbrs) >MAX_NBRS):
        MAX_NBRS = len(road_nbrs)

    for road_nbr in road_nbrs:
        MASK_MATRIX[road][road_nbr] =1

print("max neighbours {}".format(MAX_NBRS))

dict_index_of_road_segment_in_neighborhood = data_utils.dict_index_of_road_segment_in_neighborhood
# breakpoint()
# check if dict_index_of_road_segment_in_neighborhood has road id d0 with non zero nbrs

#############################################################################

## Parameters
# dim_s1, dim_s2, dim_s3 = 64, 32, 16
hidden_size = 256
hidden_size1, hidden_size2, hidden_size3 = 128, 500, 600
temp_min = 0.5
ANNEAL_RATE = 0.00003

Params = namedtuple("params", ["dim_c", "dim_pi", "hidden_size3", "hidden_size2", "hidden_size1", 
                                "hidden_size", "dropout", "use_selu"])
params = Params(dim_c=args.dim_c, dim_pi=args.dim_pi,
                hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3,
                hidden_size = hidden_size, dropout=args.dropout, use_selu=args.use_selu)

## Model
probtraffic = ProbTraffic(1, hidden_size2, args.dim_c,
                          args.dropout, args.use_selu).to(device)

probiplus1 = Probr_i_plus_1(args.dim_pi, args.dim_h, args.dim_c,  args.dim_n_r, args.dim_n_x,
                       hidden_size1, args.dropout, args.use_selu, MAX_NBRS, NUM_ROADS,MASK_MATRIX)

probPi = ProbPi(hidden_size, args.dim_pi, args.dropout, args.use_selu).to(device)  # CHECK - what should be the hidden size
probX = ProbX(args.dim_pi, hidden_size, args.dropout, args.use_selu).to(device)

## Optimiser
optimizer_pi = torch.optim.Adam(probPi.parameters(), lr=args.lr, amsgrad=True)
optimizer_X = torch.optim.Adam(probX.parameters(), lr=args.lr, amsgrad=True)
optimizer_traffic = torch.optim.Adam(probtraffic.parameters(), lr=args.lr, amsgrad=True)
optimizer_iplus1 = torch.optim.Adam(probiplus1.parameters(), lr=args.lr, amsgrad=True)

def validate(num_iterations):
    probrho.eval()
    probtraffic.eval()
    probttime.eval()

    with torch.no_grad():
        total_loss, total_mse = 0.0, 0.0
        for _ in range(num_iterations):
            data = valid_dataloader.order_emit(args.batch_size)
            road_lens = probrho.roads_length(data.trips, data.ratios)
            l = road_lens.sum(dim=1) # the whole trip lengths
            w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
            rho = probrho(data.trips)
            c, mu_c, logvar_c = probtraffic(data.S.to(device))
            logμ, logλ = probttime(rho, c, w, l)
            times = data.times.to(device)
            loss = log_prob_loss(logμ, logλ, times) + args.kl_decay*KLD(mu_c, logvar_c)
            total_loss += loss.item() * data.trips.shape[0]
            total_mse += F.mse_loss(torch.exp(logμ), times).item() * data.trips.shape[0]
        mean_loss, mean_mse = total_loss/np.sum(valid_slot_size), total_mse/np.sum(valid_slot_size)
        print("Validation Loss {0:.4f} MSE {1:.4f}".format(mean_loss, mean_mse))
    probrho.train()
    probtraffic.train()
    probttime.train()
    return mean_loss, mean_mse

tau = 0.2
loss_threshold = -200
# τ is a temperature parameter that allows us to control how closely samples from the Gumbel-Softmax distribution approximate those from the categorical distribution. As τ→0, the softmax becomes an argmax and the Gumbel-Softmax distribution becomes the categorical distribution. During training, we let τ>0 to allow gradients past the sample, then gradually anneal the temperature τ (but not completely to 0, as the gradients would blow up).

def model_analysis(num_iterations = 1000):
    pi_to_x = [set() for _ in range(1000)]
    next_segment_prediction = []
    for it in range(1, num_iterations+1):
        data = train_dataloader.random_emit(args.batch_size)
        c, mu_c, logvar_c = probtraffic(data.S.to(device))
        total_next_segments = 0
        matched_next_segments = 0
        for trip_id in range(args.batch_size):
            if(trip_id>=len(data.destinations)):
                break

            ## pi prediction
            dest = data.destinations[trip_id]
            pi, y = probPi(dest, tau, args.hard)
            cluster_id = torch.argmax(pi)
            pi_to_x[cluster_id].add(dest)

            ## next road segment prediction
            trip = data.trips[trip_id]
            trip_seq = trip.tolist()
            hidden_state = probiplus1.initHidden(hidden_size1)
            for i in range(0, len(trip_seq)-1):
                road_segment_id = trip_seq[i]
                correct_nbr_id = trip_seq[i+1]
                if(correct_nbr_id==0):
                    break
                prob_dist_i_plus1, hidden_state = probiplus1(c, pi, road_segment_id, hidden_state)
                predicted_nbr_id = torch.argmax(prob_dist_i_plus1) + 1      ### CHECK - indexing

                total_next_segments += 1
                if (predicted_nbr_id == correct_nbr_id):
                    matched_next_segments += 1
                next_segment_prediction.append((correct_nbr_id, predicted_nbr_id))

    pifile = open('pi2x', 'wb')
    pickle.dump(pi_to_x, pifile)
    pifile.close()

    print("Accuracy of next road segment prediction is {}".format((1.0*matched_next_segments)/total_next_segments))
    segfile = open('next_segment_predictions', 'wb')
    pickle.dump(next_segment_prediction, segfile)
    segfile.close()
    breakpoint()

def my_train(num_iterations):
    epoch_loss, epoch_mse, stage_mse = 0., 0., 0.
    for it in range(1, num_iterations+1):
        ## Loading the data
        if args.random_emit == True:
            data = train_dataloader.random_emit(args.batch_size)
        else:
            data = train_dataloader.order_emit(args.batch_size)
        ## forward computation
        loss = 0
        c, mu_c, logvar_c = probtraffic(data.S.to(device))
        loss_kld = KLD(mu_c, logvar_c)

        for trip_id in range(args.batch_size):

            if(trip_id>=len(data.destinations)):
                break

            dest = data.destinations[trip_id]
            
            pi, y = probPi(dest, tau, args.hard)
            loss_g = loss_gumbel(y, args.dim_pi)

            reconstructed_dest, mu, logvar = probX(pi)
            loss_x = loss_X(dest, mu, logvar)
            loss += (loss_kld + loss_g + loss_x)
            ###
            # trip = data.trips[trip_id]
            # trip_seq = trip.tolist()
            # hidden_state = probiplus1.initHidden(hidden_size1)
            # loss_trips = 0
            # for i in range(0, len(trip_seq)-1):
            #     road_segment_id = trip_seq[i]
            #     prob_dist_i_plus1, hidden_state = probiplus1(c, pi, road_segment_id, hidden_state)
            #     # prob_dist_i_plus1 = probiplus1(c, pi, road_segment_id)
            #     correct_nbr_id = trip_seq[i+1]
            #     if(correct_nbr_id==0):
            #         break
            #     loss_trip = log_Piplus1(prob_dist_i_plus1, correct_nbr_id)
            #     loss_trips += loss_trip
            ## print("loss c {}, gumbel {}, X {}, trip {}".format(loss_kld, loss_g, loss_x, loss_trips))
            # loss += loss_trips

        # breakpoint()
        ## backward optimisation
        optimizer_pi.zero_grad()
        optimizer_X.zero_grad()
        optimizer_traffic.zero_grad()
        # optimizer_iplus1.zero_grad()
        loss.backward()

        ## optimising
        clip_grad_norm_(probPi.parameters(), args.max_grad_norm)
        clip_grad_norm_(probX.parameters(), args.max_grad_norm)
        # clip_grad_norm_(probiplus1.parameters(), args.max_grad_norm)
        clip_grad_norm_(probtraffic.parameters(), args.max_grad_norm)
        optimizer_pi.step()
        optimizer_X.step()
        optimizer_traffic.step()
        # optimizer_iplus1.step()
        print("Loss: {}".format(loss))

def train(num_iterations=1000):
    epoch_loss, epoch_mse, stage_mse = 0., 0., 0.
    for it in range(1, num_iterations+1):
        ## Loading the data
        if args.random_emit == True:
            data = train_dataloader.random_emit(args.batch_size)
        else:
            data = train_dataloader.order_emit(args.batch_size)
        ## forward computation
        road_lens = probrho.roads_length(data.trips, data.ratios)
        l = road_lens.sum(dim=1) # the whole trip lengths
        w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
        rho = probrho(data.trips)
        c, mu_c, logvar_c = probtraffic(data.S.to(device))
        logμ, logλ = probttime(rho, c, w, l)
        ## move to gpu
        times = data.times.to(device)
        loss = log_prob_loss(logμ, logλ, times) + args.kl_decay*KLD(mu_c, logvar_c)
        epoch_loss += loss.item()
        ## Measuring the mean square error
        mse = F.mse_loss(torch.exp(logμ), times)
        epoch_mse += mse.item()
        stage_mse += mse.item()
        if it % args.print_freq == 0:
            print("Stage MSE: {0:.4f} at epoch {1:} iteration {2:}".format\
                  (stage_mse/args.print_freq, epoch, it))
            stage_mse = 0
        ## backward optimization
        optimizer_rho.zero_grad()
        optimizer_traffic.zero_grad()
        optimizer_ttime.zero_grad()
        loss.backward()
        ## optimizing
        clip_grad_norm_(probrho.parameters(), args.max_grad_norm)
        clip_grad_norm_(probtraffic.parameters(), args.max_grad_norm)
        clip_grad_norm_(probttime.parameters(), args.max_grad_norm)
        optimizer_rho.step()
        optimizer_traffic.step()
        optimizer_ttime.step()
    print("\nEpoch Loss: {0:.4f}".format(epoch_loss / num_iterations))
    print("Epoch MSE: {0:.4f}".format(epoch_mse / num_iterations))

tic = time.time()
min_mse = 1e9
for epoch in range(1, args.num_epoch+1):
    print("epoch {} =====================================>".format(epoch))
    my_train(train_num_iterations)
    if (epoch%5 == 0):
        model_analysis(train_num_iterations)
    # mean_loss, mean_mse = validate(valid_num_iterations)
    # if mean_mse < min_mse:
    #     print("Saving model...")
    #     torch.save({
    #         "probrho": probrho.state_dict(),
    #         "probtraffic": probtraffic.state_dict(),
    #         "probttime": probttime.state_dict(),
    #         "params": params._asdict()
    #     }, "best-model.pt")
    #     min_mse = mean_mse
    # adjust_lr(optimizer_rho, epoch)
    # adjust_lr(optimizer_traffic, epoch)
    # adjust_lr(optimizer_ttime, epoch)
breakpoint()
print("Saving model...")

state = {
    "epoch": epoch,
    "probtraffic": probtraffic.state_dict(),
    "probiplus1": probiplus1.state_dict(),
    "probPi": probPi.state_dict(),
    "probX": probX.state_dict(),
    "params": params._asdict(),
    "optimizer_pi": optimizer_pi.state_dict(),
    "optimizer_X": optimizer_X.state_dict(),
    "optimizer_traffic": optimizer_traffic.state_dict(),
    "optimizer_iplus1": optimizer_iplus1.state_dict()
}
torch.save(state, "model2.pt")
cost = time.time() - tic
print("Time passed: {} hours".format(cost/3600))
