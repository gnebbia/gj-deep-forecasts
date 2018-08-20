import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gjnn.model
import gjnn.dataloader
import gjnn.dataset_preprocessing
import timeit
import operator
import os.path
import csv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Get a ranking from the win/losses in the tournament mtx.
## We implement INCR-INDEG as a 5-approximation for the best ranking
## ref: "Ordering by weighted number of wins gives a good ranking for weighted tournaments"
## authors: coppersmith, fleischer, rurda
def make_ranking(tournament_matrix):
    x = tournament_matrix
    for i in xrange(0,tournament_matrix.shape[0]):
        tournament_matrix[i, i] = 0  # Just in case there is bogus data

    y = np.zeros(tournament_matrix.shape)
    for i in xrange(0, tournament_matrix.shape[0]):
        for j in xrange(0, tournament_matrix.shape[0]):
            y[i, j] = tournament_matrix[i, j] / \
                    (tournament_matrix[i, j]+tournament_matrix[j, i]) if tournament_matrix[i, j] > 0 \
                    else 0.0

    scores = {}
    for i in xrange(0, tournament_matrix.shape[0]):
        scores[i] = np.sum(y[i, ])
    return sorted(scores.items(), key=operator.itemgetter(1), reverse=True)


def get_sample_question_data_loaders():
    data_loaders = []
    minmax_values = np.load("gjnn/min_max_dataset_ds_sample_5M.csv.npy").item()
    for j in range(1050, 1105):
        print("Building DataLoader for IFIP_{}...".format(j))
        if j == 1079:
            continue
        filename = "ds_" + str(j) + "-0.csv"
        d = pd.read_csv(filename, sep=None, engine='python',
                              dtype={'user_id_1': "category", "user_id_2": "category"})
        d = gjnn.dataset_preprocessing.prepare_out_of_sample_data(minmax_values,d)
        test = gjnn.dataloader.Dataset(d)
        data_loaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=False))
    return data_loaders

# Dataset Loading
print("Building data loaders for all out of training IFIPs...")
x = timeit.time.time()
data_loaders = get_sample_question_data_loaders()
print("Done! It took {:.3f} seconds".format(timeit.time.time() - x))

# Load model from disk
hidden_layer_size = 64
siamese_layer_size = 32
output_layer_size = 1
num_features_per_branch = 13
model = gjnn.model.SiameseNetwork(num_features_per_branch, siamese_layer_size, hidden_layer_size, output_layer_size)
model.load_state_dict(torch.load("use_model_epoch_1.pt"))
model = model.to(device)
criterion = nn.modules.loss.BCEWithLogitsLoss()

model.eval()

minmax = np.load("gjnn/min_max_dataset_ds_sample_5M.csv.npy").item()
print("Running Tournaments for out-of-sample IFIPs...")
IFIP_id_list = range(1050, 1105)
for i, loader in enumerate(data_loaders):
    if IFIP_id_list[i] == 1079:
        continue ## This IFIP is missing.
    k = 0
    competitors = {}
    print("Finding competitors on IFIP_{}...".format(IFIP_id_list[i]))
    for o, (oot_user_1, oot_user_2, _, _) in enumerate(loader):
        user_id1 = str("%.8f" % oot_user_1[0][7].item())
        user_id2 = str("%.8f" % oot_user_2[0][7].item())
        if user_id1 not in competitors.keys():
            competitors[user_id1] = k
            k += 1
        if user_id2 not in competitors.keys():
            competitors[user_id2] = k
            k += 1
        # Safe assumption: all forecasters will be seen in the first 100k entries of the data file.
        # This is because entries in file are ordered by id1, id2. So we will see id1 vs all other
        # forecasters in the first entries of the data file.
        if o == 100000:
            break

    print(k)
    tournament_matrix = np.zeros([len(competitors.keys()), len(competitors.keys())])
    if not os.path.isfile("last_tmatrix_{}.npy".format(IFIP_id_list[i])):
        for o, (oot_user_1, oot_user_2, oot_user_1_dist, oot_user_2_dist) in enumerate(loader):
            player_1 = competitors[str("%.8f" % oot_user_1[0][7].item())]
            player_2 = competitors[str("%.8f" % oot_user_2[0][7].item())]

            oot_user_1 = oot_user_1.to(device)
            oot_user_2 = oot_user_2.to(device)
            oot_user_1_dist = oot_user_1_dist.to(device)
            oot_user_2_dist = oot_user_2_dist.to(device)

            ## output == 1: player_1 is closer than player_2
            ## output == 0: player_2 is closer than player_1
            output = (model(oot_user_1, oot_user_2).squeeze() > 0)
            truth = (oot_user_2_dist - oot_user_1_dist > 0).type_as(output)

            if output == 1:
                tournament_matrix[player_1, player_2] += 1
            else:
                tournament_matrix[player_2, player_1] += 1

            if o % 10000 == 0:
                print("Still working... {}/{}".format(o, len(loader)))

        np.save("last_tmatrix_{}.npy".format(IFIP_id_list[i]), tournament_matrix)
    else:
        tournament_matrix = np.load("last_tmatrix_{}.npy".format(IFIP_id_list[i]))

    ranking = make_ranking(tournament_matrix)
    # The competitor map is invertible.
    inverse_competitors = {v: k for k, v in competitors.iteritems()}
    ranking_ids = range(0,len(ranking))
    for k in xrange(0,len(ranking)):
        this_rank = ranking[k]

        # Undo the normalized user_id.
        user_id_norm = float(inverse_competitors[this_rank[0]])
        user_id = user_id_norm * (minmax["id"][1] - minmax["id"][0]) + minmax["id"][0]

        # Round the undone-normalized user_id.
        a = np.ceil(user_id)
        b = np.floor(user_id)
        int_user_id = int(np.ceil(user_id)) if np.abs(user_id-a) < np.abs(user_id-b) else int(np.floor(user_id))
        ranking_ids[k] = (user_id,
                          int_user_id,siamese_layer_size
                          this_rank[1])

    with open('ranking_IFIP_{}.csv'.format(IFIP_id_list[i]), 'wb') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['un_norm_id', 'est_id', 'rank_score'])
        for row in ranking_ids:
            csv_out.writerow(row)


