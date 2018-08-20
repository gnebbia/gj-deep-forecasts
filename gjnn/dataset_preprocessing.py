import pandas as pd
import torch
import gjnn.dataloader
from sklearn import preprocessing
import timeit


## This function also returns the min and max of the original data for all columns
## that are normalized. This will give us a normalization scale when
## Each observed value #  problem: We need to normalize the tournament ifp's
## using the normalization scale
## from the training data.

def prepare_data(dataset):
    # To modify when dataset column order will change

    dataset = pd.read_csv("ds_sample_5M.csv", sep=None, engine='python',
                          dtype={'user_id_1': "category", "user_id_2": "category"})
    dataset.drop(["ifp_id"], axis=1, inplace=True)
    dataset = dataset.apply(pd.to_numeric)

    ## categorical columns
    dataset["user_id_1"] = dataset["user_id_1"].fillna(0.0).astype(int)
    dataset["user_id_2"] = dataset["user_id_2"].fillna(0.0).astype(int)
    dataset["expertise_1"] = dataset["expertise_1"].fillna(0.0).astype(int)
    dataset["expertise_2"] = dataset["expertise_2"].fillna(0.0).astype(int)

    min_id = min(min(dataset["user_id_1"]), min(dataset["user_id_2"]))
    max_id = max(max(dataset["user_id_1"]), max(dataset["user_id_2"]))
    min_expertise = min(min(dataset["expertise_1"]), min(dataset["expertise_2"]))
    max_expertise = max(max(dataset["expertise_2"]), max(dataset["expertise_2"]))
    min_value_a = min(min(dataset["value_a_1"]), min(dataset["value_a_2"]))
    min_value_b = min(min(dataset["value_b_1"]), min(dataset["value_b_2"]))
    min_value_c = min(min(dataset["value_c_1"]), min(dataset["value_c_2"]))
    max_value_a = max(max(dataset["value_a_1"]), max(dataset["value_a_2"]))
    max_value_b = max(max(dataset["value_b_1"]), max(dataset["value_b_2"]))
    max_value_c = max(max(dataset["value_c_1"]), max(dataset["value_c_2"]))
    min_days_from_start = min(min(dataset["days_from_start_1"]), min(dataset["days_from_start_2"]))
    max_days_from_start = max(max(dataset["days_from_start_1"]), max(dataset["days_from_start_2"]))

    for i in dataset.columns:
        norm_x = []
        if i.startswith("distance") or i.startswith("topic"):
            # Don't normalize the distance or topic columns!
            continue
        if i == "user_id_1":
            dataset["user_id_1"] = dataset["user_id_1"].transform(
                lambda x: (float(x) - min_id) / (max_id - min_id))
        elif i == "user_id_2":
            dataset["user_id_2"] = dataset["user_id_2"].transform(
                lambda x: (float(x) - min_id) / (max_id - min_id))
        elif i == "expertise_1":
            dataset["expertise_1"] = dataset["expertise_1"].transform(
                lambda x: (float(x) - min_expertise) / (max_expertise - min_expertise))
        elif i == "expertise_2":
            dataset["expertise_2"] = dataset["expertise_2"].transform(
                lambda x: (float(x) - min_expertise) / (max_expertise - min_expertise))
        elif i.startswith("value_a"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - min_value_a) / (max_value_a - min_value_a))
        elif i.startswith("value_b"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - min_value_b) / (max_value_b - min_value_b))
        elif i.startswith("value_c"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - min_value_c) / (max_value_c - min_value_c))
        elif i.startswith("days_from_start"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - min_days_from_start) / (max_days_from_start - min_days_from_start))

    return {"id":[min_id,max_id], "expertise":[min_expertise, max_expertise],
            "value_a": [min_value_a, max_value_a], "value_b": [min_value_b, max_value_b],
            "value_c": [min_value_c, max_value_c],
            "days_from_start": [min_days_from_start, max_days_from_start]}, dataset

def prepare_out_of_sample_data(minmax, dataset):

    dataset.drop(["ifp_id"], axis=1, inplace=True)
    dataset = dataset.apply(pd.to_numeric)

    for i in dataset.columns:
        if i.startswith("distance") or i.startswith("topic"):
            # Don't normalize the distance or topic columns!
            continue
        if i == "user_id_1":
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["id"][0]) / (minmax["id"][1] - minmax["id"][0]))


        elif i == "user_id_2":
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["id"][0]) / (minmax["id"][1] - minmax["id"][0]))
        elif i == "expertise_1":
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["expertise"][0]) / (minmax["expertise"][1] - minmax["expertise"][0]))
        elif i == "expertise_2":
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["expertise"][0]) / (minmax["expertise"][1] - minmax["expertise"][0]))
        elif i.startswith("value_a"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["value_a"][0]) / (minmax["value_a"][1] - minmax["value_a"][0]))
        elif i.startswith("value_b"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["value_b"][0]) / (minmax["value_b"][1] - minmax["value_b"][0]))
        elif i.startswith("value_c"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["value_c"][0]) / (minmax["value_c"][1] - minmax["value_c"][0]))
        elif i.startswith("days_from_start"):
            dataset[i] = dataset[i].transform(
                lambda x: (float(x) - minmax["days_from_start"][0]) / (minmax["days_from_start"][1] - minmax["days_from_start"][0]))

    return dataset

def get_sample_question_data_loaders():
    data_loaders = []
    for j in range(1050, 1052):  # 1105):
        if j == 1079:
            continue
        filename = "ds_" + str(j) + "-0.csv"
        d = pd.read_csv(filename, sep=None, engine='python',
                        dtype={'user_id_1': "category", "user_id_2": "category"})
        d.drop(["ifp_id"], axis=1, inplace=True)
        d = d.apply(pd.to_numeric)
        d = prepare_data(d)
        test = gjnn.dataloader.Dataset(d)
        data_loaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=False))
    return data_loaders




def prepare_data_old(dataset):
    # To modify when dataset column order will change
    features_user_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]
    features_user_2 = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22]
    topics = [0, 1, 2, 3, 4, 5]

    ## categorical columns
    dataset["user_id_1"] = dataset["user_id_1"].fillna(0.0).astype(int)
    dataset["user_id_2"] = dataset["user_id_2"].fillna(0.0).astype(int)
    dataset["expertise_1"] = dataset["expertise_1"].fillna(0.0).astype(int)
    dataset["expertise_2"] = dataset["expertise_2"].fillna(0.0).astype(int)
    min_id = min(min(dataset["user_id_1"]), min(dataset["user_id_2"]))
    max_id = max(max(dataset["user_id_1"]), max(dataset["user_id_2"]))
    min_expertise = min(min(dataset["expertise_1"]), min(dataset["expertise_2"]))
    max_expertise = max(max(dataset["expertise_2"]), max(dataset["expertise_2"]))

    ## Normalize columns
    min_max_scaler = preprocessing.MinMaxScaler()
    for i in dataset.columns:
        if i.startswith("distance"):
            # Don't normalize the distance columns!
            continue
        if i == "user_id_1":
            dataset["user_id_1"] = dataset["user_id_1"].transform(lambda x: (float(x) - min_id) / (max_id - min_id))
        elif i == "user_id_2":
            dataset["user_id_2"] = dataset["user_id_2"].transform(lambda x: (float(x) - min_id) / (max_id - min_id))
        elif i == "expertise_1":
            dataset["expertise_1"] = dataset["expertise_1"].transform(
                lambda x: (float(x) - min_expertise) / (max_expertise - min_expertise))
        elif i == "expertise_2":
            dataset["expertise_2"] = dataset["expertise_2"].transform(
                lambda x: (float(x) - min_expertise) / (max_expertise - min_expertise))
        else:
            x = dataset[i]  # returns a numpy array
            x_scaled = min_max_scaler.fit_transform(x)
            dataset[i] = x_scaled

    #user_1 = dataset.iloc[:, features_user_1]
    #user_2 = dataset.iloc[:, features_user_2]

    return dataset
