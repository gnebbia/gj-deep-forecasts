# coding: utf-8
import os
import pandas as pd
import numpy as np
import argparse
import itertools
import gjnn.dsutils as utils
import subprocess
from tqdm import *
from scipy.spatial import distance


def generate_pairs_for_ifip(infile, outfile):
    print("Loading " + infile + "...")
    ds = pd.read_csv(infile, sep=None, engine='python', dtype={'user_id': "category"})

    tuples =[]
    for subset in itertools.combinations(ds.values.tolist(), 2):
        tuples += subset

    df = pd.DataFrame(data=tuples, columns=ds.columns)

    df_even = df.iloc[::2]
    df_odd = df.iloc[1::2]

    df_even = df_even.rename(columns={'user_id': 'user_id_1', 'value_a': 'value_a_1', 'value_b': 'value_b_1',
                                      'value_c': 'value_c_1', 'days_from_start': 'days_from_start_1',
                                      'distance': 'distance_1', 'expertise': 'expertise_1'})

    df_odd = df_odd.rename(columns={'user_id': 'user_id_2', 'value_a': 'value_a_2', 'value_b': 'value_b_2',
                                    'value_c': 'value_c_2', 'days_from_start': 'days_from_start_2',
                                    'distance': 'distance_2', 'expertise': 'expertise_2'})
    df_odd = df_odd.drop(['ifp_id', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'a', 'b', 'c'], 1)

    df_odd = df_odd.reset_index(drop=True)
    df_even = df_even.reset_index(drop=True)

    df_complete = pd.concat([df_even, df_odd], axis=1)
    print("Size of dataset: " + str(len(df_complete)))

    print("Writing to csv....")
    df_complete.to_csv(outfile, index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--inputsurv", help="the input dataset, representing the good judgement yearly survey csv file", default="survey_fcasts.yr1.csv")
parser.add_argument("--inputquest", help="the input dataset, representing the good judgement yearly survey csv file", default="questions_w_topics.csv")
parser.add_argument("--outputf", help="the output: a processed file that can be used to feed a general neural network", default="ds.csv")
parser.add_argument("--ifipdir", help="the directory where ifip pairs are stored", default="ifips/")
parser.add_argument('--dontformat', help="Do not format to ds.csv.", action='store_true')
parser.add_argument("--dontsplit", help="Do not split output by ifip.", action='store_true')
parser.add_argument("--dontpair", help="Do not pairs for each ifip.", action='store_true')

args = parser.parse_args()

if not args.dontformat:
    ifps_raw = pd.read_csv(args.inputquest, sep=None, engine='python', parse_dates=['date_start'])
    survey_1st_raw = pd.read_csv(args.inputsurv, sep=None, engine='python', parse_dates=['fcast_date','timestamp'], dtype={'user_id': object})

    ifps_raw.dropna(inplace=True)

    ifps_needed_columns = ["ifp_id","topic_0","topic_1","topic_2","topic_3","topic_4","topic_5","date_start","outcome","n_opts","options"]

    ifps_w_options = ifps_raw.loc[:, ifps_needed_columns]


    survey_needed_columns = ["ifp_id","user_id","expertise","answer_option","value","fcast_date"]

    survey = survey_1st_raw.loc[:, survey_needed_columns]


    ex1 = ifps_w_options

    ex2 = survey

    c = pd.merge(ex1,ex2,on='ifp_id')
    preferred_column_order = ['ifp_id',
     'topic_0',
     'topic_1',
     'topic_2',
     'topic_3',
     'topic_4',
     'topic_5',
     'date_start',
     'n_opts',
     'user_id',
     'expertise',
     'answer_option',
     'value',
     'fcast_date',
     'outcome']
    c = c[preferred_column_order]

    num_topics = 6


    middle_columns = utils.generate_column_names_per_user(3)
    columns = [
        'ifp_id',
        'topic_0',
        'topic_1',
        'topic_2',
        'topic_3',
        'topic_4',
        'topic_5',
        'expertise',
        'date_start',
        'n_opts',
        'user_id']
    columns += middle_columns
    columns.append('outcome')

    ds = pd.DataFrame(columns=columns, index=np.arange(survey.shape[0]))

    # There is a question with 5 options, at the moment we get rid of that
    c = c[c.n_opts != 5]

    counter_opts = 0
    out_index = 0

    print("Formatting " + args.inputsurv + "...")
    print("Number of rows: {}".format(c.shape[0]))
    for index, row in tqdm(c.iterrows()):
        current_nopts = row["n_opts"]
        ds.loc[out_index]["ifp_id"] = row["ifp_id"]
        ds.loc[out_index]["topic_0"] = row["topic_0"]
        ds.loc[out_index]["topic_1"] = row["topic_1"]
        ds.loc[out_index]["topic_2"] = row["topic_2"]
        ds.loc[out_index]["topic_3"] = row["topic_3"]
        ds.loc[out_index]["topic_4"] = row["topic_4"]
        ds.loc[out_index]["topic_5"] = row["topic_5"]
        ds.loc[out_index]["expertise"] = row["expertise"]
        ds.loc[out_index]["date_start"] = row["date_start"]
        ds.loc[out_index]["n_opts"] = row["n_opts"]
        ds.loc[out_index]["outcome"] = row["outcome"]

        ds.loc[out_index]["user_id"] = row["user_id"]
        ds.loc[out_index]["answer_option_"+ chr(counter_opts+ord('a'))] = row["answer_option"]
        ds.loc[out_index]["value_"+chr(counter_opts+ord('a'))] = row["value"]
        ds.loc[out_index]["fcast_date_" + chr(counter_opts+ord('a'))] = row["fcast_date"]
        counter_opts += 1
        if current_nopts == 2 and counter_opts == 2:
            ds.loc[out_index]["answer_option_" + chr(counter_opts+ord('a'))] = "c"
            ds.loc[out_index]["value_" + chr(counter_opts+ord('a'))] = 0.0
            ds.loc[out_index]["fcast_date_" + chr(counter_opts+ord('a'))] = row["fcast_date"]
            counter_opts = 0
            out_index += 1
        elif current_nopts == 3 and counter_opts == 3:
            counter_opts = 0
            out_index += 1



    ds.dropna(how='all', inplace = True)


    ds["days_from_start"] = (ds["fcast_date_a"] - ds["date_start"]).dt.days
    ds.loc[ds.days_from_start < 0,'days_from_start'] = 0
    ds.drop(['answer_option_a', 'answer_option_b', 'answer_option_c', 'n_opts', 'fcast_date_a', 'fcast_date_b', 'fcast_date_c', 'date_start'], axis=1, inplace = True)
    cols = ds.columns.tolist()
    cols = cols[:-2] + [cols[-1]] + [cols[-2]]


    ds = ds[cols]

    outcome_one_hot = pd.get_dummies(ds['outcome'])
    ds.drop('outcome', 1, inplace = True)

    ds = ds.join(outcome_one_hot)

    def compute_euclidean_distance(row):
        a = np.array([row['value_a'], row['value_b'], row['value_c']])
        b = np.array([row['a'], row['b'], row['c']])
        return distance.euclidean(a, b)

    ds['distance'] = ds.apply(compute_euclidean_distance, axis=1)

    print('writing dataset to disk...')
    ds.to_csv(args.outputf, index = False)

## Now split the dataset by ifip_id. For survey_fcasts.yr1.csv there are 104 ifips, numbered 1001 to 1104
if not args.dontsplit:
    print("Splitting dataset into ifips...")

    ##create header file
    #cmd = "echo ifp_id,topic_0,topic_1,topic_2,topic_3,topic_4,topic_5,a,b,c,user_id_1,value_a_1,value_b_1,value_c_1,days_from_start_1,distance_1,expertise_1,user_id_2,value_a_2,value_b_2,value_c_2,days_from_start_2,distance_2,expertise_2 > header.csv"
    cmd = "echo ifp_id,topic_0,topic_1,topic_2,topic_3,topic_4,topic_5,expertise,user_id,value_a,value_b,value_c,days_from_start,a,b,c,distance > header.csv"
    subprocess.Popen(cmd, shell=True).communicate()

    for i in range(1, 10):
        cmd = "grep '^100" + str(i) + "-0' ds.csv > " + args.ifipdir + "ds_100" + str(i) + "-0.csv"
        cmd += " && cat header.csv " + args.ifipdir + "ds_100" + str(i) + "-0.csv >> " + args.ifipdir + "ds1_100" + str(i) + "-0.csv"
        cmd += " && mv " + args.ifipdir + "ds1_100" + str(i) + "-0.csv " + args.ifipdir + "ds_100" + str(i) + "-0.csv"
        subprocess.Popen(cmd, shell=True).communicate()

    for i in range(10, 100):
        cmd = "grep '^10" + str(i) + "-0' ds.csv > " + args.ifipdir + "ds_10" + str(i) + "-0.csv"
        cmd += " && cat header.csv " + args.ifipdir + "ds_10" + str(i) + "-0.csv >> " + args.ifipdir + "ds1_10" + str(i) + "-0.csv"
        cmd += " && mv " + args.ifipdir + "ds1_10" + str(i) + "-0.csv " + args.ifipdir + "ds_10" + str(i) + "-0.csv"
        subprocess.Popen(cmd, shell=True).communicate()

    for i in range(100, 105):
        cmd = "grep '^1" + str(i) + "-0' ds.csv > " + args.ifipdir + "ds_1" + str(i) + "-0.csv"
        cmd += " && cat header.csv " + args.ifipdir + "ds_1" + str(i) + "-0.csv >> " + args.ifipdir + "ds1_1" + str(i) + "-0.csv"
        cmd += " && mv " + args.ifipdir + "ds1_1" + str(i) + "-0.csv " + args.ifipdir + "ds_1" + str(i) + "-0.csv"
        subprocess.Popen(cmd, shell=True).communicate()

##generate pairs for siamese net for each ifip.
if not args.dontpair:
    files = os.listdir(args.ifipdir)
    files.sort()
    for infile in files:
        generate_pairs_for_ifip(args.ifipdir + infile, args.ifipdir + "pairs_"+infile)

