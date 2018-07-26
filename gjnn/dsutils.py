import urllib3 as urllib
import os
import errno
import zipfile
import logging
from sklearn import preprocessing

logger = logging.getLogger(__name__)


url = 'https://dataverse.harvard.edu/api/access/datafiles/2917332,2917333,2917331,2917330,2917334,2917335,2917336,2917337,2917339,2917340,2917341,2917342,2917343,2917344,2917345,2917346,2917347,2917348,2917349,2917338,2917350,2917351,2917352,2917353,2917354'

ds_directory      = ".dscache"
extract_directory = ".dscache/dataset"

def download_dataset():
    try:
            os.makedirs(ds_directory)
    except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    zip_ds = ds_directory + '/dataset.zip'
    if not (os.path.isdir(extract_directory)):
        if not (os.path.isfile(zip_ds)):
            urllib.request.urlretrieve(url, zip_ds)
        unzip_dataset(zip_ds, extract_directory)
    else:
        print("Dataset already ready")


def unzip_dataset(zip_file_path, extract_dir):
    zip_ref = zipfile.ZipFile(zip_file_path, 'r')
    zip_ref.extractall(extract_dir)
    zip_ref.close()



def generate_column_names(forecaster_set_size=3, number_options=3):
    """
    This function generates a list which will be used for columns,
    
    Arguments:
    forecaster_set_size -- this determines the number of forecasters we will use per row (by default=3)
    number_options -- this determines the number of options a forecaster has to fultill (by default=3)
    
    Returns:
    column_names -- a list containing the column names
    Example of return list:
    'user_id_1_a',
    'answer_option_1_a',
    'value_1_a',
    
    'user_id_1_b',
    'answer_option_1_b',
    'value_1_b',
    
    'user_id_1',
    'answer_option_1_c',
    'value_1_c',
    
    fcast_date_1,
    
    'user_id_2_a',
    'answer_option_2_a',
    'value_2_a', 
    ...
    
    """
    if number_options < 1 or number_options > 26:
        raise ValueError('Number of options out of range, sould be between 1 and 26')
    column_names = []
    
    
    for f_id in range(forecaster_set_size):
        for n_opt in range(number_options):
            if n_opt == 0:
                column_names.append("user_id_{}".format(f_id+1))
            column_names.append("answer_option_{}_{}".format(f_id+1,chr(n_opt+ord('a'))))
            column_names.append("value_{}_{}".format(f_id+1,chr(n_opt+ord('a'))))
        column_names.append("date_fcast_{}".format(f_id+1))
    return column_names



def generate_column_names_per_user(number_options=3):
    """
    This function generates a list which will be used for columns,
    
    Arguments:
    number_options -- this determines the number of options a forecaster has to fultill (by default=3)
    
    Returns:
    column_names -- a list containing the column names
    Example of return list:
    'user_id',
    
    'answer_option_a',
    'value_a',
    'fcast_date_a',
    
    'answer_option_b',
    'value_b',
    'fcast_date_b',
    
    'answer_option_c',
    'value_c',
    'fcast_date_c',
    
    """
    if number_options < 1 or number_options > 26:
        raise ValueError('Number of options out of range, sould be between 1 and 26')
    column_names = []
    
    
    for n_opt in range(number_options):
        column_names.append("answer_option_{}".format(chr(n_opt+ord('a'))))
        column_names.append("value_{}".format(chr(n_opt+ord('a'))))
        column_names.append("fcast_date_{}".format(chr(n_opt+ord('a'))))
    return column_names

import subprocess
for i in range(50, 100):
     cmd = "grep '^10"+str(i)+"-0' ds_with_combinations_yr1.csv > ds_10"+str(i)+"-0.csv"
     subprocess.Popen(cmd, shell=True).communicate()
     cmd = "cat ds_10" + str(i) + "-0.csv >> header.csv"
     subprocess.Popen(cmd, shell=True).communicate()
     cmd = "mv header.csv ds_10" + str(i) + "-0.csv"
     subprocess.Popen(cmd, shell=True).communicate()
     cmd = "cp header2.csv header.csv"
     subprocess.Popen(cmd, shell=True).communicate()

for i in range(100, 105):
     cmd = "grep '^1"+str(i)+"-0' ds_with_combinations_yr1.csv > ds_1"+str(i)+"-0.csv"
     subprocess.Popen(cmd, shell=True).communicate()
     cmd = "cat ds_1" + str(i) + "-0.csv >> header.csv"
     subprocess.Popen(cmd, shell=True).communicate()
     cmd = "mv header.csv ds_1" + str(i) + "-0.csv"
     subprocess.Popen(cmd, shell=True).communicate()
     cmd = "cp header2.csv header.csv"
     subprocess.Popen(cmd, shell=True).communicate()
