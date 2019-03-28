import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
np.random.seed(seed=0)

def print_metrics(gold,predictions,random,zeros):
    """
    Print baseline metrics such as f1, accuracy, precision, and recall with baselines for comparison
        - random can be 50/50 or split by percentage of bins where BAF binds in average cell line
    :param gold: numpy array of 0/1 for no-BAF/BAF as output by CHIPSeq
    :param predictions: numppy array of 0/1 for no-BAF/BAF as predicted by majority vote over other cells
    :param random: numpy array of 0/1 for comparison to intelligent methods
    :param zeros: numpy array of 0 for performance metrics when always guessing majority class
    """
    print('F1 Random Guess: {:.1f}'.format(100*f1_score(gold,random)))
    print('F1: {:.1f}'.format(100*f1_score(gold,predictions)))
    print('Precision: {:.1f}'.format(100*precision_score(gold,predictions)))
    print('Recall: {:.1f}'.format(100*recall_score(gold,predictions)))
    print('Accuracy: {:.1f}'.format(100*accuracy_score(gold,predictions)))
    print('Random Guess Accuracy: {:.1f}'.format(100*accuracy_score(gold,random)))
    print('Zeros Accuracy: {:.1f}'.format(100*accuracy_score(gold,zeros)))

def get_related_cell_lines(key):
    """
    get a family of cell lines over which to perform majority voting
    :param key: string indicating which family of cell lines to use in majority voting 
        -example: 'oranges' or 'all_lines'
    :returns: list of strings that indicate cell line name 
        -example: ['BIN67','ES2']
    """
    related_lines = {'greens': ['VAESBJ','HSES2M','TTC1240','G401'],
        'yellows': ['EOL1','JURKAT','MOLM13'],
        'blues': ['293T','SLR24'],
        'oranges': ['BIN67','ES2'],
        'greys': ['HEPG2']
        }
    all_lines = []
    for _,vals in related_lines.items():
        all_lines = all_lines + vals
    related_lines['all_lines'] = all_lines
    group = ['_' + item for item in related_lines[key]]
    return group

def get_filtered_df(regex):
    """
    Load CHIPSeq data from csv and filter by subunit or other substring in name
    :param regex: string that is in the column name for all columns of interest
    :returns filtered: pandas dataframe with only columns in which regex is a substring
    """
    df = pd.read_csv('raw_data/BAF_peakMatch_25022018.csv')
    # Filter by subunit
    filtered = df.filter(regex=regex,axis=1)
    return filtered

def get_predictions(primary,group,df):
    """
    Predicts whether or not BAF binds at each bin (row) of the primary cell line 
    using the other cell lines in group
    :param primary: string indicating name of cell line to predict
    :param group: list of strings corresponding to cell lines in same group as primary (including primary)
    :param df: dataframe with 0/1 corresponding to no-BAF/BAF respectively for each cell line in group
    :returns predictions: a numpy array of length equal to number of rows in df.
        Each entry corresponds to a DNA bin from Basset
    """
    # Get related cell lines only
    related = [item for item in group if item != primary]
    related = '|'.join(related)
    print("\nPredicting:")
    print(primary[1:])
    print("Using:")
    print(related)
    related = df.filter(regex=related)
    # Get number of related cell lines where BAF bound (break ties with random vote)
    num_relatives = len(group)-1
    count = related.sum(axis=1).values
    if num_relatives % 2 == 0:
        count = count + np.random.randint(2,size=count.shape)
    # Get predictions vector using majority vote from related cell lines and 0s for comparison
    predictions = (count > num_relatives/2).astype(int).squeeze()
    return predictions 

if __name__ == '__main__':
    """
    Uses data in the raw_data folder to predict BAF-subunit binding in every cell line of a given family
    """
    parser = argparse.ArgumentParser(description='Calculate Baseline Metrics for BAF binding')
    parser.add_argument('--subunit',default='BRG1', type=str, help='subunit to filter by (if any)')
    parser.add_argument('--family',default='all_lines', type=str, help='family of cell lines to use in majority vote')
    args = parser.parse_args()
    filtered_df = get_filtered_df(args.subunit)
    group = get_related_cell_lines(args.family)
    for primary in group:
        # Get true results from CHIPseq on line of interest
        gold = filtered_df.filter(regex=primary).values.squeeze()
        # Get predicted results from taking majority vote over other cell lines
        predictions = get_predictions(primary,group,filtered_df)
        # As a baseline, see what would happen if randomly guessed
        random = np.random.choice(2,size=predictions.shape,p=[1-0.055,0.055])
        # As a baseline, see what would happen if always guessed 0 (majority class)
        zeros = np.zeros(predictions.shape,dtype=int)
        print_metrics(gold,predictions,random,zeros)
