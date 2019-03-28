import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
np.random.seed(seed=0)

def print_metrics(gold,predictions,random,zeros):
    print('F1 Random Guess: {:.1f}'.format(100*f1_score(gold,random)))
    print('F1: {:.1f}'.format(100*f1_score(gold,predictions)))
    print('Precision: {:.1f}'.format(100*precision_score(gold,predictions)))
    print('Recall: {:.1f}'.format(100*recall_score(gold,predictions)))
    print('Accuracy: {:.1f}'.format(100*accuracy_score(gold,predictions)))
    print('Random Guess Accuracy: {:.1f}'.format(100*accuracy_score(gold,random)))
    print('Zeros Accuracy: {:.1f}'.format(100*accuracy_score(gold,zeros)))

def get_related_cell_lines(key):
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
    df = pd.read_csv('raw_data/BAF_peakMatch_25022018.csv')
    # Filter by subunit
    brg1 = df.filter(regex=regex,axis=1)
    return brg1

def get_predictions(primary,group,df):
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
