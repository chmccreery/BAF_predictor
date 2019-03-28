import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

greens = ['HSES2M','TTC1240','G401','VAESBJ']
yellows = ['EOL1','JURKAT','MOLM13']
blues = ['293T','SLR24']
oranges = ['BIN67','ES2']
greys = ['HEPG2']

group = ['_' + item for item in greens]
print(group)

df = pd.read_csv('raw_data/BAF_peakMatch_25022018.csv')
# Filter by subunit
brg1 = df.filter(regex='BRG1',axis=1)
for primary in group:
    # Get related cell lines only
    related = [item for item in group if item != primary]
    related = '|'.join(related)
    print(primary)
    print(related)
    related = brg1.filter(regex=related)
    # Get true results from CHIPseq on line of interest
    gold = brg1.filter(regex=primary)
    count = related.sum(axis=1).values
    predictions = (count > 1).astype(int).squeeze()
    zeros = np.zeros(predictions.shape,dtype=int)
    gold = gold.values.squeeze()

    print('F1: {:.1f}'.format(100*f1_score(gold,predictions)))
    print('Precision: {:.1f}'.format(100*precision_score(gold,predictions)))
    print('Recall: {:.1f}'.format(100*recall_score(gold,predictions)))
    print('Accuracy: {:.1f}'.format(100*accuracy_score(gold,predictions)))
    print('Zeros Accuracy: {:.1f}'.format(100*accuracy_score(gold,zeros)))
