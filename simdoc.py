import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import csv
import re

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))

df = pd.read_csv(args.vectorfile,header=None,low_memory=False)
print(df)
filenames = list(df[0])
filenames.remove('filename') #remove header because header=None ain't working

topic_names = list(set([re.sub(r'/\w+.txt', '', filename) for filename in filenames]))
topics = {}

for topic in topic_names:
    '''Split data into two matrices for comparison
    '''
    topic_index = df[0].str.contains(topic) #get boolean – topic? True/False
    topic_df = df[topic_index] #single out vectors where topic = True
    topic_df = topic_df.drop(topic_df.columns[0],axis='columns')
    vectors = [np.array(topic_[1]) for topic_ in topic_df.iterrows()]
    topics.update({topic: vectors})

topic_vectors = [topics[topic] for topic in topic_names]

def cosine(vectors1, vectors2, i, n):
    '''Calculate cosine similarity between grain and crude
    Args:
        vectors1(array)
            array of count vectors
        vectors2(array)
            array of count vectors for vectors1 to be compared with
    
    '''

    no_files = len(vectors1) # should be == len(vectors2)
    avg_cos = 0
    cossim = 0

    print('Calculating cosine similarity...')

    for vector in vectors1:
        vector = vector.reshape(1,-1)
        if cossim is not 0:
            avg_cos += cossim/no_files
        cossim = 0
        for item in vectors2:
            item = item.reshape(1,-1)
            cos = cosine_similarity(vector, item)
            cossim += cos
    
    print('average cos between', topic_names[i], 'and', topic_names[n])
    avg_cos = avg_cos/no_files
    print(avg_cos)

cosine(topic_vectors[0], topic_vectors[1], 0, 1)
cosine(topic_vectors[0], topic_vectors[0], 0, 0)
cosine(topic_vectors[1], topic_vectors[0], 1, 0)
cosine(topic_vectors[1], topic_vectors[1], 1, 1)
