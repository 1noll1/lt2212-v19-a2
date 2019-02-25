import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))

matrix = pd.read_csv(args.vectorfile)
df = pd.DataFrame(matrix)
df.index = df["filename"]
df.drop("filename", axis=1,inplace=True)

grain = [filename for filename in df.index if 'grain' in filename]
crude = [filename for filename in df.index if 'crude' in filename]

matrix = np.array(df)
grain_matrix = df.loc[grain]
crude_matrix = df.loc[crude]
print("Average similarity, other topic:")
print(cosine_similarity(grain_matrix, crude_matrix))
print(pd.DataFrame(cosine_similarity(grain_matrix, crude_matrix)))
print(cosine_similarity(grain_matrix, crude_matrix))
print("Average similarity, same topic:")
print(cosine_similarity(grain_matrix, grain_matrix))
print(cosine_similarity(crude_matrix, crude_matrix))
#print(matrix[0:1])
#print(matrix)
#print(cosine_similarity(matrix[0:1],matrix))

'''For each of the two topics, print
the average similarity of every document vector with that topic to every other vector with that same topic
(for simplicity, including itself, if you want), averaged over the entire topic.
the average similarity of every document vector with that topic to every document vector in the other topic,
averaged over the entire topic.
'''
