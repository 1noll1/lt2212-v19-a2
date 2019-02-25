
import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
#import string

parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()

m = args.basedims

if args.basedims:
    cv = CountVectorizer(max_features=m)
else:
    cv = CountVectorizer()

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))

def fetchcorpus(foldername): # this would be scratch
    '''Read all words and store them in an array.
    Input:
       	foldername (str)
            The base folder name containing the two topic subfolders
    Returns:
       	corpus (array):
            All the documents (as strings)
    '''
    #remove_punct = str.maketrans('', '', string.punctuation)
    corpus = []

    base_folders = os.listdir(args.foldername) #list of both topics
    for folder in base_folders:
       	subfolders = glob.glob('{}/{}/article*'.format(args.foldername, folder)) #get list of all textfiles
       	for file_path in subfolders:
            '''For each document, remove punctuation and create word vectors
            '''
            with open(file_path, 'r') as f:
                docstring = ''
               	for line in f:
                    line = re.sub(r'\n|\d+|["\'!.,;\-&]', '', line) #ye olde punctuation strip
                    #print(line)
                    docstring += line
            corpus.append(docstring)
    return corpus

def create_vectors(corpus):
#corpus = fetchtokens(args.foldername)
    #x = cv.fit_transform(corpus)
    x = cv.fit_transform(corpus)
    if args.tfidf:
        print("Applying tf-idf to raw counts.")
        tfidf_transformer = TfidfTransformer()
        x = tfidf_transformer.fit_transform(x)
    matrix = x.toarray()
    if args.svddims:
        TS = TruncatedSVD(args.svddims)
        matrix = TS.fit_transform(matrix)
    df = pd.DataFrame(matrix)
    df.columns = cv.get_feature_names() #not fitted for svddims ;(
    base_folders = os.listdir(args.foldername)
    df.index = glob.glob('{}/*/article*'.format(args.foldername))
    #print(df)
    return df

def dubs(df):
    '''Find and delete all duplicate vectors
    '''
    duplicates = df[df.duplicated(keep=False)]
    duplicate_index = duplicates.index
#       print(duplicates)
#       print(duplicate_index)
    for duplicate in duplicate_index:
       	print('Duplicate {} removed'.format(duplicate))
    df.drop_duplicates()
    #print(df, file=args.outputfile)
    return df

def write_file(df):
    with open(args.outputfile, 'w+') as f:
        print(df, file=f)
    #df.to_csv(args.outputfile,index_label='filename')

if __name__ == '__main__':
    print("Loading data from directory {}.".format(args.foldername))
    corpus = fetchcorpus(args.foldername)
    df = create_vectors(corpus)
    print("Eliminating duplicate vectors:")
    dubs(df)
    print("Writing matrix to {}.".format(args.outputfile))
    write_file(df)
