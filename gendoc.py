import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions.")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing.")
parser.add_argument("--to-csv", dest="csv", action="store_true",
                    help="Save output to csv file.")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

def counts(document):
    return nltk.FreqDist(document)

def fetchvocab(foldername): # this would be scratch
    '''Read all words and store them in an array.
    Input:
       	foldername (str)
            The base folder name containing the two topic subfolders
    Returns:
       	vocab (list)
            All words of all documents, alphabetically ordered
        vectors (dictionary)
            Mapping from doc names to raw word counts, same order as vocab
    '''
    vocab = []
    doctexts = {}
    vectors = {}

    stop_words = set(stopwords.words('english'))

    base_folders = os.listdir(args.foldername)
    for folder in base_folders:
        subfolders = glob.glob('{}/{}/article*'.format(args.foldername, folder))
        for file_path in subfolders:
            '''For each document, remove punctuation and add each word in the document to vocabulary
            '''
            with open(file_path, 'r') as f:
                doc = f.read().lower()
                doc = re.sub(r'\n|\d+|["\'!.,;\-&:<>()/]', '', doc).split() #remove punctuation and split per token
                vocab.extend([w for w in doc if w not in stop_words]) #add every word to vocab, remove stopwords
                file_name = re.sub(r'{}/'.format(args.foldername), '', file_path)
                doctexts.update({file_name: doc})
                vectors.update({file_name: {}})

    if args.basedims: #if M is set, include only top M counts from corpus in vocab
        M = args.basedims
        vcounts = counts(vocab)
        vocab = [x[0] for x in vcounts.most_common(M)]

    for doctext in doctexts: #filter out all words not in most common
        doctexts[doctext] = list(filter(lambda x: x in vocab, doctexts[doctext]))

    empty_vectors = {word:0 for word in vocab}
    for vector in vectors:
        vectors[vector].update(empty_vectors)

    for doctext in doctexts: #
        doccounts = counts(doctexts[doctext]) #count words in text doc
        vectors[doctext].update(doccounts) #add raw word counts to vector
        sorted_ = sorted(list(vectors[doctext].items()))
        vectors[doctext] = [x[1] for x in sorted_]

    vocab = sorted(set(vocab))

    return vocab, vectors

def create_vectors(corpus):
    vocab, vectors = corpus
    
    df = pd.DataFrame.from_dict(vectors, orient='index')
    df.columns = vocab
    # df.index.names = ['filepath/filename']
    filenames = df.index
    #df.set_columns('filepath/filename')
    print("Eliminating duplicate vectors:")
    dubs(df)
    print(df)

    if args.tfidf:
        print("Applying tf-idf to raw counts.")
        tfidf_transformer = TfidfTransformer()
        df = tfidf_transformer.fit_transform(df)
        matrix = df.toarray() #just to be able to create a new df
        df = pd.DataFrame(matrix, columns=vocab, index=filenames)

    if args.svddims:
        print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
        tsvd = TruncatedSVD(n_components=args.svddims)
        tsvd.fit(df)
        matrix = tsvd.transform(df)
        df = pd.DataFrame(matrix, columns=[i for i in range(0,args.svddims)], index=filenames) 

    pd.options.display.max_rows = 999
    pd.options.display.max_columns = (len(vocab) + 1)

    return df

def dubs(df):
    '''Find and delete all duplicate vectors
    '''
    duplicates = df[df.duplicated(keep=False)]
    duplicate_index = duplicates.index
    for duplicate in duplicate_index:
        print('Duplicate {} removed'.format(duplicate))
    df.drop_duplicates()
    return df

def write_file(df):
    '''If option --to-csv is True, the output is written to a csv file (for further processing),
    else the terminal output is piped to a txt file.
    '''
    if args.csv == True:
        df.to_csv(args.outputfile,index_label='filename')
    else:
        with open(args.outputfile, 'w+') as f:
            print(df, file=f)

if __name__ == '__main__':
    print("Loading data from directory {}.".format(args.foldername))
    corpus = fetchvocab(args.foldername)
    df = create_vectors(corpus)
    print("Writing matrix to {}.".format(args.outputfile))
    write_file(df)


