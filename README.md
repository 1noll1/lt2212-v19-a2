# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Linnea Strand

## Additional instructions

The instructions asked for the output to be "minimally human readable". For this reason, I added the command --to-csv, which is used to generate the csv file to be used in simdoc.py. If you only want to have a look at the data, leave out this option and the terminal output is piped to a txt.

## Results and discussion

I chose 50 words as base dims to compare with the unrestricted vocab (10k+ words). This massive reduction was just to show what a difference the amount of data makes. Despite having removed stopwords, it seems fairly obvious that restricting base dims to as few as 50 results in having more words common between the two topics.


### Result table

 commands | grain-crude | crude-grain | grain-grain | crude-crude 
 -------- | ----------- | ----------- | ----------- | -----------
 unrestricted | 0.08154768 | 0.08259355 | 0.14017378 | 0.16903438
 -B50 | 0.20064142 | 0.20326421 | 0.3178804] | 0.40708906
 --tfidf | 0.02246328 | 0.02246328 | 0.05640822 | 0.07975089
 -B50 --tfidf | 0.11712405 | 0.11870799 | 0.25873433 | 0.29264549
 --svd100 | 0.14817596 | 0.15015391 | 0.25616774 | 0.30308441
 --svd1000 | 0.08276389 | 0.08171589 | 0.14080625 | 0.16950324
 --svd100 --tfidf | 0.06129149 | 0.06217377 | 0.13264903 | 0.14989123
 --svd1000 --tfidf | 0.02257613 | 0.02287987 | 0.05685462 | 0.0566786
 
 Values in columns grain-crude and crude-grain are nearly 100 % symmetrical as they should be. The differences, I suspect, is simply *floats*. Too many floats.

### The hypothesis in your own words

I believe the goal of this experiment was to show us what vector transformations (=what type of data) work better than others in classification tasks.

One might guess that
1. lower basedims => vectors more alike, and
2. more fancy transformation => greater difference between grain vs. crude and vice versa than comparison within the same category, as these methods are intended to capture important words.

### Discussion of trends in results in light of the hypothesis

Some of the numbers look reasonable, such as lower basedims => vectors being more similar. However, I'm surprised to find that applying both SVD and tdidf returns such low cosine similarity scores. These are both methods that should capture word weights, and I would therefore suspect them to have a certain impact on the results (making grain-crude significantly less similar than e.g. grain-grain). However, this proportion doesn't seem to change compared to the other vector transformations. Perhaps these methods "cancel each other out" – fewer dims means fewer words that can be used to distinguish documents.

It is also clear from my results that crude vectors are more "like themselves". My guess would be that "crude" appearing with "oil" is what makes these documents easier to distinguish, while "grain" could really refer to any grain, making this a more diverse and chaotic topic.

## Bonus answers

Suggested improvements: 
* Working with ngrams rather than unigram counts/bag of words to better capture context
* Lemmatization to give a more fair picture of the word counts
* Narrow down out target classes (grain could be wheat)
