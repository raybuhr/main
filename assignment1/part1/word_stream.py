def context_windows(words, C=5):
    '''A generator that yields context tuples of words, length C.
       Don't worry about emitting cases where we get too close to
       one end or the other of the array.

       Your code should be quite short and of the form:
       for ...:
         yield the_next_window
    '''
    # START YOUR CODE HERE
    for i in range(len(words)):
        if len(words[i:i+C]) == C:
            yield(words[i:i+C])
    # END YOUR CODE HERE


def cooccurrence_table(words, C=2):
    '''Generate cooccurrence table of words.
    Args:
       - words: a list of words
       - C: the # of words before and the number of words after
            to include when computing co-occurrence.
            Note: the total window size will therefore
            be 2 * C + 1.
    Returns:
       A list of tuples of (word, context_word, count).
       W1 occuring within the context of W2, d tokens away
       should contribute 1/d to the count of (W1, W2).
    '''
    table = []
    # START YOUR CODE HERE
    codict = {}
    for item in context_windows(words, C*2+1):
        for i in range(C*2+1):
            if i == C: 
                pass
            else:
                try:
                    codict[item[C], item[i]] += 1./abs(C-i)
                except:
                    codict[item[C], item[i]] = 1./abs(C-i)
    table = [(key[0], key[1], codict[key]) for key in codict]
    # END YOUR CODE HERE
    return table


def score_bigram(bigram, unigram_counts, bigram_counts, delta):
    '''Return the score of bigram.
    See Section 4 of Word2Vec (see notebook for link).

    Args:
      - bigram: the bigram to score: ('w1', 'w2')
      - unigram_counts: a map from word => count
      - bigram_counts: a map from ('w1', 'w2') => count
      - delta: the adjustment factor
    '''
    # START YOUR CODE HERE
    if bigram in bigram_counts:
        return(1.0*(bigram_counts[bigram] - delta)/(unigram_counts[bigram[0]]*unigram_counts[bigram[1]]))
    else:
        return(1.0*0) 
    # END YOUR CODE HERE
