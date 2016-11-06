import tensorflow as tf

def wordids_to_tensors(wordids, embedding_dim, vocab_size, seed=0):
    '''Convert a list of wordids into embeddings and biases.

    This function creates a variable for the embedding matrix, dimension |E x V|
    and a variable to hold the biases, dimension |V|.

    It returns an op that will accept the output of "wordids" op and lookup
    the corresponding embedding vector and bias in the table.

    Args:
      - wordids |W|: a tensor of wordids
      - embedding_dim, E: a scalar value of the # of dimensions in which to embed words
      - vocab_size |V|: # of terms in the vocabulary

    Returns:
      - a tuple (w, b, m) where w is a tensor of word embeddings and b is a vector
        of biases.  w is |W x E| and b is |W|.  m is the full |V x E| embedding matrix.
        Each of these should contain values of type tf.float32.

    To get the tests to pass, initialize w with a random_uniform [-1, 1] using
    the provided seed.

    As usual the "b" vector should be initialized to 0.
    '''
    # START YOUR CODE

    m = tf.get_variable("m", shape=[vocab_size, embedding_dim],
                    initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
    biases = tf.get_variable("b", shape=[vocab_size], initializer = tf.constant_initializer(value = 0.))
    
    w = tf.nn.embedding_lookup(m, wordids)
    b = tf.nn.embedding_lookup(biases, wordids)
    return(w, b, m)

    # END YOUR CODE


def example_weight(Xij, x_max, alpha):
    '''Scale the count according to Equation (9) in the Glove paper.

    This runs as part of the TensorFlow graph.  You must do this with
    TensorFlow ops.

    Args:
      - Xij: a |batch| tensor of counts.
      - x_max: a scalar, see paper.
      - alpha: a scalar, see paper.

    Returns:
      - A vector of corresponding weights.
    '''
    # START YOUR CODE

    return tf.minimum(tf.pow(tf.div(Xij, x_max), alpha), tf.constant(1.))

    # END YOUR CODE


def loss(w, b, w_c, b_c, c):
    '''Compute the loss for each of training examples.

    Args:
      - w |batch_size x embedding_dim|: word vectors for the batch
      - b |batch_size|: biases for these words
      - w_c |batch_size x embedding_dim|: context word vectors for the batch
      - b_c |batch_size|: biases for context words
      - c |batch_size|: # of times context word appeared in context of word

    Returns:
      - loss |batch_size|: the loss of each example in the batch
    '''
    # START YOUR CODE
    print w_c
    print w
    return tf.reduce_sum(tf.sub(tf.add(tf.add(tf.matmul(w, tf.transpose(w_c)), b_c ), b), tf.log(c)),1)
    #return tf.add(tf.add(tf.reduce_sum(tf.mul(w, w_c), 1), b_c), b)
    # END YOUR CODE
