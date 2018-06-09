#Building a Chatbot with Deep NLP

#Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time

####### Part - 1 DATA PREPROCESSING ###########

lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

#Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')  #_ denotes a local variable
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

#Creating a list of conversations
conversations_ids = []
for conversation in conversations[:-1]:  #Since last row of conversations is an empty row
    _coversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ", "")
    conversations_ids.append(_coversation.split(','))
    
#Getting separately the questions and answers
questions = []
answers = []

for conv in conversations_ids:
    for i in range(len(conv) - 1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
        
        
#Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am",text)
    text = re.sub(r"he's", "he is",text)
    text = re.sub(r"she's", "she is",text)
    text = re.sub(r"that's", "that is",text)
    text = re.sub(r"what's", "what is",text)
    text = re.sub(r"where's", "where is",text)
    text = re.sub(r"\'ll", " will",text)
    text = re.sub(r"\'ve", " have",text)
    text = re.sub(r"\'re", " are",text)
    text = re.sub(r"\'d", "would",text)
    text = re.sub(r"wont't", "will not",text)
    text = re.sub(r"can't", "cannot",text)
    text = re.sub(r"[-()\"'#/@;:<>+=~|.?,]", "",text)
    return text                 

#Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
#Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
#Creating a dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] = word2count[word] + 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] = word2count[word] + 1  

#Creating two dictionaries that maps the question words and the answer words to a unique integer
#Removing 5% least frequent words
threshold = 20
questionswords2int = {}
word_number = 0
for word,count in word2count.items():
    if count > threshold:
        questionswords2int[word] = word_number
        word_number += 1
        
answerswords2int = {}
word_number = 0
for word,count in word2count.items():
    if count > threshold:
        answerswords2int[word] = word_number
        word_number += 1  

#Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1 
    
#Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i:w for w,w_i in answerswords2int.items()}

#Adding the End of string token to end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
#translating all the questions and answers into integers
# and replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:    
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
    
#sorting questions and answers by the length of the questions
# We do this because this will speed up the training and reduce the loss. This is cuz it reduces the padding

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25+1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
        
    

##########3# PART-2 - BUILDING THE SEQ2SEQ MODEL #######

# Creating placeholders for the inputs and the targets
# In tensorflow , all variables are used in tensors.
# Tensors are advanced numpy array where all elements are of same type.
# All variables in a tensor must be defined in a tensorflow placeholder
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None],name = 'input')
    targets = tf.placeholder(tf.int32, [None,None],name = 'target')
    lr = tf.placeholder(tf.float32,name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') #controls dropout rate
    return inputs,targets,lr,keep_prob


# PreProcessing the targets
# RNN accepts the target answers only as batches. We need to choose appropriate batch size
# Each answer should start with a <SOS>

def preprocess_targets(targets, word2int , batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size,-1], [1,1])
    preprocessed_target = tf.concat([left_side,right_side],axis = 1)
    return preprocessed_target

#Creating the Encoder RNN 
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    #Bidirectional Dynamic RNN returns 2 outputs. The tuple containing forward and backward RNN's and a tuple containing forward and backward states
    #We need only the 2nd element which is extracted by '_,'
    return encoder_state

#Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function , keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                 attention_option = 'bahdanau',
                                                                                                                                 num_units = decoder_cell.output_size())
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state = encoder_state[0],
                                                                              attention_keys = attention_keys,
                                                                              attention_values = attention_values,
                                                                              attention_score_function = attention_score_function,
                                                                              attention_construct_function = attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output,decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                             training_decoder_function,
                                                                                                             decoder_embedded_input,
                                                                                                             sequence_length,
                                                                                                             scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob) #used to prevent overfitting and improve accuracy
    return output_function(decoder_output_dropout)

# Decoding the test/Validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function , keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                 attention_option = 'bahdanau',
                                                                                                                                 num_units = decoder_cell.output_size())
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions,decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                               test_decoder_function,
                                                                                                               scope = decoding_scope)
    return test_predictions

#Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weight = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weight,
                                                                      biases_initializer = biases)
        #Creates output function for fully connected layer after RNN
        training_prediction = decode_training_set(encoder_state,
                                                  decoder_cell,
                                                  decoder_embedded_input,
                                                  sequence_length,
                                                  decoding_scope,
                                                  output_function,
                                                  keep_prob,
                                                  batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length-1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    
    return training_prediction,test_predictions
    

#Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_normal_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions,test_predictions = decoder_rnn(decoder_embedded_input,
                                                        decoder_embeddings_matrix,
                                                        encoder_state,
                                                        questions_num_words,
                                                        sequence_length,
                                                        rnn_size,
                                                        num_layers,
                                                        questionswords2int,
                                                        keep_prob,
                                                        batch_size)
    return training_predictions,test_predictions
    
    
##### PART-3 TRAINING THE SEQ2SEQ MODEL ###########
    
# Setting the Hyperparameters
epochs = 100 #50
batch_size = 64 #128
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
key_probability = 0.5 #Probability of dropout of a neuron

#Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading the model inputs
inputs, targets, lr, key_prob = model_inputs()

#Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

#Getting the shape of the input tensor
input_shape = tf.shape(inputs)

#Getting the training and test predictions
training_predictions,test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets,
                                                      key_prob,
                                                      batch_size,
                                                      sequence_length,
                                                      len(answerswords2int),
                                                      len(questionswords2int),
                                                      encoding_embedding_size,
                                                      decoding_embedding_size,
                                                      rnn_size,
                                                      num_layers,
                                                      questionswords2int)

#Setting up the Loss error, optimzer, gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    #Lets cap these gradients
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0),grad_variable)for grad_tensor,grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


#Padding the sequences with the <PAD> token
# Question : [ 'Who', 'are', 'you' ]
# Answer : [ <SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequences) for sequences in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences

#Splitting data into batches of questions and answers
def split_into_batches(questions,answers,batch_size):
    for batch_index in range(0, len(questions) // batch_size):  # // denotes an integer result after division
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch,questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch,answerswords2int))
        yield padded_questions_in_batch,padded_answers_in_batch

# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]


#Training
batch_index_check_training_loss = 100 #Check training loss every 100 batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1 #Check loss in the middle of the epoch
total_training_loss_error = 0 #used to c0mpute the training losses on 100 batches
list_validation_loss_error = [] #all the losses are put in this list
early_stopping_check = 0 # each time there is no improvement of validation loss, it is incremented
early_stopping_stop = 1000 #Increment till 1000

checkpoint = "chatbot_weights.ckpt" #Save the weights which we can load when we want to talk to our trained chatbot
session.run(tf.global_variables_initializer())
for epoch in range(1,epochs + 1):
    for batch_index ,(padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions,training_answers,batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping,loss_error],{inputs: padded_questions_in_batch,targets: padded_answers_in_batch,lr: learning_rate,sequence_length: padded_answers_in_batch.shape[1],keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{} , Training Loss Error: {:>6.3f}, Training Time on 100 batches: {:d} seconds'.format(epoch,epochs,batch_index,len(training_questions) // batch_index , total_training_loss_error / batch_index_check_training_loss , int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation ,(padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers_answers,batch_size)):
                starting_time = time.time()
                batch_validation_loss_error = session.run(loss_error,{inputs: padded_questions_in_batch,targets: padded_answers_in_batch,lr: learning_rate,sequence_length: padded_answers_in_batch.shape[1],keep_prob: 1})
                total_validation_loss_error_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions)) / batch_size
            print('Validation Loss Error: {:>6.3f}, Batch Validation time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            
            
            
            













    




