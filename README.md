# The Hewlett Foundation: Short Answer Scoring
The notebook contains my solution to the given problem.<br>
https://www.kaggle.com/c/asap-sas<br>
The problem is to Score Student-Answers in a Scale of 0-3.<br>
The dataset contains Answers from 10 Different Categories/Topics.<br>
For full description visit the link.<br>

# Instruction
You can find the data from the link above.<br>
The Dataset is provided in a TSV File. It contains the following Columns.<br>

id----->       Essay id or answer id<br>
EssaySet-----> 1-10, an id for each set of essays.<br>
Score 1------> The human rater's score for the answer. This is the final score for the answer and the score that you are                    trying to predict.<br>
Score2-------> A second human rater's score for the answer. This is provided as a measure of reliability, but had no bearing                on the score the essay received.<br>
EssayText----> The ascii text of a student's response.<br>


# Dependencies
ipython<br>
Keras<br>
Theano<br>
Cuda(For GPU processing)<br>
Numpy<br>
Pandas<br>

# Preprocessing
We cannot directly pass raw-text to our network for training. So we have to convert the input data into a format suitable for our network.<br>

1. Raw Data Encoding
   Format the text samples and labels into tensors that can be fed into a neural network

2. convert class vector to binary class matrix, for use with categorical_crossentropy.
   Or in simple words to convert numbers into ONE-HOT Vector for MultiClass classification


# Model: 
This is a simple flowchart diagram of my model.

    Text(Binary Class Matrix)
     |
     Embedding
     |
     LSTM(100)
     |
     LSTM(150)      Features (Essay_Set_Number, Word_counts, Sentence_counts)
       \             /
        \           /
         \         /
            MERGE
              |
         Dropout 20%
              |
     Dense(200) with L2 regularization(reg_parma=0.01)
            |
        ELU activation
            |
     Score with softmax activation (dim=4)
     

1. The text input is an integer matrix. 
2. The input text(matrix) is fed to Embedding layer which outputs a 3-d matrix acceptable by a LSTM layer
3. 2 consecutive LSTMs: I have tried various combinations of LSTMs and this was the configuration which extracted optimal        information from the text.
   The reason of using two consecutive LSTM is to create a more complex feature representation from the input.

4. Features (dim=3): essay_set, word_counts, sent_counts
5. The features and the the output of the LSTM are then merged together into a single vector.
6. Then I introduces Dropout in between. I tried various dropout fractions and found 20% to optimum dropout fraction (keeping
   in mind that the data is less and less data would be a barrier for the model’s capacity to learn redundant representation    which is the result of dropout).

7. Single fully connected hidden layer of 200 units with L2 regularisation is used to extract features from the merged data   
   and also prevent the model from overfitting.
8. I have used advanced activation function called ELU provided by Keras which gave a better performance over conventional 
   relu and sigmoid activations.
9. The next layer is the output layer of dim=4 where each element in 4 dimensional vector corresponds to a score.
10. **Optimizer**: Adam with learning rate 0.001 and batch size = 128
11. **Loss function**: Categorical cross entropy

# Results:
Validation accuracy: 86.47%<br>
Test accuracy: 86.61%
