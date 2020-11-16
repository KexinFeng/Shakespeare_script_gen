## Shakespeare_script_gen

# Introduction
In supervised machine learing tasks, what is often seen is to properly fit a data set and use the trained model to predict for the label. In this project, instead of making prediction for label, we explore generating new data set that is similar to the training data set, which shapes up to be more challenging.

In this project, we use gated RNN to build a character-level language model to generate character sequences. Using the corpus of a Shakespeare script text, the total number of characters is 65. In the training stage, each character is input into the model and its immediate preceding character is its label. In the prediction, the output is the probability of each of the 65 characters, then the most likely character will be taken as the prediction. So in essence this is a classification problem.

The loss function to delineate the proximity between two sequences is perplexity loss function. It can be shown that perplexity is equivalent to crossentropy summated over the sequence, so in the code, we still use crossentropy as loss function. 

The basic structure of RNN model:
<p align="center">
    <img src="./media/BasicRNNLabeled.png"><br/>
    <em>Basic structure of RNN.</em>
</p>





