## Shakespeare_script_gen

# Introduction
In supervised machine learing tasks, what is often seen is to properly fit a data set and use the trained model to predict for the label. In this project, instead of making prediction for label, we explore generating new data set that is similar to the training data set, which shapes up to be more challenging.

In this project, we use gated RNN to build a character-level language model to generate character sequences. Using the corpus of a Shakespeare script text, the total number of characters is 65. In the training stage, each character is input into the model and its immediate preceding character is its label. In the prediction, the output is the probability of each of the 65 characters, then the most likely character will be taken as the prediction. So in essence this is a classification problem.

The loss function to delineate the proximity between two sequences is perplexity loss function. It can be shown that perplexity is equivalent to crossentropy summated over the sequence, so in the code, we still use crossentropy as loss function. 

The basic structure of gated RNN model:
<p align="center">
    <img src="./media/BasicRNNLabeled.png"><br/>
    <em>Basic structure of RNN.</em>
</p>

Basic structure of gated RNN model:
<p align="center">
    <img src="./media/LSTM3-chain.png"><br/>
    <em>Basic structure of LSTM.</em>
</p>

# Usage

`shakespeare_gen.py` is the main function, which builds the graph and train the model. The parameters are saved in dir `./saves/`. 

`read_saved_model.py` reads the saved trained parameters and generates new Shakespeare style script.

The other two files `reader.py` and `utils.py` are utility functions.

# Model selection

We have built three models: GRU, LSTM, LN_LSTM (LSTM with layer normalization). The layer normalization is based on Ref . It is similar to batch normalization in terms of reducing the covariate, but is normalized over the feature space instead of across the batches. It is added before the non-linearity.





# Result

We use the following parameters:

```python
state_size = 100,
num_classes = vocab_size,
batch_size = 32,
num_steps = 80,
num_layers = 3,
learning_rate = 1e-4:
epochs = 20
```

The generated Shakespeare-like script is 
```
HIRI OF ICHERD:
Hell bees should but ance thee,
And shestall that hean heave houng shourded one
thy sollook see that a more tay see how,
There a sechich a masters olly the cournt on hear the chear,
And menting triendy and my sellond, then the best the
That saless the compined his madester.

KING LARENLO:
No see, are songess oun horghand hast on arters.

GLOUMEREN RIO:
He sall'd that thing to be call, a sourth, trick tood.

LEONTER:
That so dost, there hast a wime of a till a mant.

KING II:
A mister, were with his will strang thas witio well thit
And the woor that wish a that he than, hath his are.
With, sill thou tall stele thow sto seed a sucented,
I wisth mest in the will mastied that,
For to the seed tay warth have trenent
An
```


This is an example of the training Shakespeare script:
```
HASTINGS:
So thrive I, as I truly swear the like!

KING EDWARD IV:
Take heed you dally not before your king;
Lest he that is the supreme King of kings
Confound your hidden falsehood, and award
Either of you to be the other's end.

HASTINGS:
So prosper I, as I swear perfect love!

RIVERS:
And I, as I love Hastings with my heart!

KING EDWARD IV:
Madam, yourself are not exempt in this,
Nor your son Dorset, Buckingham, nor you;
You have been factious one against the other,
Wife, love Lord Hastings, let him kiss your hand;
And what you do, do it unfeignedly.

QUEEN ELIZABETH:
Here, Hastings; I will never more remember
Our former hatred, so thrive I and mine!

KING EDWARD IV:
Dorset, embrace him; Hastings, love lord marquess.

```

We can see that the similarity between the generated text and the original one is almost to the level words. But the generated text still have too many misspelled words. A comparison can be drawn with the known Karpathy's Shakespeare results (http://karpathy.github.io/2015/05/21/rnn-effectiveness/#shakespeare). There the text is much more readable. The difference is that the hidden unit size (ie state_size) he uses is 512, while mine is 100 for the sake of saving time consumption. Also he uses a much larger corpus, while mine is a tiny version of the Shakespeare script. 