# Shakespeare_script_gen


## Result

We use the following parameters:

```python
state_size = 100,
num_classes = vocab_size,
batch_size = 32,
num_steps = 80,
num_layers = 3,
learning_rate = 1e-4:
epochs = 50
```

The generated Shakespeare-like script is 
```
He'ns show thing.

KING RIKH:
The with me, thou have seed to servan morne,
Which was and such one wan should with.

KING EDWARD IV:
Nay they forth that wonging hatity best,
The common'd the prains on my with though a dost
The comsant to the concend at are arm to these sen
To most the count that, and hold.

SICINIUS:
The water that I has so his worth of my,
That where to brought the dosts with son,
Where to him serven in the cames on and to to
To made thou his work and betture
And shall. In this that strants whest
I will thy fried, is true, that seal bart
The poor of the weader, then thee artise my seen
For the peitury she shall and them.

LIORAN:
So, must be suchs on true were was stand
The plaserencing the with out was the consel
Ard it son
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

We can see that the similarity between the generated text and the original one is almost to the level words. Most of the words are spelled correctly, which indicates the effectiveness of pattern learning given that this is a charater-level sequence generation.  A comparison can be drawn with the known Karpathy's Shakespeare results (http://karpathy.github.io/2015/05/21/rnn-effectiveness/#shakespeare). There the text is much more readable. The difference is that the hidden unit size (ie state_size) he uses is 512, while mine is 100 for the sake of saving time consumption. Also he uses a much larger corpus, while mine is a tiny version of the Shakespeare script. So this model is expcted to have better performance with more parameters and larger data set.
