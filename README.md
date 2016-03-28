# Char level rnn shakespeare generator
Character level rnn generator written from scratch in theano. Writes better drama than me at least.

## Running:
Run python rnn.py. Datasets are included

## Sample output from gru:
```
CLAGEECE:

QAWARE EVWARD IV:

LIDY EDE:
The, then in thy pronce wints do suck you thene.

LADY GREY:

TARWE ED:
Thou lond; live tess thou mey, Cobely and we whelds it grecon?

LADY GREY:
Ewar you be you queather: what, hus; loth a tabk;
But it you in thy clought, much queaty,
To as me that my belquring further;
If mide the gracow dokn on yhath coresion:

DULEYE VERWARD I VAR:
Kefy then spreak I lenere my miness.

LADY GREY:
To no maty souls HEnds it it star:
Lith no why, her comenit
```

## Todo:
* Implement Gated Recurrent Unit/LSTM and see how good of a text it generates
* Implement adagrad adpative learning algorithm
* Add plotting code
* Add visualization code for theano network

## Done:
* Basic working rnn
* Text output generation along with probabilities
