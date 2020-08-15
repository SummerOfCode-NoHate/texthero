"""
New Hero VectorDF
=================

## Overview

When calculating any kind of mathematical representation
of our documents (e.g. through `embed, tfidf, pca, ...`)
we need to store these in a Pandas DataFrame / Series.

Before the new VectorDF, we have done this by simply storing
lists in cells, e.g. like this:

```
                        nmf
0                       [0.0, 0.0, 0.548022198777919]
1                       [1.6801770114633772, 0.0, 0.0]
2                       [1.678910685857782, 0.0, 0.00015405496961668698]
3                       [0.0, 2.1819904843216302, 0.0]
dtype: object

```

This has two main disadvantages:

- Performance: storing complex structures like lists is the opposite
                of what Pandas is designed to do (and where it excels
                at). Essentially, Pandas _does not understand_ what
                we're storing in the `nmf` column (which is why the
                dtype is `object`).

- User Experience: the Series above just does not look nice for
                    users; it provides no intuition as to what
                    it's about.


For the reasons above (and some other small ones), we switch to
the new VectorDF. Now, we separate every entry of the vector/list
to an own subcolumn. Example:

```
        nmf                    
       nmf1      nmf2      nmf3
0 -1.319228 -0.388168  0.483747
1 -0.080735 -2.190099  0.594459
2 -0.132196  0.786491  0.715910
3 -0.233909 -1.365364  0.156663
4 -0.207253  0.880211 -0.156841

```

This preserves the _atomicity_ in the cells that Pandas is designed
for; we can now do vectorized operations on the columns etc.; no more
`object` datatype! It's also much more intuitive for users: they
can immediately see that what they get from `hero.nmf` looks like
a Matrix (of course it does not just look like one, it _is_ a matrix - that's
why it's called non-negative _matrix_ factorization).

We believe this is a major step forward for texthero, delivering
performance enhancements and a cleaner and more intuitive UX.


## Integration into Texthero Types

- TextSeries:
    no problem, like before, see df3 above

- VectorSeries:
    no problem, will become VectorDF

- TokenSeries: 
    will try around with representation format,
    but will probably stay a list of strings for the time being

- RepresentationSeries from tfidf, count, term_frequency: 
    will try around with new format but
    probably keep as RepresentationSeries for the time being;
    check out `unstack`

- Categorical: 
    no problem, like before


## Integration into Texthero Modules

- preprocessing/nlp: 
    depends on what happens with TokenSeries

- representation:
    - tfidf, count, term_frequency: TokenSeries -> RepresentationSeries
    - embed: TokenSeries -> SubcolumnDF
    - dim. red: RepresentationSeries or SubcolumnDF -> SubcolumnDF
    - clustering: RepresentationSeries or SubcolumnDF -> SubcolumnDF


## Pandas Trouble: Integration with `df["pca"] = hero.pca(df["texts"])`

It's really important that users can seamlessly integrate texthero's function
output with their code. Let's assume a user has his documents in a DataFrame
`df["texts"]` that looks like this:

```
>>> df = pd.DataFrame(["Text of doc 1", "Text of doc 2", "Text of doc 3"], columns=["text"])
>>> df
            text
0  Text of doc 1
1  Text of doc 2
2  Text of doc 3

```

 Let's look at an example output that `hero.pca` could
return with the new type:

```
>>> hero.pca(df["texts"])
        pca          
       pca1      pca2
0  0.754675  1.868685
1 -1.861651 -0.048236
2 -0.797750  0.388400
```

(you can generate a mock output like this e.g. with
`pd.DataFrame(np.random.normal(size=(6,)).reshape((3,2)), columns=pd.MultiIndex.from_product([['pca'], ["pca1", "pca2"]]))`)

That's a DataFrame. Great! Of course, users can
just store this somewhere as e.g. `df_pca = hero.pca(df["texts"])`,
and that works great. Accessing is then also as always: to get the
pca values, they can just do `df_pca.values` and have the pca matrix
right there!

However, what we see really often is users wanting to do this:
`df["pca"] = hero.pca(df["texts"])`. This sadly does not work out
of the box. The reason is that this subcolumn type is implemented
internally through a _Multiindex in the columns_. So we have

```
>>> df.columns
Index(['text'], dtype='object')
>>> hero.pca(df["texts"]).columns
MultiIndex([('pca', 'pca1'), ('pca', 'pca2')])

```

Pandas _cannot_ automatically combine these. So what we will
do is this: Calling `df["pca"] = hero.pca(df["texts"])` is
internally this: `pd.DataFrame.__setitem__(self=df, key="pca", value=hero.pca(df["texts"]))`.
We will overwrite this method so that if _self_ is not multiindexed yet
and _value_ is multiindexed, we transform _self_ (so `df` here) to
be multiindexed and we can then easily integrate our column-multiindexed output from texthero:

As soon as `df` is multiindexed, we get the desired result through 
`df[hero.pca(df["texts"]).columns] = hero.pca(df["texts"])`. So we
only need to:

1. make df multiindexed if it isn't already
2. call df[hero.pca(df["texts"]).columns] = hero.pca(df["texts"]) internally


Aren't we destroying Pandas?

    - we don't change any pandas functionality as currently calling
      `__setitem__` with a Multiindexed value is just not possible, so
      our changes to Pandas do not break any Pandas functionality for
      the users. We're only _expanding_ the functinoality

    - after multiindexing, users can still access their
      "normal" columns like before; e.g. `df["texts"]` will
      behave the same way as before even though it is now internally
      multiindexed as `MultiIndex([('pca', 'pca1'), ('pca', 'pca2'), ('texts', '')])`.


Here's the code to do this (for examples and guidelines on how
to integrate the new VectorDF into functions see the bottom of
this file):

"""
import pandas as pd
import numpy as np

from pandas.core import common as com
from pandas.core.indexing import convert_to_index_sliceable


from texthero.representation import *


# Store the original __setitem__ function as _original__setitem__
_pd_original__setitem__ = pd.DataFrame.__setitem__
pd.DataFrame._original__setitem__ = _pd_original__setitem__


# Define a new __setitem__ function that will replace pd.DataFrame.__setitem__
def _hero__setitem__(self, key, value):
    '''
    Called when doing self["key"] = value.
    E.g. df["pca"] = hero.pca(df["texts"]) is internally doing
    pd.DataFrame.__setitem__(self=df, key="pca", value=hero.pca(df["texts"]).

    So self is df, key is the new column's name, value is
    what we want to put into the new column.

    What we do:

    1. If user calls __setitem__ with value being multiindexed, e.g.
       df["pca"] = hero.pca(df["texts"]),
       so __setitem__(self=df, key="pca", value=hero.pca(df["texts"])

        2. we make self multiindexed if it isn't already
            -> e.g. column "text" internally becomes multiindexed
               to ("text", "") but users do _not_ notice this.
               This is a very quick operation that does not need
               to look at the df's values, we just reassign
               self.columns

        3. we change value's columns so the first level is named `key`
            -> e.g. a user might do df["haha"] = hero.pca(df["texts"]),
               so just doing df[hero.pca(df["texts"]).columns] = hero.pca(df["texts"])
               would give him a new column that is named like texthero's output,
               e.g. "pca" instead of "haha". So we internally rename the
               value columns (e.g. [("pca", "pca1"), ("pca", "pca2")] to
               [("haha", "pca1"), ("haha", "pca2")])

        4. we do self[value.columns] = value as that's exactly the command
           that correctly integrates the multiindexed `value` into `self`

    '''


    # 1.
    if isinstance(value, pd.DataFrame) and isinstance(value.columns, pd.MultiIndex) and isinstance(key, str):

        # 2.
        if not isinstance(self.columns, pd.MultiIndex):
            self.columns = pd.MultiIndex.from_tuples([(col_name, "") for col_name in self.columns.values])

        # 3.
        value.columns = pd.MultiIndex.from_tuples([(key, subcol_name) for _, subcol_name in value.columns.values])

        # 4.
        self[value.columns] = value

    else:

       self._original__setitem__(key, value)


# Replace __setitem__ with our custom function
pd.DataFrame.__setitem__ = _hero__setitem__


"""
Examples for how to integrate this with other functions:
TODO

df = pd.DataFrame(np.random.normal(size=(6,)).reshape(
    (3, 2)), columns=pd.MultiIndex.from_product([['pca'], ["pca1", "pca2"]]))
df2 = pd.DataFrame(np.random.normal(size=(6,)).reshape(
    (3, 2)), columns=pd.MultiIndex.from_product([['pipapo'], ["nmf1", "nmf2"]]))


df["nmf"] = df2
print(df)
"""

"""
Integration of RepresentationSeries:

tfidf, count, term_frequency give us a sparse matrix.
We could make it seem for users like it is in a column in their
DataFrame but internally only store pointers to the actual sparse matrix.
They'd thus have the best of both worlds:
profit from sparseness to get the whole document-term matrix and also
keep everything in their DataFrame.

Approach 1:

    Using pd.DataFrame.sparse (so a sparse DF directly) works
    and looks great for users (they can really see the
    document-term matrix). Code to try it out:

    >>> df = pd.DataFrame(["Text one", "Text two"], columns=["text"])

    >>> df_count = data["text"].pipe(hero.count)
    >>> sparse_matrix, index, columns = data_count.sparse.to_coo()
    >>> # of course we can do the two steps above faster by changing hero.count to
    >>> # directly get the Sparse DF, this is just to showcase

    >>> multiindexed_columns = pd.MultiIndex.from_tuples([("count", col) for col in columns])
    >>> df_count_sparse = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index, multiindexed_columns)
    >>> df_count_sparse
    count        
    Text one two
    0     1   1   0
    1     1   0   1
    >>> df_count_sparse.sparse.density
    0.66666666666

    It _seems to work great_. However, doing the same thing with
    `data = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")`
    is really slow. We have identified the bottleneck in
    the "else"-case in pd.DataFrame._setitem_array which
    loops over all columns (so all words appearing in all texts)
    and puts them into the df one by one. This is of course really
    slow when dealing with thousands of columns.


Approach 2:

    
"""


data = pd.read_csv(
    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)


# data = pd.DataFrame(["Text one", "Text two"], columns=["text"])

data_count = data["text"].pipe(count, max_features=300)

sparse_matrix, index, columns = data_count.sparse.to_coo()

multiindexed_columns = pd.MultiIndex.from_tuples(
    [("count", col) for col in columns])
x = pd.DataFrame.sparse.from_spmatrix(
    sparse_matrix, index, multiindexed_columns)
print(x.sparse.density)


data["count"] = x
