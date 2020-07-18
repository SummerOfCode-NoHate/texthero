"""
Visualize insights and statistics of a text-based Pandas DataFrame.
"""

import pandas as pd
import plotly.express as px

from wordcloud import WordCloud

from texthero import preprocessing
from texthero._helper import handle_nans
import string

from matplotlib.colors import LinearSegmentedColormap as lsg
import matplotlib.pyplot as plt

from collections import Counter


@handle_nans(input_only=True)
def scatterplot(
    df: pd.DataFrame,
    col: str,
    color: str = None,
    hover_name: str = None,
    hover_data: [] = None,
    title="",
    return_figure=False,
):
    """
    Show scatterplot of DataFrame column using python plotly scatter.


    Parameters
    ----------
    df: DataFrame with a column to be visualized.
    col: str
        The name of the column of the DataFrame to use for x and y (and z) axis.
    color: str, default to None.
        Name of the column to use for coloring (rows with same value get same color).
    title: str, default to "".
        Title of the plot.
    return_figure: optional, default to False.
        Function returns the figure if set to True.
    hover_data: List[str], default to [].
        List of column names to supply data when hovering over a point.
    hover_name: str, default to None
        Name of the column to supply title of data when hovering over a point.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> doc1 = "Football, Sports, Soccer"
    >>> doc2 = "music, violin, orchestra"
    >>> doc3 = "football, fun, sports"
    >>> doc4 = "music, fun, guitar"
    >>> df = pd.DataFrame([doc1, doc2, doc3, doc4], columns=["texts"])
    >>> df["texts"] = hero.clean(df["texts"])
    >>> df["texts"] = hero.tokenize(df["texts"])
    >>> df["tfidf"] = hero.tfidf(df["texts"])
    >>> df["topics"] = hero.kmeans(df["tfidf"], n_clusters=2)
    >>> df["pca"] = hero.pca(df["tfidf"], n_components=3)
    >>> hero.scatterplot(df, col="pca", color="topics", hover_name="texts")
    """

    x = df[col].apply(lambda x: x[0])
    y = df[col].apply(lambda x: x[1])

    if len(df[col][0]) == 3:
        z = df[col].apply(lambda x: x[2])
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, color=color, hover_data=hover_data, title=title, hover_name=hover_name
        )
    else:
        fig = px.scatter(
            df, x=x, y=y, color=color, hover_data=hover_data, title=title, hover_name=hover_name
        )

    # fig.show(config={'displayModeBar': False})
    fig.show()

    if return_figure:
        return fig


"""
Wordcloud
"""


@handle_nans(input_only=True)
def wordcloud(
    s: pd.Series,
    font_path: str = None,
    width: int = 400,
    height: int = 200,
    max_words=200,
    mask=None,
    contour_width=0,
    contour_color="PAPAYAWHIP",
    min_font_size=4,
    background_color="PAPAYAWHIP",
    max_font_size=None,
    relative_scaling="auto",
    colormap=None,
    return_figure=False,
):
    """
    Plot wordcloud image using WordCloud from word_cloud package.

    Most of the arguments are very similar if not equal to the mother function. In constrast, all words are taken into account when computing the wordcloud, inclusive stopwords. They can be easily removed with preprocessing.remove_stopwords.

    Word are compute using generate_from_frequencies.

    Parameters
    ----------
    s : pd.Series
    font_path : str
        Font path to the font that will be used (OTF or TTF). Defaults to DroidSansMono path on a Linux machine. If you are on another OS or don't have this font, you need to adjust this path.
    width : int
        Width of the canvas.
    height : int
        Height of the canvas.
    max_words : number (default=200)
        The maximum number of words.
    mask : nd-array or None (default=None)
        When set, gives a binary mask on where to draw words. When set, width and height will be ignored and the shape of mask will be used instead. All white (#FF or #FFFFFF) entries will be considerd "masked out" while other entries will be free to draw on.
    contour_width: float (default=0)
        If mask is not None and contour_width > 0, draw the mask contour.
    contour_color: color value (default="PAPAYAWHIP")
        Mask contour color.
    min_font_size : int (default=4)
        Smallest font size to use. Will stop when there is no more room in this size.
    background_color : color value (default="PAPAYAWHIP")
        Background color for the word cloud image.
    max_font_size : int or None (default=None)
        Maximum font size for the largest word. If None, height of the image is used.
    relative_scaling : float (default='auto')
        Importance of relative word frequencies for font-size.  With
        relative_scaling=0, only word-ranks are considered.  With
        relative_scaling=1, a word that is twice as frequent will have twice
        the size.  If you want to consider the word frequencies and not only
        their rank, relative_scaling around .5 often looks good.
        If 'auto' it will be set to 0.5 unless repeat is true, in which
        case it will be set to 0.
    colormap : string or matplotlib colormap, default="viridis"
        Matplotlib colormap to randomly draw colors from for each word.
    """
    text = s.str.cat(sep=" ")

    if colormap is None:

        # Custom palette.
        # TODO move it under tools.
        corn = (255.0 / 256, 242.0 / 256, 117.0 / 256)
        mango_tango = (255.0 / 256, 140.0 / 256, 66.0 / 256)
        crayola = (63.0 / 256, 136.0 / 256, 197.0 / 256)
        crimson = (215.0 / 256, 38.0 / 256, 61.0 / 256)
        oxford_blue = (2.0 / 256, 24.0 / 256, 43.0 / 256)

        texthero_cm = lsg.from_list(
            "texthero", [corn, mango_tango, crayola, crimson, oxford_blue]
        )

        colormap = texthero_cm

    words = s.str.cat(sep=" ").split()

    wordcloud = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        max_words=max_words,
        mask=mask,
        contour_width=contour_width,
        contour_color=contour_color,
        min_font_size=min_font_size,
        background_color=background_color,
        max_font_size=max_font_size,
        relative_scaling=relative_scaling,
        colormap=colormap,
        # stopwords=[],  # TODO. Will use generate from frequencies.
        # normalize_plurals=False,  # TODO.
    ).generate_from_frequencies(dict(Counter(words)))

    # fig = px.imshow(wordcloud)
    # fig.show()

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    if return_figure:
        return fig


@handle_nans(input_only=True)
def top_words(s: pd.Series, normalize=False) -> pd.Series:
    r"""
    Return a pandas series with index the top words and as value the count.

    Tokenization: split by space and remove all punctuations that are not between characters.
    
    Parameters
    ----------
    normalize : optional, default to False.
        When set to true, return normalized values.

    Examples
    --------
    >>> s = pd.Series("one two two three three three")
    >>> hero.top_words(s)
    three    3
    two      2
    one      1
    dtype: int64

    """

    # Replace all punctuation that are NOT in-between chacarters
    # This means, they have either a non word-bounding \B, are at the start ^, or at the end $
    # As re.sub replace all and not just the matching group, add matching parenthesis to the character
    # to keep during replacement.

    # TODO replace it with tokenizer.

    pattern = (
        rf"((\w)[{string.punctuation}](?:\B|$)|(?:^|\B)[{string.punctuation}](\w))"
    )

    return (
        s.str.replace(
            pattern, r"\2 \3"
        )  # \2 and \3 permits to keep the character around the punctuation.
        .str.split()  # now split by space
        .explode()  # one word for each line
        .value_counts(normalize=normalize)
    )
