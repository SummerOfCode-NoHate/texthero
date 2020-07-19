import pandas as pd
import numpy as np
from texthero import representation
from texthero import preprocessing

from . import PandasTestCase

import doctest
import unittest
import string
import math
import warnings

"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(representation))
    return tests


class TestRepresentation(PandasTestCase):
    """
    Term Frequency.
    """

    def test_term_frequency_single_document(self):
        s = pd.Series("a b c c")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1, 2]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.term_frequency(s, return_flat_series=True), s_true)

    def test_term_frequency_multiple_documents(self):
        s = pd.Series(["doc_one", "doc_two"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1, 1, 0], [1, 1, 0, 1]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.term_frequency(s, return_flat_series=True), s_true)

    def test_term_frequency_not_lowercase(self):
        s = pd.Series(["one ONE"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.term_frequency(s, return_flat_series=True), s_true)

    def test_term_frequency_punctuation_are_kept(self):
        s = pd.Series(["one !"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.term_frequency(s, return_flat_series=True), s_true)

    def test_term_frequency_not_tokenized_yet(self):
        s = pd.Series("a b c c")
        s_true = pd.Series([[1, 1, 2]])
        s_true.rename_axis("document", inplace=True)


        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(representation.term_frequency(s, return_flat_series=True), s_true)

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.term_frequency(s, return_flat_series=True)

    """
    TF-IDF
    """

    def test_tfidf_formula(self):
        s = pd.Series(["Hi Bye", "Test Bye Bye"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series(
            [
                [
                    1.0 * (math.log(3 / 3) + 1),
                    1.0 * (math.log(3 / 2) + 1),
                    0.0 * (math.log(3 / 2) + 1),
                ],
                [
                    2.0 * (math.log(3 / 3) + 1),
                    0.0 * (math.log(3 / 2) + 1),
                    1.0 * (math.log(3 / 2) + 1),
                ],
            ]
        )
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

    def test_tfidf_single_document(self):
        s = pd.Series("a", index=["yo"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1]], index=["yo"])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

    def test_tfidf_not_tokenized_yet(self):
        s = pd.Series("a")
        s_true = pd.Series([[1]])
        s_true.rename_axis("document", inplace=True)

        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.tfidf(s, return_flat_series=True)

    def test_tfidf_single_not_lowercase(self):
        s = pd.Series("ONE one")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1.0, 1.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

    def test_tfidf_max_features(self):
        s = pd.Series("one one two")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[2.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, max_features=1, return_flat_series=True), s_true)

    def test_tfidf_min_df(self):
        s = pd.Series([["one"], ["one", "two"]])
        s_true = pd.Series([[1.0], [1.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, min_df=2, return_flat_series=True), s_true)

    def test_tfidf_max_df(self):
        s = pd.Series([["one"], ["one", "two"]])
        s_true = pd.Series([[0.0], [1.4054651081081644]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, max_df=1, return_flat_series=True), s_true)


    """
    Representation series testing
    """
    """
    Term Frequency.
    """

    def test_term_frequency_single_document_representation_series(self):
        s = pd.Series([list("abbccc")])

        idx = pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c")], names=("document", "word")
        )

        s_true = pd.Series([1, 2, 3], index=idx, dtype="int").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_multiple_documents_representation_series(self):

        s = pd.Series([["doc_one"], ["doc_two"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "doc_one"), (1, "doc_two")], names=("document", "word")
        )

        s_true = pd.Series([1, 1], index=idx, dtype="int").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_not_lowercase_representation_series(self):

        s = pd.Series([["A"], ["a"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "A"), (1, "a")], names=("document", "word")
        )

        s_true = pd.Series([1, 1], index=idx, dtype="int").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_punctuation_are_kept_representation_series(self):

        s = pd.Series([["number", "one", "!"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "!"), (0, "number"), (0, "one")], names=("document", "word")
        )

        s_true = pd.Series([1, 1, 1], index=idx, dtype="int").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_raise_when_not_tokenized_representation_series(self):
        s = pd.Series("not tokenized")
        with self.assertRaisesRegex(ValueError, r"tokenized"):
            representation.term_frequency(s)

    """
    TF-IDF
    """

    def test_tfidf_simple_representation_series(self):
        s = pd.Series([["a"]])

        idx = pd.MultiIndex.from_tuples([(0, "a")], names=("document", "word"))
        s_true = pd.Series([1.0], index=idx).astype("Sparse")
        self.assertEqual(representation.tfidf(s), s_true)

    def test_idf_single_not_lowercase_representation_series(self):
        tfidf_single_smooth = 0.7071067811865475  # TODO

        s = pd.Series([list("Aa")])

        idx = pd.MultiIndex.from_tuples(
            [(0, "A"), (0, "a")], names=("document", "word")
        )

        s_true = pd.Series(
            [tfidf_single_smooth, tfidf_single_smooth], index=idx
        ).astype("Sparse")

        self.assertEqual(representation.tfidf(s), s_true)

    def test_idf_single_different_index_representation_series(self):
        # compute s_true
        idx = pd.MultiIndex.from_tuples(
            [(10, "a"), (11, "b")], names=("document", "word")
        )
        s_true = pd.Series([1.0, 1.0], index=idx).astype("Sparse")

        s = pd.Series([["a"], ["b"]], index=[10, 11])
        self.assertEqual(representation.tfidf(s), s_true)

    def test_idf_raise_when_not_tokenized_representation_series(self):
        s = pd.Series("not tokenized")
        with self.assertRaisesRegex(ValueError, r"tokenized"):
            representation.tfidf(s)

    """
    PCA
    """

    def test_pca_tf_simple_representation_series(self):
        idx = pd.MultiIndex.from_tuples(
            [(0, "a"), (1, "b"), (2, "c")], names=("document", "word")
        )
        s = pd.Series([1, 1, 1], index=idx)
        s = representation.pca(s)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        s_true = pca.fit_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        s_true = pd.Series(s_true.tolist())

        self.assertEqual(s, s_true)

    # TODO check raise warning

    """
    NMF
    """

    def test_nmf_tf_simple_representation_series(self):
        idx = pd.MultiIndex.from_tuples(
            [(0, "a"), (1, "b"), (2, "c")], names=("document", "word")
        )
        s = pd.Series([1, 1, 1], index=idx)
        s = representation.nmf(s, random_state_arg=1)

        from sklearn.decomposition import NMF

        nmf = NMF(n_components=2, random_state=1)
        s_true = nmf.fit_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        s_true = pd.Series(s_true.tolist())

        self.assertEqual(s, s_true)

    """
    TruncatedSVD
    """

    '''
    def test_nmf_tf_simple_representation_series(self):
        idx = pd.MultiIndex.from_tuples(
            [(0, "a"), (1, "b"), (2, "c")], names=("document", "word")
        )
        s = pd.Series([1, 1, 1], index=idx)
        s = representation.truncated_svd(s, random_state=1)

        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=2, random_state=1)
        s_true = svd.fit_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        s_true = pd.Series(s_true.tolist())

        self.assertEqual(s, s_true)
    '''