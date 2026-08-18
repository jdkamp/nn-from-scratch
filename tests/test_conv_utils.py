import numpy as np
from autograd.conv_utils import unfold, fold


def test_unfold_basic():
    """One row per sliding window, read row by row."""
    expected = np.array([
        [ 0,  1,  4,  5], [ 1,  2,  5,  6], [ 2,  3,  6,  7],
        [ 4,  5,  8,  9], [ 5,  6,  9, 10], [ 6,  7, 10, 11],
        [ 8,  9, 12, 13], [ 9, 10, 13, 14], [10, 11, 14, 15],
    ], dtype=float)

    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    np.testing.assert_allclose(unfold(x, 2, 2), expected)


def test_unfold_stride2():
    """At stride=2 the windows go over the image without overlapping."""
    expected = np.array([
        [ 0,  1,  4,  5], [ 2,  3,  6,  7],
        [ 8,  9, 12, 13], [10, 11, 14, 15],
    ], dtype=float)

    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    np.testing.assert_allclose(unfold(x, 2, 2, stride=2), expected)


def test_unfold_stride_drops_pixels():
    """A stride the image size does not divide truncates: row 2 and column 2
    of a 5x5 are not processed."""
    expected = np.array([
        [ 0,  1,  5,  6], [ 3,  4,  8,  9],
        [15, 16, 20, 21], [18, 19, 23, 24],
    ], dtype=float)

    x = np.arange(25, dtype=float).reshape(1, 1, 5, 5)
    np.testing.assert_allclose(unfold(x, 2, 2, stride=3), expected)


def test_unfold_padding():
    """Padding adds a zero border, so corner windows are mostly zeros."""
    expected = np.array([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
        [0, 0, 0, 2], [0, 1, 2, 3], [1, 0, 3, 0],
        [0, 2, 0, 0], [2, 3, 0, 0], [3, 0, 0, 0],
    ], dtype=float)

    x = np.arange(4, dtype=float).reshape(1, 1, 2, 2)
    np.testing.assert_allclose(unfold(x, 2, 2, pad=1), expected)


def test_fold_counts_overlaps():
    """fold sums overlapping windows, folding ones counts how many windows each pixel belongs to."""
    expected = np.array([
        [1, 2, 2, 1],
        [2, 4, 4, 2],
        [2, 4, 4, 2],
        [1, 2, 2, 1],
    ], dtype=float)

    np.testing.assert_allclose(fold(np.ones((9, 4)), (1, 1, 4, 4), 2, 2)[0, 0], expected)


def test_fold_stride2_no_overlaps():
    """At stride 2 nothing overlaps, so every pixel is counted exactly once."""
    expected = np.ones((4, 4))

    np.testing.assert_allclose(fold(np.ones((4, 4)), (1, 1, 4, 4), 2, 2, stride=2)[0, 0], expected)


def test_fold_padding_is_cropped():
    """A 2x2 image padded by 1 gives 3x3 = 9 windows. Every real pixel sits at
    a padded position covered by exactly 4 of them, so cropping the padding
    away must leave a uniform 4.
    """
    expected = np.array([
        [4, 4],
        [4, 4],
    ], dtype=float)

    np.testing.assert_allclose(fold(np.ones((9, 4)), (1, 1, 2, 2), 2, 2, pad=1)[0, 0], expected)
