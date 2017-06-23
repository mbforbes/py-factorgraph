# -*- coding: utf-8 -*-

"""
End-to-end tests of factor graph construction and loopy belief propagation.

author: mbforbes
"""


# Imports
# -----------------------------------------------------------------------------

# 3rd party
import numpy as np

# Local
from .context import factorgraph as fg


# Helpers
# -----------------------------------------------------------------------------

def compare_marginals_to_ref(g, ref):
    """
    Runs LBP on a graph, extracts marginals, and asserts that the results are
    equal (close) to the reference values provided.

    Args:
        g (fg.Graph)
        ref ({str: [float]})
    """
    # run lbp
    iters, converged = g.lbp(normalize=True)
    assert converged, 'LBP did not converge!'

    # get marginals and stringify (uses names)
    marginals = {str(rv): vals for rv, vals in g.rv_marginals(normalize=True)}

    # check all values in reference match those in the computed marginals
    for var_name, values in ref.iteritems():
        for i in range(len(values)):
            assert np.isclose(values[i], marginals[var_name][i])

    # to ensure extra values aren't prodcued, check the reverse: that all
    # values in computed marginals match those in the reference
    for var_name, values in marginals.iteritems():
        for i in range(len(values)):
            assert np.isclose(values[i], marginals[var_name][i])


# Tests
# -----------------------------------------------------------------------------

def test_pyfac_toygraph():
    """
    ToyGraph test from pyfac
    (https://github.com/rdlester/pyfac/blob/master/graphTests.py).

    Small note: this test uses explicit construction (make `RV`s explicitly,
    then make `Factor`s explicitly, then create graph and add them to it).
    """
    # rvs
    a = fg.RV('a', 3)
    b = fg.RV('b', 2)

    # facs
    f_b = fg.Factor([b])
    f_b.set_potential(np.array([0.3, 0.7]))
    f_ab = fg.Factor([a, b])
    f_ab.set_potential(np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]]))

    # make graph
    g = fg.Graph()
    g.add_rv(a)
    g.add_rv(b)
    g.add_factor(f_b)
    g.add_factor(f_ab)

    # quick sanity check: make sure a couple joints are correct:
    assert np.isclose(g.joint({'a': 0, 'b': 0}), 0.06)
    assert np.isclose(g.joint({'a': 2, 'b': 1}), 0.63)

    # reference comparison: ground truth vals (thanks pyfac!)
    ref = {
        'a': [
            0.34065934,
            0.2967033,
            0.36263736,
        ], 'b': [
            0.11538462,
            0.88461538,
        ],
    }

    # for this size of a graph, we can try all the assignments by hand to
    # verify brute force is working correctly.
    best_assignment, best_score = None, 0.0
    for a_val in [0, 1, 2]:
        for b_val in [0, 1]:
            assignment = {'a': a_val, 'b': b_val}
            score = g.joint({'a': a_val, 'b': b_val})
            if score > best_score:
                best_assignment = assignment.copy()
                best_score = score
    bf_assignment, bf_score = g.bf_best_joint()
    assert bf_score == best_score, 'Brute force got wrong assignment score'
    assert set(bf_assignment.items()) == set(best_assignment.items())

    # heavy lifting (lbp, marginals, reference comparison)
    compare_marginals_to_ref(g, ref)


def test_pyfac_testgraph():
    """
    TestGraph test from pyfac
    (https://github.com/rdlester/pyfac/blob/master/graphTests.py).

    Small note: this test uses implicit construction (make `RV`s and `Factor`s
    by calling graph.rv(...) and graph.factor(...)).
    """
    # start an empty graph
    g = fg.Graph()

    # rvs
    g.rv('a', 2)
    g.rv('b', 3)
    g.rv('c', 4)
    g.rv('d', 5)

    # factors
    g.factor(['a'], potential=np.array([0.3, 0.7]))
    g.factor(['b', 'a'], potential=np.array([
            [0.2, 0.8],
            [0.4, 0.6],
            [0.1, 0.9],
    ]))
    g.factor(['d', 'c', 'a'], potential=np.array([
        [
            [3., 1.],
            [1.2, 0.4],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        [
            [11., 9.],
            [8.8, 9.4],
            [6.4, 0.1],
            [8.8, 9.4],
        ],
        [
            [3., 2.],
            [2., 2.],
            [2., 2.],
            [3., 2.],
        ],
        [
            [0.3, 0.7],
            [0.44, 0.56],
            [0.37, 0.63],
            [0.44, 0.56],
        ],
        [
            [0.2, 0.1],
            [0.64, 0.44],
            [0.37, 0.63],
            [0.2, 0.1],
        ],
    ]))

    # quick sanity check: make sure a couple joints are correct:
    assert np.isclose(g.joint({'a': 0, 'b': 0, 'c': 0, 'd': 0}), 0.18)
    assert np.isclose(g.joint({'a': 1, 'b': 2, 'c': 3, 'd': 4}), 0.063)

    # ground truth (thanks pyfac!)
    ref = {
        'a': [
            0.13755539,
            0.86244461,
        ], 'b': [
            0.33928227,
            0.30358863,
            0.3571291,
        ], 'c': [
            0.30378128,
            0.29216947,
            0.11007584,
            0.29397341,
        ], 'd': [
            0.076011,
            0.65388724,
            0.18740039,
            0.05341787,
            0.0292835,
        ],
    }

    # heavy lifting (lbp, marginals, reference comparison)
    compare_marginals_to_ref(g, ref)
