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
from .context import factorgraph


# Tests
# -----------------------------------------------------------------------------

def test_pyfac_testgraph():
    """
    TestGraph test from pyfac
    (https://github.com/rdlester/pyfac/blob/master/graphTests.py).
    """

    # start an empty graph
    g = factorgraph.Graph()

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

    # Assert equal to pyfac reference values
    iters, converged = g.lbp(normalize=True)
    assert converged, 'LBP did not converge!'

    # get marginals and stringify (uses names)
    marginals = {str(rv): vals for rv, vals in g.rv_marginals(normalize=True)}

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

    # check all values in reference match those in the computed marginals
    for var_name, values in ref.iteritems():
        for i in range(len(values)):
            assert np.isclose(values[i], marginals[var_name][i])

    # to ensure extra values aren't prodcued, check the reverse: that all
    # values in computed marginals match those in the reference
    for var_name, values in marginals.iteritems():
        for i in range(len(values)):
            assert np.isclose(values[i], marginals[var_name][i])
