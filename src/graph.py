'''
Implementation of graph.

Author: mbforbes

Current approach (order matters):
-   (1) add RVs
-   (2) add factors to connect them
-   (3) set beliefs on factors
-   (4) run inference
-   (5) compute marginals

For some things below, we'll want to represent what's going on in mathematical
notation. Let's define some variables that we'll use throughout to help:

RV vars:
    X       the set of n random variables
    X_i     random variable i (1 <= i <= n)
    v_i     number of values that X_i can take (nonstandard but I wanted one)
    x_ij    a particular value for X_i (1 <= j <= v_i)
    x_i     a simpler (lazy) notation for x_ij (which j doesn't matter)
    x       a set of x_i for i = 1..n (imagine x_1j, x_2k, ..., x_nz)

Factor vars:
    F       the set of m factors
    f_a     factor a (1 <= a <= m) connecting a subset of X
    X_a     the subset of X (RVs) that f_a connects
    x_a     the subset of x (values for RVs) that f_a connects

Functions:
    p(x)    joint distribution for p(X = x)

Notes:
    f_a(x) = f_a(x_a)   Because f_a only touches (is only a function of) x_a,
                        it will "ignore" the other x_i in x that aren't in x_a.
                        Thus, we write f_a(x_a) for convenience to show exactly
                        what f_a operates on.
'''

# Imports
# -----------------------------------------------------------------------------

# 3rd party
import numpy as np


# Constants
# -----------------------------------------------------------------------------

# Settings

# Use this to turn all debugging on or off. Intended use: keep on when you're
# trying stuff out. Once you know stuff works, turn off for speed. Can also
# specify when creating each instance, but this global switch is provided for
# convenience.
DEBUG_DEFAULT = True


# Classes
# -----------------------------------------------------------------------------

class Graph(object):
    '''
    Graph right now has no point, really (except bookkeeping all the RVs and
    factors, assuming we remember to add them), so this might be removed or
    functionality might be stuffed in here later.
    '''

    def __init__(self, debug=DEBUG_DEFAULT):
        # add now
        self.debug = debug

        # added later
        self._rvs = {}
        self._factors = []  # TODO: Better data structure?

    def add_rv(self, rv):
        '''
        Node (RV|Factor)
        '''
        # Check RV with same name not already added.
        if self.debug:
            assert rv.name not in self._rvs
        # Add it.
        self._rvs[rv.name] = rv

    def add_factor(self, factor):
        if self.debug:
            # Check the same factor hasn't already been added.
            assert factor not in self._factors

            # Check factor connecting to exactly the same set of nodes doesn't
            # already exist. This isn't mandated by factor graphs by any means,
            # but it's a heuristic to prevent bugs; if you're adding factors
            # that connect the same set of ndoes, you're either doing something
            # weird (and can probably reformulate your graph structure to avoid
            # this duplication), or you have a bug.
            factor_rvs = sorted(factor._rvs)
            for f in self._factors:
                rvs = sorted(f._rvs)
                assert factor_rvs != rvs
        # Add it.
        self._factors += [factor]


    def joint(self, x):
        '''
        Joint is over the factors.

        For a probability, we use the normalization constant 1/Z

            p(x) = 1/Z \product_a^{1..m} f_a(x_a)

        If we don't care what the normalization is, we just write this without
        1/Z:

            p(x) = \product_a^{1..m} f_a(x_a)

        This is currently implemented without normalization. I might want to
        add it in the future. I don't know yet.

        Args:
            x ({str: str|int}) map of node names to assignments. The
                assignments can be labels or indexes
        '''
        # ensure the assignment x given is valid
        if self.debug:
            # check the length (that assignments to all RVs are provided)
            assert len(x) == len(self._rvs)

            # check that each assignment is valid (->)
            for name, label in x.iteritems():
                assert name in self._rvs
                assert self._rvs[name].has_label(label)

            # check that each RV has a valid assignment (<-)
            for name, rv in self._rvs.iteritems():
                assert name in x
                assert rv.has_label(x[name])

        # Do the actual computation.
        # NOTE: This could be sped up as all factors can be computed in
        # parallel.
        prod = 1.0
        for f in self._factors:
            prod *= f.eval(x)
        return prod

    def bf_best_joint(self):
        '''
        Brute-force algorithm to compute the best joint assignment to the
        factor graph given the current beliefs in the factors.

        This takes O(v^n) time, where
            v   is the # of possible assignments to each RV
            n   is the # of RVs

        This is bad. This function is given for debugging / proof of concept
        only.

        Returns:
            ({str: int}, float)
        '''
        return self._bf_bj_recurse({}, self._rvs.values())

    def _bf_bj_recurse(self, assigned, todo):
        '''
        Helper method for bf_best_joint.

        Args:
            assigned ({str: int})
            todo ([RV])
        '''
        # base case: just look up the current assignment
        if len(todo) == 0:
            return assigned, self.joint(assigned)

        # recursive case: pull off one RV and try all options.
        best_a, best_r = None, 0.0
        rv = todo[0]
        todo = todo[1:]
        for val in range(rv.n_opts):
            new_a = assigned.copy()
            new_a[rv.name] = val
            full_a, r = self._bf_bj_recurse(new_a, todo)
            if r > best_r:
                best_r = r
                best_a = full_a
        return best_a, best_r

    def print_stats(self):
        print 'Graph stats:'
        print '\t%d RVs' % (len(self._rvs))
        print '\t%d factors' % (len(self._factors))


class RV(object):

    def __init__(self, name, n_opts, labels=[], debug=DEBUG_DEFAULT):
        '''
        name (str)
        n_opts (int)              how many values it can take
        labels ([str], opt)
        debug (bool, opt)
        '''
        # validation
        if debug:
            # labels must be [str] if provided
            for l in labels:
                assert type(l) is str

            # must have n_opts labels if provided
            assert len(labels) == 0 or len(labels) == n_opts

        # vars set at construction time
        self.name = name
        self.n_opts = n_opts
        self.labels = labels
        self.debug = debug

        # vars added later
        self._factors = []

    def has_label(self, label):
        '''
        Returns whether label indicates a valid value for this RV.

        Args:
            label (int|str)

        returns
            bool
        '''
        # If int, make sure fits in n_opts. If str, make sure it's in the list.
        if len(self.labels) == 0:
            # Tracking ints only. Provided label must be int.
            if self.debug:
                assert type(label) is int
            return label < self.n_opts
        else:
            # Tracking strs only. Provided label can be int or str.
            if self.debug:
                assert type(label) is int or type(label) is str
            if type(label) is str:
                return label in self.labels
            # Default: int
            return label < self.n_opts

    def get_int_label(self, label):
        '''
        Returns the integer-valued label for this label. The provided label
        might be an integer (in which case it's already in the correct form and
        will be returned unchanged) or a string (in which case it will be
        turned into an int).

        This assumes the caller has already ensured this is a valid label with
        has_label.

        Args:
            label (int|str)

        returns
            int
        '''
        if type(label) is int:
            return label
        # assume string otherwise
        return self.labels.index(label)

    def attach(self, factor):
        '''
        Don't call this; automatically called by Factor's attach(...). This
        doesn't update the factor's attachment (which is why you shouldn't call
        it).

        factor (Factor)
        '''
        # check whether factor already added to rv; reach factor should be
        # added at most once to an rv.
        if self.debug:
            for f in self._factors:
                # We really only need to worry about the exact instance here,
                # so just using the builtin object (mem address) equals.
                assert f != factor, ('Can\'t re-add factor %r to rv %r' %
                                     (factor, self))

        # Do the registration
        self._factors += [factor]


class Factor(object):

    def __init__(self, rvs, name='', debug=DEBUG_DEFAULT):
        '''
        rvs ([RV])
        name (str, opt)
        debug (bool, opt)
        '''
        # at construction time
        self.name = name
        self.debug = debug

        # add later using methods
        self._rvs = []
        self._belief = None

        # set the rvs now
        for rv in rvs:
            self.attach(rv)

    def attach(self, rv):
        '''
        Call this to attach this factor to the RV rv. Clears any belief that
        has been set.

        rv (RV)
        '''
        # check whether rv already added to factor; reach rv should be added at
        # most once to a factor.
        if self.debug:
            for r in self._rvs:
                # We really only need to worry about the exact instance here,
                # so just using the builtin object (mem address) equals.
                assert r != rv, 'Can\'t re-add RV %r to factor %r' % (rv, self)

        # register with rv
        rv.attach(self)

        # register rv here
        self._rvs += [rv]

        # Clear belief as dimensions no longer match.
        self._belief = None

    def set_belief(self, b):
        '''
        Call this to set the belief for a factor. The passed belief b must
        dimensionally match all attached RVs.

        The dimensions can be a bit confusing. They iterate through the
        dimensions of the RVs in order.

        For example, say we have three RVs, which can each take on the
        following values:

            A {a, b, c}
            B {d, e}
            C {f, g}

        Now, say we have a facor which connects all of them (i.e. f(A,B,C)).
        The dimensions of the belief for this factor are 3 x 2 x 2. You can
        imagine a 3d table of numbers:

                        a b c
            a b c     +------
          + -----   d | g g g
        d | f f f / e | g g g
        e | f f f /

        This looks like you have two "sheets" of numbers. The lower sheet (on
        the left) contains the values for C = f, and the upper sheet (on the
        right) contains the values for C = g. A single cell contains the joint.
        For example, the top-left cell of the bottom sheet contains the value
        for f(A=a, B=d, C=f), and the middle-bottom cell of the top sheet
        contains the value for f(A=b, B=e, c=g).

        The confusing thing (for me) is that a single belief of shape (3, 2, 2)
        is represented in numpy as the following array:

           [[[n, n],
             [n, n]],

            [[n, n],
             [n, n]],

            [[n, n],
             [n, n]]]

        Though this still has twelve numbers, it wasn't how I was
        conceptualizing it. What gives? Well, what we're doing is indexing in
        the correct order. So, the first dimension, 3, indexes the value for
        the variable A. This visually splits our table into three areas, one
        each for A=a, A=b, and A=c. For each area, we have a 2 x 2 table. These
        would be represented in our 3d diagram above by three vertial sheets.
        Each 2 x 2 table has the values for B and C.

        So it really turns out I'd drawn my first table wrong for thinking
        about numpy arrays. You want to draw them by splitting up tables by the
        earlier Rvs. This would look like:

        A = a:
                d e
              +----
            d | n n
            e | n n

        A = b:
                f g
              +----
            d | n n
            e | n n

        A = c:
                f g
              +----
            d | n n
            e | n n

        Args:
            b (np.array)
        '''
        # check that the new belief has the correct shape
        if self.debug:
            # ensure overall dims match
            got = len(b.shape)
            want = len(self._rvs)
            assert got == want, ('Belief %r has %d dims but needs %d' %
                                 (b, got, want))

            # Ensure each dim matches.
            for i, d in enumerate(b.shape):
                got = d
                want = self._rvs[i].n_opts
                assert got == want, (
                    'Belief %r dim #%d has %d opts but rv has %d opts' %
                    (b, i+1, got, want))

        # Set it
        self._belief = b

    def eval(self, x):
        '''
        If this factor is f_a, x may have assignments to more than x_a; that
        is, it might be the full x.

        This checks (if debug is on) that all attached RVs have a valid
        assignment in x. Note that if this is begin called from Graph.joint(),
        this property is also checked there.
        '''
        if self.debug:
            # check that each RV has a valid assignment (<-)
            for rv in self._rvs:
                assert rv.name in x
                assert rv.has_label(x[rv.name])

        # slim down belief into desired value.
        ret = self._belief
        for r in self._rvs:
            ret = ret[r.get_int_label(x[r.name])]

        # should return a single number
        if self.debug:
            assert type(ret) is not np.ndarray

        return ret



# TODO(mbforbes): Remove all the main stuff once you have tests for this.
def main():
    # rvs
    r1 = RV('foo', 2)
    r2 = RV('bar', 3)

    # factors
    f = Factor([r1, r2])
    f_bar = Factor([r2])

    # beliefs
    b = np.array([[3, 2, 0.32453563], [2, 5, 5]])
    f.set_belief(b)
    b2 = np.array([8.0, 2.0, 3.0])
    f_bar.set_belief(b2)

    # add to a graph
    g = Graph()
    g.add_rv(r1)
    g.add_rv(r2)
    g.add_factor(f)
    g.add_factor(f_bar)

    # Print stuff
    g.print_stats()
    print 'Joint for foo=2, bar=1:', g.joint({'foo': 1, 'bar': 1})
    print 'Best joint:', g.bf_best_joint()

if __name__ == '__main__':
    main()
