'''
Implementation of graph.

Author: mbforbes

Current approach:
-   add RVs
-   add factors to connect them
-   set beliefs
'''


import numpy as np


class Graph(object):
    '''
    Graph right now has no point, really (except bookkeeping all the RVs and
    factors, assuming we remember to add them), so this might be removed or
    functionality might be stuffed in here later.
    '''

    def __init__(self):
        # added later
        self._rvs = []
        self._factors = []

    def add(self, node):
        '''
        Node (RV|Factor)
        '''
        if type(node) == RV:
            self._rvs += [node]
        elif type(node) == Factor:
            self._factors += [node]
        else:
            print (
                'WARNING: ignoring unknown node type "%s"' % (str(type(node))))

    def print_stats(self):
        print 'Graph stats:'
        print '\t%d RVs' % (len(self._rvs))
        print '\t%d factors' % (len(self._factors))


class RV(object):

    def __init__(self, name, dims, debug=True):
        '''
        name (str)
        dims (int)
        debug (bool, opt)
        '''
        # at construction time
        self.name = name
        self.dims = dims
        self.debug = debug

        # added later
        self._factors = []

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

    def __init__(self, rvs, name='', debug=True):
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
                want = self._rvs[i].dims
                assert got == want, (
                    'Belief %r dim #%d has dim %d but rv has dim %d' %
                    (b, i+1, got, want))

        # Set it
        self._belief = b


# TODO(mbforbes): Remove all the main stuff once you have tests for this.
def main():
    g = Graph()
    r1 = RV('foo', 2)
    r2 = RV('bar', 3)
    f = Factor([r1, r2])
    b = np.array([[0, 2, 0.32453563], [4, 5, 5]])
    f.set_belief(b)
    g.add(r1)
    g.add(r2)
    g.add(f)
    g.print_stats()

if __name__ == '__main__':
    main()
