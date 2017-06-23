import numpy as np
import factorgraph as fg

# Make an empty graph
g = fg.Graph()

# Add some discrete random variables (RVs)
g.rv('a', 2)
g.rv('b', 3)

# Add some factors, unary and binary
g.factor(['a'], potential=np.array([0.3, 0.7]))
g.factor(['b', 'a'], potential=np.array([
        [0.2, 0.8],
        [0.4, 0.6],
        [0.1, 0.9],
]))

# Run (loopy) belief propagation (LBP)
iters, converged = g.lbp(normalize=True)
print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
print

# Print out the final messages from LBP
g.print_messages()
print

# Print out the final marginals
g.print_rv_marginals()
