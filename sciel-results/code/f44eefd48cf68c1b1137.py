import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The Cohn-Goncalves construction for the uncertainty bound is sensitive to the
    # parameter alpha of the generalized Laguerre polynomial L_m^(alpha).
    # The relevant uncertainty constant scales with the energy of the polynomial,
    # which is proportional to m(m + alpha + 1). To minimize the upper bound,
    # we minimize alpha, subject to the constraint alpha > -1 for positive distinct roots.
    # We choose alpha close to -1 to minimize the root spread and energy.
    alpha = -0.999
    roots, _ = roots_genlaguerre(12, alpha)
    # Return as a list of floats
    return [float(r) for r in roots]
