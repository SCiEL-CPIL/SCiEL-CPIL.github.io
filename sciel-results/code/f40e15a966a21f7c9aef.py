<<<<<<< SEARCH
from scipy.special import roots_legendre, roots_cheby1
=======
from scipy.special import roots_legendre
>>>>>>> REPLACE
<<<<<<< SEARCH
    # --- Initial guess: Moment-matching with Chebyshev polynomials ---
    # Use Chebyshev nodes and weights as a starting point
    n = 8
    cheb_nodes, cheb_weights = roots_cheby1(n)
    # Map Chebyshev nodes from [-1,1] to [0,1]
    cheb_nodes_mapped = 0.5 * (cheb_nodes + 1.0)
    cheb_weights_mapped = cheb_weights / 2.0

    # Encode Chebyshev nodes to gap parameterization
    # We need to find g such that decode(g, v) gives nodes close to cheb_nodes_mapped
    # This is an inverse problem. Instead, we'll use a simple heuristic:
    # Assume uniform gaps initially, then optimize.
    g_init = np.zeros(7)
    v_init = np.log(cheb_weights_mapped)
    x0 = np.concatenate([g_init, v_init])
=======
    # --- Initial guess: Uniform nodes and weights ---
    # Start with uniform spacing and equal weights
    g_init = np.zeros(7)
    v_init = np.zeros(8)
    x0 = np.concatenate([g_init, v_init])
>>>>>>> REPLACE
