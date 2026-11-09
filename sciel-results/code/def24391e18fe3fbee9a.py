import numpy as np
from scipy.special import eval_legendre

def _shifted_legendre(t, deg):
    """
    Compute shifted Legendre polynomials P_0(t) through P_deg(t) for t in [0, 1].
    Uses the three-term recurrence for numerical stability.
    
    Parameters
    ----------
    t : ndarray
        Values in [0, 1].
    deg : int
        Maximum degree of the polynomial.
    
    Returns
    -------
    ndarray
        Array of shape (deg + 1, N) where N is the size of t.
    """
    p = np.empty((deg + 1, len(t)))
    p[0] = 1.0
    if deg >= 1:
        p[1] = 2.0 * t - 1.0
        for k in range(2, deg + 1):
            # Shifted Legendre recurrence
            p[k] = (2.0 * k + 1.0) * (2.0 * t - 1.0) * p[k-1] - (k - 1.0) * p[k-2]
    return p

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Symbolic Regression Basis.
    
    This function constructs a flexible 10-term basis designed to capture a wide
    range of functional forms common in symbolic regression tasks on the domain
    [0.5, 2]^3.
    
    The basis includes:
    1.  Constant term.
    2.  Linear terms in x, y, z.
    3.  Quadratic terms in x, y, z.
    4.  Interaction terms xy, xz, yz.
    5.  One higher-order interaction term xyz to capture potential synergistic effects.
    
    All inputs are normalized to [0, 1] to improve the conditioning of the linear
    least-squares problem (implicit in the BFGS fit of params). Normalization helps
    ensure that the parameter space is well-scaled, which can aid the optimizer's
    convergence and robustness.
    
    Basis Terms:
    0: 1
    1: xn
    2: yn
    3: zn
    4: xn^2
    5: yn^2
    6: zn^2
    7: xn * yn
    8: xn * zn
    9: yn * zn
    
    Note: I chose to prioritize pairwise interactions over the cubic term xyz or
    higher-order single-variable terms. In many physical and mathematical models,
    pairwise interactions are more prevalent than triple interactions. Additionally,
    the full quadratic model (degree 2) is a standard strong baseline. If the target
    function has significant triple interactions, the pairwise terms combined with
    the flexibility of the fit might still capture much of the variance, or the
    optimizer might find that a subset of these terms is sufficient.
    
    Alternative considered: Using Legendre polynomials. While orthogonal, the
    non-orthogonal monomial basis with normalized inputs is often sufficient and
    more interpretable. The normalization handles the scaling issue that usually
    motivates orthogonal polynomials.
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1]
    # This improves the condition number of the feature matrix
    xn = (x - 0.5) / 1.5
    yn = (y - 0.5) / 1.5
    zn = (z - 0.5) / 1.5
    
    # Compute basis terms
    term0 = 1.0
    term1 = xn
    term2 = yn
    term3 = zn
    term4 = xn * xn
    term5 = yn * yn
    term6 = zn * zn
    term7 = xn * yn
    term8 = xn * zn
    term9 = yn * zn
    
    # Construct the prediction as a linear combination of the basis terms
    # We explicitly sum them to ensure all 10 parameters are used and to maintain
    # a clear structure.
    pred = (p[0] * term0 + 
            p[1] * term1 + 
            p[2] * term2 + 
            p[3] * term3 + 
            p[4] * term4 + 
            p[5] * term5 + 
            p[6] * term6 + 
            p[7] * term7 + 
            p[8] * term8 + 
            p[9] * term9)
            
    return pred
