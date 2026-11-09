"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for basis matrices to avoid recomputation
_basis_cache = {}

def _get_basis_matrix(x: np.ndarray) -> np.ndarray:
    """
    Return a matrix B of shape (n_points, 10) where B[:, j] is the j-th basis function evaluated at x.
    Caches results based on the tuple of x values.
    """
    # Create a hashable key from x
    key = tuple(x)
    
    if key in _basis_cache:
        return _basis_cache[key]
        
    # Ensure x is a 1D array of floats
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    
    # Define basis functions.
    # We use a mix of polynomial, trigonometric, exponential, and log bases.
    # Scaling is crucial for BFGS to converge well when starting from ones.
    # We want each basis function to have a mean magnitude of roughly O(1) over the domain [0.1, 6].
    
    b0 = np.ones_like(x)
    
    # Polynomial terms, scaled by 3.0 to keep them bounded
    # Alternatively, just use x, x^2, etc. but let the params adapt.
    # However, x^4 at x=6 is 1296. If param is ~1, this dominates.
    # Better to scale basis functions so that max |b_i| is around 1 or 10.
    
    # Let's use a set of well-conditioned bases.
    # 0: 1
    # 1: x / 3.0  (range 0.03 to 2)
    # 2: (x/3.0)^2 (range 0.001 to 4)
    # 3: (x/3.0)^3 (range 0.0001 to 8)
    # 4: (x/3.0)^4 (range 0.00001 to 16)
    # 5: sin(x)
    # 6: cos(x)
    # 7: exp(x) / exp(3)  (centered around 1, range exp(-2.9)/1 to exp(3)/1 ~ 18) -- maybe too wide
    #    Let's use exp((x-3)/3) which is exp(-0.93) to exp(1). Range ~0.4 to 2.7. Better.
    # 8: log(x) (range -2.3 to 1.8)
    # 9: 1/x / 1.0 (range 0.17 to 10) -- maybe scale by 0.2 to get 0.03 to 2
    
    # Revised basis for better conditioning:
    s = x / 3.0  # normalized x, range ~0.033 to 2.0
    
    b1 = s
    b2 = s ** 2
    b3 = s ** 3
    b4 = s ** 4
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp((x - 3.0) / 3.0)
    b8 = np.log(x)
    b9 = 0.2 / x  # 1/x is 10 to 0.17, so 0.2/x is 0.2 to 0.033
    
    B = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Store in cache (limit cache size if necessary, but 200+200 points is small)
    _basis_cache[key] = B
    
    return B

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a linear combination of scaled basis functions.
    The parameters are fit by the evaluator using BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Get the precomputed basis matrix
    B = _get_basis_matrix(x)
    
    # Compute the linear combination
    # B is (n, 10), params is (10,)
    y = B @ params
    
    # Ensure finite output for safety, although inputs are valid
    # If any non-finite values appear, clip them to large finite numbers
    # to prevent optimization failure due to inf/nan
    if not np.all(np.isfinite(y)):
         y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
    return y

# EVOLVE-BLOCK-END
