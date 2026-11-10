"""Symbolic regression with hybrid local-global basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None


def _cubic_b_spline_basis(x, knots, n_ctrl):
    """
    Compute cubic B-spline basis functions for given knots and control points.
    Uses de Boor's algorithm for evaluation.
    
    x: array of evaluation points
    knots: knot vector (length n_ctrl + 4 for cubic)
    n_ctrl: number of control points
    
    Returns: (n_points, n_ctrl) array of basis values
    """
    n = len(x)
    order = 4  # cubic
    
    # Build knot vector: extend by 3 on each side
    # knots should already be the full knot vector of length n_ctrl + 4
    K = knots
    nk = len(K)
    
    # For each x, find the knot span
    # Use searchsorted to find span
    # For cubic B-spline, span i means K[i] <= x < K[i+1]
    # We need spans in range [order-1, nk-order)
    
    spans = np.zeros(n, dtype=np.int64)
    # Use searchsorted: find rightmost knot <= x
    # np.searchsorted(K, x, side='right') - 1 gives the span index
    # But we need to handle boundaries
    idx = np.searchsorted(K, x, side='right') - 1
    # Clamp to valid range
    idx = np.clip(idx, order - 1, nk - order - 1)
    spans = idx
    
    # Compute basis values using de Boor's algorithm
    N = np.zeros((n, n_ctrl), dtype=np.float64)
    
    # Initialize N for the lowest degree (linear, degree 1)
    # Actually, let's use the standard recursive definition
    # N_{i,0}(x) = 1 if K[i] <= x < K[i+1], else 0
    # N_{i,k}(x) = (x - K[i])/(K[i+k] - K[i]) * N_{i,k-1}(x) + (K[i+k+1] - x)/(K[i+k+1] - K[i+k+1]) * N_{i+1,k-1}(x)
    
    # More efficient: use the fact that we only need the basis functions
    # that are non-zero at each x point.
    
    # For each point, the non-zero basis functions are those with indices
    # spans[i] - 3, spans[i] - 2, spans[i] - 1, spans[i]
    
    # Let's compute using the Cox-de Boor recursion
    # Start with degree 0
    N_curr = np.zeros((n, nk - 1), dtype=np.float64)
    for i in range(nk - 1):
        mask = (x >= K[i]) & (x < K[i+1])
        if i == nk - 2:
            mask = (x >= K[i]) & (x <= K[i+1])  # include right endpoint
        N_curr[:, i] = mask.astype(np.float64)
    
    # Recurse up to degree 3
    for k in range(1, order):
        N_next = np.zeros((n, nk - 1 - k), dtype=np.float64)
        for i in range(nk - 1 - k):
            denom1 = K[i + k] - K[i]
            denom2 = K[i + k + 1] - K[i + 1]
            
            if denom1 > 1e-12:
                term1 = (x - K[i]) / denom1 * N_curr[:, i]
            else:
                term1 = np.zeros(n, dtype=np.float64)
            
            if denom2 > 1e-12:
                term2 = (K[i + k + 1] - x) / denom2 * N_curr[:, i + 1]
            else:
                term2 = np.zeros(n, dtype=np.float64)
            
            N_next[:, i] = term1 + term2
        
        N_curr = N_next
    
    # N_curr is now (n, n_ctrl) since nk - 1 - (order-1) = nk - order = n_ctrl
    return N_curr


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining local B-spline basis with global Chebyshev basis.
    
    y(x) = sum_{i=0}^{5} p_i * B_i(x) + sum_{j=0}^{3} q_j * T_j(x/2)
    
    where B_i are cubic B-spline basis functions with fixed knots,
    and T_j are Chebyshev polynomials of the first kind.
    
    The 10 params are: p0..p5 (B-spline control points) and q0..q3 (Chebyshev coeffs).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > 145.0:
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x)
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    p0, p1, p2, p3, p4, p5, q0, q1, q2, q3 = params
    
    # --- B-spline part ---
    # Define knot vector for 6 cubic B-spline basis functions
    # Interior knots at: -1, 0, 1 (3 interior knots)
    # Full knot vector: [-2-3h, -2-2h, -2-h, -2, -1, 0, 1, 2, 2+h, 2+2h, 2+3h]
    # With h = 2/3 for uniform spacing over [-2, 2]
    h = 2.0 / 3.0
    knots = np.array([
        -2 - 3*h, -2 - 2*h, -2 - h, -2,
        -1, 0, 1,
        2, 2 + h, 2 + 2*h, 2 + 3*h
    ])
    
    # Compute B-spline basis values
    N = _cubic_b_spline_basis(x, knots, 6)
    
    # B-spline contribution
    bspline_part = (p0 * N[:, 0] + p1 * N[:, 1] + p2 * N[:, 2] +
                    p3 * N[:, 3] + p4 * N[:, 4] + p5 * N[:, 5])
    
    # --- Chebyshev part ---
    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Chebyshev polynomials T0 through T3
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    
    # Chebyshev contribution
    cheb_part = q0 * T0 + q1 * T1 + q2 * T2 + q3 * T3
    
    # Combine
    result = bspline_part + cheb_part
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
