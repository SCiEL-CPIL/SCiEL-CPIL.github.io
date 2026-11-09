"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def construct():
    """
    Maximize ratio = ||f||_2^2 / (||f||_1 * sup|f*f|)
    
    For step functions on [-1/4, 1/4] with N=50 equal intervals:
    - ||f||_2^2 = sum(a_i^2) * dx
    - ||f||_1 = sum(a_i) * dx
    - sup|f*f|: The autoconvolution of a step function has triangular peaks
      at the edges. For nonnegative f, the max is at the edges.
      At the left edge, the autoconvolution value is approximately
      max(a_i)^2 * dx (when the tallest bar is at position 0).
      More precisely, sup|f*f| ≈ max_i(a_i)^2 * dx for the edge contribution.
    
    So ratio ≈ sum(a_i^2) / (sum(a_i) * max(a_i)^2)
    
    This is maximized when many bars are at the maximum height,
    favoring uniform or plateau shapes.
    """
    
    dx = 1.0 / N
    
    def compute_ratio(heights):
        a = np.array(heights, dtype=np.float64)
        if len(a) == 0:
            return 0.0
        norm2_sq = np.sum(a**2) * dx
        norm1 = np.sum(a) * dx
        
        # For nonnegative step functions, sup of autoconvolution
        # occurs at the edges. The peak value at an edge is:
        # (f*f)(-L/2 + k*dx) for k near 0
        # For the leftmost edge: integral f(s)f(s+t) ds where t is at the boundary
        # This gives a triangular shape with peak = max(a_i)^2 * dx
        # when the tallest bar is at the edge.
        # 
        # More carefully: for a step function, the autoconvolution at the
        # left edge (t = -L/2) is:
        #   sum_i a_i * a_{i+1} * dx  (approximately, for the overlap)
        # But the TRUE supremum for nonnegative functions on a bounded domain
        # is at t=0: (f*f)(0) = integral f(x)^2 dx = sum(a_i^2) * dx
        #
        # Wait - let me reconsider. For nonnegative f, (f*f)(t) is maximized
        # at t=0 by the autocorrelation property. So sup|f*f| = (f*f)(0) = sum(a_i^2)*dx
        #
        # Then ratio = sum(a_i^2)*dx / (sum(a_i)*dx * sum(a_i^2)*dx) = 1/(sum(a_i)*dx)
        # This is scale-invariant and only depends on the shape through sum(a_i).
        # But since we can scale, the ratio is actually:
        # ratio = ||f||_2^2 / (||f||_1 * ||f*f||_inf)
        #       = (sum a_i^2 * dx) / ((sum a_i * dx) * (sum a_i^2 * dx))
        #       = 1 / (sum a_i * dx)
        #
        # Hmm, this means for ANY shape, the ratio is the same if we normalize L1?
        # No wait - this can't be right because the seed (uniform) scores 1.0.
        #
        # Let me re-examine. The issue is that for step functions, the autoconvolution
        # is NOT simply sum(a_i^2)*dx at lag 0. The discrete autoconvolution of
        # step functions has a specific structure.
        #
        # Actually, I think the correct interpretation is:
        # f is a step function: f(x) = a_i for x in [x_i, x_{i+1})
        # (f*f)(t) = integral f(s) f(s-t) ds
        # At t=0: integral f(s)^2 ds = sum(a_i^2) * dx
        # At other t: it's a sum of products with triangular weights
        #
        # For nonnegative f, (f*f)(t) <= (f*f)(0) is NOT necessarily true for
        # step functions! The autoconvolution can have local maxima away from 0.
        #
        # Actually for L2 functions, (f*f)(0) = ||f||_2^2 and by Young's inequality
        # ||f*f||_inf <= ||f||_1 * ||f||_inf, but that's a different bound.
        #
        # Let me just compute the autoconvolution numerically for step functions.
        
        # Compute autoconvolution properly for step functions
        # The domain is [-1/4, 1/4], so L = 1/2, dx = L/N = 1/(2*N)
        # Wait, the problem says "equal intervals of [-1/4, 1/4]"
        # So the total width is 1/2, and dx = (1/2)/N = 1/(2*N)
        
        # Let me redo with correct dx
        L = 0.5  # total width of [-1/4, 1/4]
        dx_correct = L / N
        
        norm2_sq = np.sum(a**2) * dx_correct
        norm1 = np.sum(a) * dx_correct
        
        # Compute autoconvolution at all relevant lags
        # For step functions, (f*f)(k*dx) for integer k
        # (f*f)(k*dx) = sum_i a_i * a_{i-k} * dx_correct  (with appropriate bounds)
        # But this is an approximation. The exact autoconvolution of step functions
        # has a more complex form.
        
        # For now, use the discrete approximation:
        # autoconv[k] = sum_i a[i] * a[i-k] * dx for valid i
        autoconv = np.correlate(a, a, mode='full') * dx_correct
        
        sup_autoconv = np.max(autoconv)
        
        if sup_autoconv < 1e-15 or norm1 < 1e-15:
            return 0.0
        
        ratio = norm2_sq / (norm1 * sup_autoconv)
        return ratio
    
    best_ratio = 0.0
    best_heights = [1.0] * N
    
    # Test various shapes
    idx = np.arange(N)
    center = (N - 1) / 2.0
    
    # Uniform
    r = compute_ratio([1.0] * N)
    best_ratio = r
    best_heights = [1.0] * N
    
    # Plateau at edges (tall at edges, zero in middle)
    plateau = np.zeros(N)
    plateau[:N//4] = 1.0
    plateau[-(N//4):] = 1.0
    r = compute_ratio(list(plateau))
    if r > best_ratio:
        best_ratio = r
        best_heights = list(plateau)
    
    # Single tall bar at edge
    for edge_pos in [0, N-1, N//2]:
        h = np.zeros(N)
        h[edge_pos] = 1.0
        r = compute_ratio(list(h))
        if r > best_ratio:
            best_ratio = r
            best_heights = list(h)
    
    # Try scipy optimization
    try:
        from scipy.optimize import minimize
        
        def neg_ratio(x):
            if time.time() - START_TIME > TIME_LIMIT:
                return 0.0
            h = np.maximum(x, 0.0)  # enforce nonnegativity
            return -compute_ratio(list(h))
        
        # Start from uniform
        x0 = np.ones(N)
        
        result = minimize(neg_ratio, x0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if result.fun < 0:
            r_opt = -result.fun
            if r_opt > best_ratio:
                best_ratio = r_opt
                best_heights = [max(0.0, x) for x in result.x]
        
    except Exception:
        pass
    
    # Ensure nonnegative and valid
    best_heights = [max(0.0, h) for h in best_heights]
    if np.sum(best_heights) < 1e-10:
        best_heights = [1.0] * N
    
    return best_heights
