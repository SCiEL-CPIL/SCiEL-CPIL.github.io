"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        # Compute squared distances between all pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    def objective(angles):
        # angles: shape (16,)
        # Construct coefficients c_k = exp(i * angles_k)
        # P(z) = sum_{k=0}^{15} c_k z^k
        # The roots of P(z) are the 16 points.
        
        # Build the companion matrix to find roots.
        # P(z) = c_15 z^15 + c_14 z^14 + ... + c_0 = 0
        # Let coeffs = [c_15, c_14, ..., c_0]
        coeffs = np.exp(1j * angles)
        
        # np.roots expects [a_n, ..., a_1, a_0]
        # So we reverse coeffs
        coeffs_rev = coeffs[::-1]
        
        try:
            roots = np.roots(coeffs_rev)
        except Exception:
            return 1e10
            
        pts = np.column_stack([roots.real, roots.imag])
        
        # Check for distinct points (min distance should be > 0)
        # Also check for numerical stability (points shouldn't be too far apart or too close)
        # Normalize to ensure scale invariance is handled consistently, though ratio is scale invariant.
        # However, normalization helps with numerical precision in distance calculation.
        pts = normalize(pts)
        
        # Ensure no points are at infinity or NaN
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        return score_raw(pts)

    # Strategy: Differential Evolution (DE)
    # DE is a population-based stochastic method that does not use gradient information.
    # It is robust against local minima and works well for multi-modal, non-smooth functions.
    # The parameter space is 16 angles (0 to 2*pi).
    
    bounds = [(0, 2 * np.pi)] * N
    
    # Use a smaller population and fewer iterations to fit within time limit,
    # but DE is generally efficient.
    # maxiter=100, popsize=20 -> 2000 evaluations.
    # Each evaluation involves finding roots of a 15th degree polynomial (fast, ~1ms).
    # 2000 evals * 1ms = 2s. We have 145s.
    # We can afford much more. Let's try maxiter=500, popsize=30.
    # 500 * 30 = 15000 evals. 15s.
    # Let's go even higher to ensure convergence. maxiter=1000, popsize=30 -> 30000 evals. 30s.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            popsize=30, 
            tol=1e-8, 
            polish=False, # We will do a final polish if needed, or just return best
            update_interval=10,
            callback=None
        )
        
        best_angles = result.x
        best_score = result.fun
        
    except Exception as e:
        # Fallback if DE fails
        best_angles = np.random.uniform(0, 2*np.pi, N)
        best_score = objective(best_angles)
        
    # Get the best points
    coeffs = np.exp(1j * best_angles)
    coeffs_rev = coeffs[::-1]
    roots = np.roots(coeffs_rev)
    final_pts = np.column_stack([roots.real, roots.imag])
    final_pts = normalize(final_pts)
    
    # Optional: Local polish with Nelder-Mead on the angles?
    # The objective function (root calculation) is differentiable but np.roots is not smooth in all regimes.
    # Direct coordinate optimization might be better for polish, but we don't have a direct mapping from angles to coordinates that is easy to invert locally.
    # Given the time, we'll stick with the DE result.
    
    return final_pts

# EVOLVE-BLOCK-END
