# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_area(x):
    """Compute the union area of 64 triangles given base positions x."""
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(20000) + 0.5) / 20000
    # L_i(y) = x_i + a_i * y
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    L_sorted = np.sort(L, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    return np.mean(union_len)


def _compute_area_grad(x):
    """Compute union area and its gradient w.r.t. x using automatic differentiation-like approach.
    
    The gradient can be computed analytically:
    For each y_m, the union length is w + sum over consecutive sorted pairs of min(L_{j+1} - L_j, w).
    
    The derivative of the union length w.r.t. x_i is:
    - If L_i is not at the boundary of the union (i.e., it's covered), derivative is 0
    - If L_i is the leftmost point of the union, derivative is +1
    - If L_i is the rightmost point of the union, derivative is -1
    - More generally, we need to track which segments contribute
    
    Actually, let's use a simpler approach: finite differences for the gradient,
    or better yet, use the fact that the objective is piecewise linear in x.
    
    For now, let's use a vectorized approach for the gradient:
    The union length at height y is:
    U(y) = w + sum_{j=0}^{n-2} min(L_{(j+1)} - L_{(j)}, w)
    where L_{(j)} is the j-th smallest L value.
    
    dU/dx_i = sum over all j where L_i participates in a min term.
    
    Let's think more carefully. Let the sorted L values be L_(1) <= L_(2) <= ... <= L_(n).
    The union is [L_(1), L_(1) + w] union [L_(2), L_(2) + w] union ... union [L_(n), L_(n) + w].
    
    Actually, the union of intervals [L_j, L_j + w] for j=1..n, when sorted by L_j, has total length:
    If we merge overlapping intervals, the total length is:
    Start from the leftmost, keep extending the right endpoint.
    
    But the formula given is: w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    This is the length of the union of the intervals [L_j, L_j + w] when the L_j are sorted.
    
    Wait, that's not quite right. The union of intervals [a_j, a_j + w] where a_1 <= a_2 <= ... <= a_n:
    The union length is: if a_{j+1} <= a_j + w, then the intervals overlap and the union extends to max(a_j + w, a_{j+1} + w) = a_{j+1} + w (since a_{j+1} >= a_j).
    
    Actually, the standard formula for the length of the union of intervals [a_j, a_j + w] sorted by a_j is:
    union_len = w + sum_{j=1}^{n-1} min(a_{j+1} - a_j, w)
    
    This is because:
    - Start with the first interval [a_1, a_1 + w], length w.
    - For each subsequent interval [a_{j+1}, a_{j+1} + w]:
      - If a_{j+1} <= a_j + w (overlap), the new union extends by (a_{j+1} + w) - (a_j + w) = a_{j+1} - a_j, but capped at w... 
      
    Hmm, let me reconsider. Actually the formula is correct as given in the problem.
    
    For the gradient, since the objective is piecewise linear (sum of min functions), the gradient exists almost everywhere and can be computed as:
    
    dU/dx_i at a given y = sum over j where L_i is involved in a min(L_{j+1} - L_j, w) term.
    
    Specifically, if L_i = L_{(j)} (the j-th smallest), then:
    - It appears in min(L_{(j+1)} - L_{(j)}, w) with coefficient -1 if L_{(j+1)} - L_{(j)} < w
    - It appears in min(L_{(j)} - L_{(j-1)}, w) with coefficient +1 if L_{(j)} - L_{(j-1)} < w
    
    So dU/dx_i = (1 if L_{(j)} - L_{(j-1)} < w else 0) - (1 if L_{(j+1)} - L_{(j)} < w else 0)
    with appropriate boundary conditions (j=1 has no left neighbor, j=n has no right neighbor).
    
    Let's implement this.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(20000) + 0.5) / 20000
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (20000,)
    
    # Sort L and track original indices
    L_sorted = np.sort(L, axis=1)  # shape (20000, 64)
    
    # Compute diffs
    diffs = np.diff(L_sorted, axis=1)  # shape (20000, 63)
    
    # Union length
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient computation
    # For each y_m, for each original index i, find its position in the sorted order
    # and determine the gradient contribution.
    
    # To avoid expensive argsort for each y, we can use the fact that we already have L_sorted.
    # Let's compute the gradient by tracking which original index is at each sorted position.
    
    # Actually, let's use a different approach: compute the gradient via the sorted positions.
    # For each y_m, let the sorted order be perm_m, so L_sorted[m, j] = L[m, perm_m[j]].
    # Then for original index i, if perm_m[j] = i, then:
    #   dU_m/dx_i = (1 if j > 0 and diffs[m, j-1] < w[m] else 0) - (1 if j < n-1 and diffs[m, j] < w[m] else 0)
    
    # We need the permutation. Let's compute it.
    # This is expensive but let's try.
    
    # Alternative: use finite differences for the gradient in the optimizer.
    # For now, let's just return the area and use finite differences in scipy.
    
    return area, None  # Use finite differences


def _objective_and_grad(x):
    """Wrapper for scipy minimize that returns (f, g) where g is None for finite differences."""
    area, _ = _compute_area_grad(x)
    return area, None


def _keich_initial(n=64):
    """Generate the Keich initial positions."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses the Keich construction as an initial guess, then applies Nelder-Mead
    optimization with random restarts to escape local minima.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    # Initial guess: Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(best_x)
    
    # Nelder-Mead optimization
    # Use a reasonable initial simplex size
    initial_simplex_size = 0.01
    
    # Try multiple random restarts
    np.random.seed(42)
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        # For the first restart, use the Keich initial; for others, add random perturbation
        if restart == 0:
            x_start = x0.copy()
        else:
            # Random perturbation from the best found so far
            perturbation_scale = 0.05 * (1.0 - restart / max_restarts)
            x_start = best_x + np.random.randn(n) * perturbation_scale
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _objective_and_grad,
                x_start,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'maxiter': 500,
                    'maxfev': 1000,
                    'adaptive': True
                }
            )
            
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective_and_grad,
                best_x,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'maxiter': 1000,
                    'maxfev': 2000,
                    'adaptive': True
                }
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_x)):
        return x0
    
    return best_x


# EVOLVE-BLOCK-END
