"""Optimal ±1 polynomials with 24 coefficients minimizing sup norm.

Structural change: Use a multi-start tabu search with a vectorized dense-grid
fitness function, combined with high-accuracy golden-section refinement at the
end. The key insight is that the known "optimal" sequence in the current
program may not be the global optimum due to measurement error or a wrong
sequence. We do an aggressive deterministic+stochastic hybrid search.

The strategy:
1. Precompute a dense grid of z values and a Vandermonde-like matrix for fast
   evaluation of |P(z)| for any coefficient vector.
2. Use a tabu search (or simulated annealing) to minimize the max |P(z)| on
   the grid.
3. Finally, do golden-section refinement on the top peaks to get an accurate
   score.
4. Track the best sequence found throughout.
"""
import time
import math
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    N = 24
    sqrtN = math.sqrt(N)

    # Dense grid for fitness evaluation
    G = 16384
    angles = 2 * np.pi * np.arange(G) / G
    zs = np.exp(1j * angles)

    # Precompute powers matrix: V[i, k] = zs[i]^k
    V = np.ones((G, N), dtype=np.complex128)
    for k in range(1, N):
        V[:, k] = V[:, k - 1] * zs

    def fast_score(coeffs):
        """Compute max |P(z)| / sqrt(N) on the grid."""
        vals = V @ coeffs
        return np.max(np.abs(vals)) / sqrtN

    def refine_sup_norm(coeffs, n_peaks=10, grid_size=65536, golden_iters=45):
        """High-accuracy sup norm using dense grid + golden section."""
        angles_r = 2 * np.pi * np.arange(grid_size) / grid_size
        zs_r = np.exp(1j * angles_r)
        
        V_r = np.ones((grid_size, N), dtype=np.complex128)
        for k in range(1, N):
            V_r[:, k] = V_r[:, k - 1] * zs_r
        vals_r = V_r @ coeffs
        abs_vals_r = np.abs(vals_r)
        
        top_indices = np.argsort(abs_vals_r)[-n_peaks:][::-1]
        
        max_peak = 0.0
        gr = (math.sqrt(5) - 1) / 2
        
        for idx in top_indices:
            theta_c = angles_r[idx]
            window = 2 * np.pi / grid_size
            a = theta_c - window
            b = theta_c + window
            
            def f(theta):
                z = np.exp(1j * theta)
                val = coeffs[N - 1]
                for k in range(N - 2, -1, -1):
                    val = val * z + coeffs[k]
                return abs(val)
            
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            fc = f(c_pt)
            fd = f(d_pt)
            
            for _ in range(golden_iters):
                if fc < fd:
                    b, d_pt, fd = d_pt, c_pt, fc
                    c_pt = b - gr * (b - a)
                    fc = f(c_pt)
                else:
                    a, c_pt, fc = c_pt, d_pt, fd
                    d_pt = a + gr * (b - a)
                    fd = f(d_pt)
            
            max_peak = max(max_peak, fc, fd)
        
        return max_peak / sqrtN

    # --- Multi-start tabu search ---
    best_coeffs = None
    best_score = float('inf')
    rng = np.random.default_rng(42)

    def tabu_search(initial_coeffs, tabu_len=15, max_iters=500):
        """Run tabu search from initial_coeffs."""
        c = initial_coeffs.copy()
        cur_score = fast_score(c)
        best_local = c.copy()
        best_local_score = cur_score
        
        tabu = []  # list of (index, flip_direction, expiry)
        iter_count = 0
        
        while iter_count < max_iters:
            iter_count += 1
            best_flip = -1
            best_flip_score = float('inf')
            
            for k in range(N):
                # Check if flipping k is tabu
                is_tabu = False
                for (tk, td, expiry) in tabu:
                    if tk == k and expiry > iter_count:
                        is_tabu = True
                        break
                
                c[k] = -c[k]
                s = fast_score(c)
                c[k] = -c[k]  # undo
                
                if not is_tabu or s < best_local_score:  # aspiration criterion
                    if s < best_flip_score:
                        best_flip_score = s
                        best_flip = k
            
            if best_flip < 0:
                break
            
            # Apply best flip
            c[best_flip] = -c[best_flip]
            cur_score = best_flip_score
            
            # Update tabu
            tabu.append((best_flip, None, iter_count + tabu_len))
            
            if cur_score < best_local_score:
                best_local_score = cur_score
                best_local = c.copy()
        
        return best_local, best_local_score

    # Start from several random initializations
    n_starts = 8
    for s_idx in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        
        # Random initial coefficients
        init = rng.choice([-1.0, 1.0], size=N)
        
        # Also try the known optimal as one start
        if s_idx == 0:
            init = np.array([
                +1, -1, +1, +1, -1, +1, -1, +1,
                +1, -1, +1, -1, -1, +1, -1, -1,
                -1, +1, -1, +1, +1, -1, -1, +1
            ], dtype=np.float64)
        
        local_best, local_score = tabu_search(init)
        
        if local_score < best_score:
            best_score = local_score
            best_coeffs = local_best.copy()
    
    # --- Refinement phase: pair-flip local search on the best candidate ---
    if best_coeffs is not None and time.time() < deadline - 30.0:
        c = best_coeffs.copy()
        cur_score = fast_score(c)
        
        # Pair-flip local search
        improved = True
        while improved and time.time() < deadline - 30.0:
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    c[i] = -c[i]
                    c[j] = -c[j]
                    s = fast_score(c)
                    if s < cur_score:
                        cur_score = s
                        improved = True
                    else:
                        c[i] = -c[i]
                        c[j] = -c[j]
            # Single flips
            for k in range(N):
                c[k] = -c[k]
                s = fast_score(c)
                if s < cur_score:
                    cur_score = s
                    improved = True
                else:
                    c[k] = -c[k]
        
        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = c.copy()
    
    # --- Final high-accuracy verification ---
    if best_coeffs is not None and time.time() < deadline - 10.0:
        refined_score = refine_sup_norm(best_coeffs)
    
    # Convert to int list
    if best_coeffs is not None:
        return [int(round(x)) for x in best_coeffs]
    
    # Fallback: known optimal
    fallback = [
        +1, -1, +1, +1, -1, +1, -1, +1,
        +1, -1, +1, -1, -1, +1, -1, -1,
        -1, +1, -1, +1, +1, -1, -1, +1
    ]
    return fallback
