import numpy as np
import time
import random

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_FLAT       # x + y
Z_VALS = X_FLAT + 2 * Y_FLAT   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    # Filter out zeros to avoid log(0)
    # Use a small epsilon to handle numerical noise in probabilities
    mask = probs > 1e-12
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    # Normalize p just in case, though input should be normalized
    p_sum = np.sum(p)
    if p_sum <= 1e-12:
        return 0.0
    p = p / p_sum
    return -np.sum(p * np.log2(p))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100, non-negative, summing to 1.
    """
    # Ensure non-negative and normalized
    P_flat = np.clip(P_flat, 0, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    # Project probabilities
    p_u = np.bincount(U_INV, weights=P_flat, minlength=len(U_UNIQ))
    p_v = np.bincount(V_INV, weights=P_flat, minlength=len(V_UNIQ))
    p_w = np.bincount(W_INV, weights=P_flat, minlength=len(W_UNIQ))
    p_z = np.bincount(Z_INV, weights=P_flat, minlength=len(Z_UNIQ))
    
    h_u = _entropy_from_probs(p_u)
    h_v = _entropy_from_probs(p_v)
    h_w = _entropy_from_probs(p_w)
    h_z = _entropy_from_probs(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _coordinate_descent(P, max_iters=500, tol=1e-7):
    """
    Refine P using coordinate ascent on the simplex.
    For each coordinate i, we can adjust P[i] by moving mass from other coordinates.
    A simpler effective move: pick i and j, move mass from j to i.
    Or: fix sum, maximize f(P) w.r.t P[i] by adjusting others proportionally?
    Actually, standard simplex coordinate ascent:
    For each i, assume P[i] changes to t, and all other P[k] (k!=i) scale by (1-t)/(1-P[i]).
    This keeps the sum 1.
    We find the optimal t in [0, 1].
    """
    P = P.copy()
    P = np.clip(P, 0, None)
    P = P / np.sum(P)
    
    for _ in range(max_iters):
        improved = False
        for i in range(100):
            # Current value
            p_i = P[i]
            if p_i > 0.9999:
                continue
                
            # If we change P[i] to t, others scale by factor s = (1-t)/(1-p_i)
            # Let Q be the vector of other probabilities.
            # New P: P[i]=t, P[k] = Q[k] * s
            # We want to maximize ratio(P).
            # This is a 1D optimization problem.
            # We can use golden section search or just a few grid points.
            # Given the landscape is smooth-ish, a few steps suffice.
            
            # Let's try a few candidate values for t
            # t can range from 0 to 1.
            # However, if we set t too high, entropy might drop.
            # Let's sample t in [0, 1] with a few points, plus current p_i.
            
            best_t = p_i
            best_val = _compute_ratio(P)
            
            # Generate candidates
            # Linear space from 0 to 1 with 10 points, plus p_i
            candidates = np.linspace(0, 1, 11)
            candidates = np.append(candidates, p_i)
            
            for t in candidates:
                if t == p_i:
                    continue
                P_new = P.copy()
                P_new[i] = t
                # Scale others
                if 1 - t > 1e-12:
                    factor = (1 - t) / (1 - p_i)
                    P_new[np.arange(100) != i] *= factor
                else:
                    # t is very close to 1, others are ~0
                    P_new[np.arange(100) != i] = 0
                    P_new[i] = 1.0
                    
                val = _compute_ratio(P_new)
                if val > best_val:
                    best_val = val
                    best_t = t
                    P = P_new
                    improved = True
        
        if not improved:
            break
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin for 600s limit
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    # List of initial strategies
    initial_guesses = []
    
    # 1. Uniform
    initial_guesses.append(np.ones(100) / 100.0)
    
    # 2. Diagonal (x=y)
    P_diag = np.zeros(100)
    for i in range(10):
        P_diag[i*10+i] = 1.0/10
    initial_guesses.append(P_diag)
    
    # 3. Anti-diagonal (x+y=9)
    P_anti = np.zeros(100)
    for x in range(10):
        y = 9 - x
        P_anti[y*10+x] = 1.0/10
    initial_guesses.append(P_anti)
    
    # 4. Row (x=5)
    P_row = np.zeros(100)
    for y in range(10):
        P_row[y*10+5] = 1.0/10
    initial_guesses.append(P_row)
        
    # 5. Column (y=5)
    P_col = np.zeros(100)
    for x in range(10):
        P_col[5*10+x] = 1.0/10
    initial_guesses.append(P_col)
    
    # 6. Random sparse sets (choose K cells)
    for K in [5, 10, 20]:
        for _ in range(5):
            P_sparse = np.zeros(100)
            indices = rng.choice(100, size=K, replace=False)
            P_sparse[indices] = 1.0/K
            initial_guesses.append(P_sparse)
            
    # 7. Random Gaussian blobs
    for _ in range(10):
        cx, cy = rng.uniform(0, 9, 2)
        sigma = rng.uniform(0.5, 3.0)
        P_gauss = np.zeros(100)
        for i in range(100):
            x, y = i % 10, i // 10
            dist_sq = (x - cx)**2 + (y - cy)**2
            P_gauss[i] = np.exp(-dist_sq / (2 * sigma**2))
        P_gauss /= np.sum(P_gauss)
        initial_guesses.append(P_gauss)

    # Phase 1: Evaluate initial guesses and refine with Coordinate Descent
    for P_init in initial_guesses:
        if time.time() > deadline:
            break
            
        P_candidate = P_init.copy()
        P_candidate = _coordinate_descent(P_candidate, max_iters=100)
        ratio = _compute_ratio(P_candidate)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P_candidate

    # Phase 2: Simulated Annealing / Random Perturbation Search
    # We will run a loop of random perturbations and local refinements.
    # This helps escape local optima found by coordinate descent.
    
    current_P = best_P.copy()
    current_ratio = best_ratio
    
    # Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Number of iterations depends on speed
    # _compute_ratio is fast (~0.1ms), _coordinate_descent is slower.
    # We will do many cheap perturbations and occasional CD.
    
    iter_count = 0
    while time.time() < deadline - 10:
        iter_count += 1
        
        # Generate a perturbation
        # Strategy: Move mass from a random set of cells to another random set
        # Or: Add random noise and renormalize
        # Let's use a "move mass" approach to keep simplex constraint naturally
        if random.random() < 0.5:
            # Move mass from i to j
            i = rng.integers(0, 100)
            j = rng.integers(0, 100)
            if i == j:
                continue
            # Amount to move: proportional to current P[i] and some random factor
            # Limit to avoid negative P
            max_move = current_P[i]
            # We want to explore significantly, so move a chunk
            # If T is high, move more
            move_frac = min(1.0, rng.exponential(1.0) * T)
            move_amt = max_move * move_frac
            
            P_new = current_P.copy()
            P_new[i] -= move_amt
            P_new[j] += move_amt
        else:
            # Add noise
            noise = rng.normal(0, T, 100)
            P_new = current_P + noise
            P_new = np.clip(P_new, 0, None)
            if np.sum(P_new) < 1e-9:
                P_new = current_P.copy()
            else:
                P_new /= np.sum(P_new)
                
        # Evaluate
        new_ratio = _compute_ratio(P_new)
        
        # Acceptance criterion (Metropolis)
        if new_ratio > current_ratio:
            current_P = P_new
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_P = P_new.copy()
        else:
            # Accept with probability exp((new - curr) / T)
            delta = new_ratio - current_ratio
            if delta > 0:
                accept = True
            else:
                # delta is negative
                # Ratio is bounded, so delta is not huge
                prob = np.exp(delta / T)
                accept = (random.random() < prob)
                
            if accept:
                current_P = P_new
                current_ratio = new_ratio
                
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # Occasionally reheat to escape local minima?
            # Let's just keep cooling but add a restart mechanism if stuck
            # Check if we've been stuck for a while
            # Simple heuristic: if T is low and we haven't improved best in last 1000 iters, restart from best_P with higher T
            # For simplicity, we just let it run. The random restarts in Phase 1 covered diverse starts.
            
        # Every N iterations, do a coordinate descent refinement on current_P
        # CD is more expensive, so do it less frequently
        if iter_count % 50 == 0:
             # Only refine if we have time
             if time.time() < deadline - 20:
                 P_refined = _coordinate_descent(current_P, max_iters=50)
                 ref_ratio = _compute_ratio(P_refined)
                 if ref_ratio > current_ratio:
                     current_P = P_refined
                     current_ratio = ref_ratio
                     if ref_ratio > best_ratio:
                         best_ratio = ref_ratio
                         best_P = P_refined.copy()
                 else:
                     # If CD didn't help, keep current_P but maybe revert to best_P if we drifted too far?
                     # No, SA handles drift.
                     pass

    # Final refinement on the best found
    if time.time() < deadline - 5:
        P_final = _coordinate_descent(best_P, max_iters=200)
        final_ratio = _compute_ratio(P_final)
        if final_ratio > best_ratio:
            best_P = P_final
            best_ratio = final_ratio

    # Convert best_P to 10x10 array
    P_out = best_P.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out

# EVOLVE-BLOCK-END
