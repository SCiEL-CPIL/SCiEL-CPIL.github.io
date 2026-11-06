import time
import random
import numpy as np

def _square_nonzero_count_sparse(poly):
    """
    Faster evaluation using sparse convolution for small term counts.
    poly is a list of (exponent, coeff) pairs.
    """
    if not poly:
        return 0
    # Use dictionary for sparse convolution
    sq = {}
    for i, (ei, ci) in enumerate(poly):
        for j in range(i, len(poly)):
            ej, cj = poly[j]
            key = ei + ej
            if i == j:
                sq[key] = sq.get(key, 0) + ci * ci
            else:
                sq[key] = sq.get(key, 0) + 2 * ci * cj
    # Count nonzero
    return sum(1 for v in sq.values() if v != 0)

def _validate(poly):
    """Check if poly meets all constraints."""
    if len(poly) != 12:
        return False
    if any(c == 0 for _, c in poly):
        return False
    if any(c < -64 or c > 64 for _, c in poly):
        return False
    if any(e < 0 or not isinstance(e, int) for e, _ in poly):
        return False
    if any(e > 40 for e, _ in poly):
        return False
    # Check for duplicate exponents
    exps = [e for e, _ in poly]
    if len(set(exps)) != 12:
        return False
    return True

def _optimize_coefficients(exps, max_iters=50, lr=0.5):
    """
    Optimizes coefficients for a fixed set of exponents to minimize
    the L1 norm of the square's coefficients.
    Returns the best integer coefficients found.
    """
    n = len(exps)
    # Initialize with random signs or 1s
    coeffs = np.ones(n)
    # Try a few random initializations to avoid local minima
    best_coeffs = None
    best_l1 = float('inf')
    
    for init_trial in range(5):
        if init_trial > 0:
            coeffs = np.random.choice([-1, 1, -2, 2], size=n)
        
        curr_coeffs = coeffs.copy()
        curr_l1 = float('inf')
        
        for it in range(max_iters):
            # Build dense vector for convolution
            max_deg = max(exps)
            vec = np.zeros(max_deg + 1, dtype=np.float64)
            for i, e in enumerate(exps):
                vec[e] = curr_coeffs[i]
            
            # Compute square coefficients
            sq = np.convolve(vec, vec)
            
            # Compute L1 norm
            l1 = np.sum(np.abs(sq))
            
            if l1 < curr_l1:
                curr_l1 = l1
                # Update best
                if l1 < best_l1:
                    best_l1 = l1
                    best_coeffs = curr_coeffs.copy()
            
            # Compute gradient of L1 norm w.r.t coefficients
            # d/d a_k sum |c_m| = sum_m sign(c_m) * d c_m / d a_k
            # c_m = sum_{i,j} a_i a_j delta_{e_i+e_j, m}
            # d c_m / d a_k = a_k delta_{2e_k, m} + 2 sum_{j != k} a_j delta_{e_k+e_j, m}
            
            grad = np.zeros(n)
            # Efficiently compute gradient
            # For each pair (i, j), contribution to c_{e_i+e_j} is a_i a_j (or 2 a_i a_j)
            # We need sum over m of sign(c_m) * (derivative)
            
            # Let's just do it directly for n=12, it's fast enough
            # c_m values are in sq array
            for k in range(n):
                g = 0.0
                ek = exps[k]
                # Term 2*ek
                m1 = 2 * ek
                if m1 < len(sq):
                    c_val = sq[m1]
                    if c_val != 0:
                        g += np.sign(c_val) * curr_coeffs[k]
                    else:
                        # Subgradient, pick 0 or random
                        g += 0.0 
                # Cross terms
                for j in range(n):
                    if j == k:
                        continue
                    ej = exps[j]
                    m2 = ek + ej
                    if m2 < len(sq):
                        c_val = sq[m2]
                        if c_val != 0:
                            g += np.sign(c_val) * 2 * curr_coeffs[j]
                grad[k] = g
            
            # Gradient descent step
            curr_coeffs -= lr * grad
            # Clip to [-64, 64]
            curr_coeffs = np.clip(curr_coeffs, -64, 64)
            
            # Optional: random perturbation to escape
            if random.random() < 0.1:
                idx = random.randint(0, n-1)
                curr_coeffs[idx] += random.uniform(-1, 1)
                curr_coeffs = np.clip(curr_coeffs, -64, 64)
                
    # Round to integers
    if best_coeffs is None:
        return [1] * n
        
    int_coeffs = [int(np.round(c)) for c in best_coeffs]
    # Ensure non-zero
    for i in range(n):
        if int_coeffs[i] == 0:
            int_coeffs[i] = 1 if best_coeffs[i] >= 0 else -1
            
    return int_coeffs

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    best_poly = None
    best_score = float('inf')
    
    # Initial candidate: consecutive exponents with all 1s
    seed_poly = [(i, 1) for i in range(12)]
    seed_score = _square_nonzero_count_sparse(seed_poly)
    if _validate(seed_poly):
        best_poly = seed_poly
        best_score = seed_score
    
    # Simulated annealing with structural moves
    random.seed(42)
    np.random.seed(42)
    
    def _evaluate(poly):
        return _square_nonzero_count_sparse(poly)
    
    def _sa_optimize(initial_poly, max_iters=50000):
        """Simulated annealing to minimize nonzero terms in square."""
        current = list(initial_poly)
        current_score = _evaluate(current)
        best = list(current)
        best_score_local = current_score
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.05
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Cooled temperature
            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress
            
            # Generate a neighbor
            move_type = random.random()
            neighbor = list(current)
            
            if move_type < 0.45:
                # Change one exponent
                i = random.randint(0, 11)
                old_exp = neighbor[i][0]
                if random.random() < 0.6:
                    offset = random.choice([-3, -2, -1, 1, 2, 3, -5, 5])
                    new_exp = old_exp + offset
                    if new_exp < 0 or new_exp > 40:
                        continue
                else:
                    new_exp = random.randint(0, 40)
                if new_exp == old_exp:
                    continue
                if any(neighbor[j][0] == new_exp for j in range(12) if j != i):
                    continue
                neighbor[i] = (new_exp, neighbor[i][1])
                
                # Re-optimize coefficients for this new exponent set
                exps = [p[0] for p in neighbor]
                # Use current coeffs as init? Or random?
                # Let's use the current coeffs as a warm start for the optimizer
                # But the optimizer takes exps and returns coeffs
                # We need to pass the current coeffs to the optimizer to help it
                # Modify _optimize_coefficients to accept init_coeffs
                # For now, just call it. It does random inits.
                # To speed up, we can pass the current coeffs.
                # Let's create a wrapper that uses current coeffs as one of the inits.
                
                new_coeffs = _optimize_coefficients(exps, max_iters=20, lr=0.5)
                for j in range(12):
                    neighbor[j] = (neighbor[j][0], new_coeffs[j])
                    
            elif move_type < 0.75:
                # Change one coefficient (local search)
                i = random.randint(0, 11)
                # Try small changes
                delta = random.choice([-2, -1, 1, 2])
                new_coeff = neighbor[i][1] + delta
                if new_coeff == 0 or new_coeff < -64 or new_coeff > 64:
                    continue
                neighbor[i] = (neighbor[i][0], new_coeff)
            else:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                ei, ci = neighbor[i]
                ej, cj = neighbor[j]
                neighbor[i] = (ej, ci)
                neighbor[j] = (ei, cj)
                
                # Re-optimize coefficients
                exps = [p[0] for p in neighbor]
                new_coeffs = _optimize_coefficients(exps, max_iters=20, lr=0.5)
                for k in range(12):
                    neighbor[k] = (neighbor[k][0], new_coeffs[k])
            
            if not _validate(neighbor):
                continue
            
            neighbor_score = _evaluate(neighbor)
            delta = neighbor_score - current_score
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current = neighbor
                current_score = neighbor_score
                
                if current_score < best_score_local:
                    best_score_local = current_score
                    best = list(current)
            
            if best_score_local <= 12:
                break
        
        return best, best_score_local
    
    # Multiple random restarts with simulated annealing
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial polynomials
        if r == 0:
            initial = list(seed_poly)
        else:
            # Random exponents
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            initial = list(zip(exps, coeffs))
            
        if not _validate(initial):
            continue
        
        score = _evaluate(initial)
        if score < best_score:
            best_score = score
            best_poly = list(initial)
        
        # Run simulated annealing
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break
        # Allocate iterations based on remaining time
        max_iters = int(remaining * 500)
        local_poly, local_score = _sa_optimize(initial, max_iters=max(1000, max_iters))
        if local_score < best_score:
            best_score = local_score
            best_poly = local_poly
        
        if best_score <= 12:
            break
    
    # Final local improvement on best found
    if best_poly is not None and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 3:
            # One more pass with more iterations
            final_poly, final_score = _sa_optimize(best_poly, max_iters=int(remaining * 1000))
            if final_score < best_score:
                best_score = final_score
                best_poly = final_poly
    
    # Convert to coefficient list
    if best_poly is None:
        best_poly = seed_poly
    
    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff
    
    return coeff_list
