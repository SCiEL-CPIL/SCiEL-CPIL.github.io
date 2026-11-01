"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # safety margin before 150s kill
    
    # Build schoolbook decomposition
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Compute full error vector for all (i,j,k)
    def compute_errors(U, V, W):
        """Return array of shape (n, m, p) with error at each (i,j,k)."""
        rank = U.shape[1]
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    errors[i, j, k] = np.dot(u_row, v_row * w_row) - 1.0
        return errors

    def max_abs_error(errors):
        return float(np.max(np.abs(errors)))

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not verify(best_U, best_V, best_W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Candidate values for entries - expanded range
    candidate_vals = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=float)
    
    np.random.seed(42)
    
    found_lower = False
    
    # Try rank 41, then 40, etc.
    for rank_to_try in range(best_rank - 1, max(20, best_rank - 12), -1):
        if time.time() > deadline - 10:
            break
        
        attempts_for_rank = 0
        max_attempts = 120
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            attempts_for_rank += 1
            
            # Random initialization with small values, mostly zeros
            U_rand = np.zeros((n * m, rank_to_try))
            V_rand = np.zeros((m * p, rank_to_try))
            W_rand = np.zeros((p * n, rank_to_try))
            
            # Fill with random candidate values, but keep sparsity
            for mat, shape in [(U_rand, (n*m, rank_to_try)), 
                                (V_rand, (m*p, rank_to_try)), 
                                (W_rand, (p*n, rank_to_try))]:
                # Each entry has 65% chance of being 0
                for row in range(shape[0]):
                    for col in range(shape[1]):
                        if np.random.random() < 0.35:
                            mat[row, col] = np.random.choice(candidate_vals)
            
            errors = compute_errors(U_rand, V_rand, W_rand)
            best_err = max_abs_error(errors)
            
            if best_err < 1e-9:
                if verify(U_rand, V_rand, W_rand):
                    best_rank = rank_to_try
                    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    found_lower = True
                    break
            
            # Local search with incremental error updates
            local_iters = 300
            for it in range(local_iters):
                if time.time() > deadline - 10:
                    break
                
                # Pick a random matrix and entry
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    mat = U_rand
                    row = np.random.randint(0, n * m)
                elif mat_idx == 1:
                    mat = V_rand
                    row = np.random.randint(0, m * p)
                else:
                    mat = W_rand
                    row = np.random.randint(0, p * n)
                
                col = np.random.randint(0, rank_to_try)
                old_val = mat[row, col]
                new_val = float(np.random.choice(candidate_vals))
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                
                # Apply the change
                mat[row, col] = new_val
                errors_new = compute_errors(U_rand, V_rand, W_rand)
                new_err = max_abs_error(errors_new)
                
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    errors = errors_new
                else:
                    mat[row, col] = old_val
            
            if best_err < 1e-9:
                if verify(U_rand, V_rand, W_rand):
                    best_rank = rank_to_try
                    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    found_lower = True
                    break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
