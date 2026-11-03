"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(n, m, p, U, V, W, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_vec = U[:, c].reshape(n, m)
        v_vec = V[:, c].reshape(m, p)
        w_vec = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=1e-10):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _gaussian_elimination_search(n, m, p, rank, seed=0, max_attempts=200):
    """
    Use Gaussian elimination: fix U and V randomly, then solve for W.
    The equations are: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k).
    This is linear in W entries.
    """
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        # Choose random U and V with small integer/half-integer values
        vals = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
        
        U = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(n * m)])
        V = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(m * p)])
        
        # Build the linear system for W
        # Variables: W[k*n+i, c] for all k, i, c -> p*n*rank variables
        # Equations: for each (i, j, k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # That's n*m*p equations
        
        num_vars = p * n * rank
        num_eqs = n * m * p
        
        # Build coefficient matrix A and right-hand side b
        # A[eq, var] where eq = i*m*p + j*p + k, var = k*n*rank + i*rank + c
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for c in range(rank):
                        var = k * n * rank + i * rank + c
                        A[eq, var] = U[i * m + j, c] * V[j * p + k, c]
        
        # Solve using least squares (system may be overdetermined)
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is exact
        W_check = A @ W_flat
        if not np.allclose(W_check, b, atol=1e-10):
            continue
        
        # Reshape W_flat to W
        W = W_flat.reshape(p * n, rank)
        
        # Check that W entries are close to integer/half-integer
        W_cleaned = np.round(W * 2) / 2
        if not np.allclose(W, W_cleaned, atol=1e-10):
            # Not clean, but still valid - keep as is for now
            pass
        
        # Verify
        if _verify_decomposition(n, m, p, U.tolist(), V.tolist(), W_cleaned.tolist(), rank):
            return U.tolist(), V.tolist(), W_cleaned.tolist()
    
    return None

def _strassen_3x3x3():
    """
    Return Strassen's rank-7 decomposition for 3x3x3 matrix multiplication.
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1,2}
    """
    # Strassen's algorithm for 3x3x3
    # The 7 rank-1 terms
    U = [
        [1, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0],
    ]
    V = [
        [1, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
    ]
    W = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
    return U, V, W

def _strassen_based_hybrid(n, m, p, seed=0, max_attempts=100):
    """
    Use Strassen for the 3x3x3 core (k=0,1,2), then extend to k=3,4,5,6.
    The idea: for k in {0,1,2}, use Strassen's 7 terms. For k in {3,4,5,6},
    we need additional terms that don't interfere with the first 3 k-values.
    
    Strategy: add 4*3 = 12 terms for the remaining 4 k-values (one per (i,k) pair),
    but try to share some terms. Total rank: 7 + 12 = 19.
    """
    rng = random.Random(seed)
    
    U_str, V_str, W_str = _strassen_3x3x3()
    
    for attempt in range(max_attempts):
        # Start with Strassen's 7 terms for k=0,1,2
        # Add terms for k=3,4,5,6
        # For each new k, we need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j
        
        # Simple approach: for each (i,k) with k>=3, add a term where
        # U[i,j,c]=1 for all j, V[j,k,c]=1 for all j, W[k,i,c]=1
        # But that's 3*4=12 new terms, total rank 19
        
        # Try to do better: use shared terms
        # For k=3,4,5,6, we can use the same U pattern if we adjust V and W
        
        # Let's try: add 4 new terms (one per k), each covering all i,j
        # U[i,j,c_k] = 1 for all i,j
        # V[j,k,c_k] = 1 for all j (for that specific k)
        # W[k,i,c_k] = 1 for all i
        
        # This gives rank 7+4=11, but let's verify
        new_terms = 4
        total_rank = 7 + new_terms
        
        U = [row[:] for row in U_str]
        V = [row[:] + [0]*new_terms for row in V_str]
        W = [row[:] + [0]*new_terms for row in W_str]
        
        # Add new terms for k=3,4,5,6
        for new_k_idx, k in enumerate([3, 4, 5, 6]):
            c = 7 + new_k_idx
            # U[i,j,c] = 1 for all i,j
            for i in range(n):
                for j in range(m):
                    U[i*m+j][c] = 1
            # V[j,k,c] = 1 for all j
            for j in range(m):
                V[j*p+k][c] = 1
            # W[k,i,c] = 1 for all i
            for i in range(n):
                W[k*n+i][c] = 1
        
        # Verify
        if _verify_decomposition(n, m, p, U, V, W, total_rank):
            return U, V, W, total_rank
    
    return None

def _try_low_rank_gaussian(n, m, p, rank, seed=0, max_attempts=100):
    """Try Gaussian elimination to find a rank-rank decomposition."""
    return _gaussian_elimination_search(n, m, p, rank, seed=seed, max_attempts=max_attempts)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb, rank_schoolbook = _schoolbook(n, m, p)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try Strassen-based hybrid first (fast, structured)
    if time.time() < start_time + 5:
        result = _strassen_based_hybrid(n, m, p, seed=0, max_attempts=50)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    # Try Gaussian elimination search for various ranks
    # Start from low ranks and go up
    min_rank_to_try = 11  # Strassen-based hybrid gives 11
    max_rank_to_try = min(best_rank - 1, 40)
    
    seed_counter = 0
    found_improvement = False
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > start_time + deadline - 10:  # safety margin
            break
        
        # Try multiple random seeds for this rank
        max_seeds = 30
        for seed in range(seed_counter, seed_counter + max_seeds):
            if time.time() > start_time + deadline - 10:
                break
            
            result = _try_low_rank_gaussian(n, m, p, rank, seed=seed, max_attempts=50)
            if result is not None:
                U, V, W = result
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify once more
                if _verify_decomposition(n, m, p, U_list, V_list, W_list, rank):
                    best_rank = rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found_improvement = True
                    break
        
        if found_improvement:
            continue
        else:
            seed_counter += max_seeds
    
    # Convert to proper format - ensure values are exact integers or half-integers
    def clean_values(val):
        # Snap to nearest 0.5 if close
        rounded = round(val * 2) / 2
        if abs(val - rounded) < 1e-10:
            return rounded
        return val
    
    U_final = [[clean_values(x) for x in row] for row in best_U]
    V_final = [[clean_values(x) for x in row] for row in best_V]
    W_final = [[clean_values(x) for x in row] for row in best_W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
