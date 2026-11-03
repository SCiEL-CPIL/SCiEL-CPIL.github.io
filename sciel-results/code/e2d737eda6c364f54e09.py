"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(n, m, p, U, V, W, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build tensor from decomposition
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_vec = U[:, c].reshape(n, m)
        v_vec = V[:, c].reshape(m, p)
        w_vec = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    # Target tensor is all ones
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=1e-10):
        return False
    return True

def _try_random_decomposition(n, m, p, rank, seed=0):
    """Try to find a random integer/half-integer decomposition of given rank."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for attempt in range(50):
        # Initialize with small random integers/half-integers
        vals = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
        U = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)])
        
        if _verify_decomposition(n, m, p, U.tolist(), V.tolist(), W.tolist(), rank):
            return U, V, W
    return None

def _try_structured_decomposition(n, m, p):
    """Try structured approaches to find lower rank."""
    # Approach: use the fact that for each j, the n×p slice is all-ones (rank 1)
    # This suggests we might be able to share structure across j values.
    
    # Let's try: for each j, we need n*p terms where U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1
    # But we can potentially reuse columns across different j if the patterns align.
    
    # Try a block-structured approach
    # For 3x3x7: n=3, m=3, p=7
    # Each j-slice is 3x7 all-ones, which has rank 1 as a matrix
    # So for each j, we need 1 "effective" rank if we could separate cleanly
    
    # Let's try: assign each (i,j,k) to a column, but try to merge columns
    # where possible by using non-1 values.
    
    # Simple approach: just return schoolbook if nothing better found
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(21, 21, 9) = 21
    # But that's for the matrix rank, not tensor rank. Tensor rank is higher.
    
    # Try ranks from lower bound up to schoolbook
    # For 3x3x7, let's try a range of ranks
    min_rank_to_try = 25  # start trying from here
    max_rank_to_try = min(rank_schoolbook - 1, 55)
    
    seed_counter = 0
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 30:  # safety margin
            break
        
        # Try multiple random seeds for this rank
        for seed in range(seed_counter, seed_counter + 20):
            if time.time() > deadline - 30:
                break
            
            result = _try_random_decomposition(n, m, p, rank, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to list format with exact integer/half-integer values
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify once more
                if _verify_decomposition(n, m, p, U_list, V_list, W_list, rank):
                    best_rank = rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    # Found a valid decomposition, break to try even lower
                    break
        
        if best_rank < max_rank_to_try:
            # Found something, continue trying lower
            continue
        else:
            # No improvement at this rank, try next (higher) rank
            seed_counter += 20
    
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
