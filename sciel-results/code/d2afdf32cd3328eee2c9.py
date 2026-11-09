"""Program that searches for a lower-rank tensor decomposition of matrix multiplication 3x5x7."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n=3, m=5, p=7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 5, 7
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the error tensor E[i,j,k] = 1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r]."""
    rank = len(U[0])
    error = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                error[i, j, k] = 1.0 - s
    return error

def _max_abs_error(U, V, W, n, m, p):
    """Return the maximum absolute error."""
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                err = abs(1.0 - s)
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for evaluation overhead
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 105
    
    # Start with schoolbook decomposition
    U_school = [[0] * base_rank for _ in range(n * m)]
    V_school = [[0] * base_rank for _ in range(m * p)]
    W_school = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = base_rank

    # Local search for lower rank
    # We'll try to reduce the rank by merging columns or using structured patterns
    
    def try_local_search(target_rank, max_iterations=500):
        """Try to find a decomposition of target_rank using local search."""
        if time.time() > deadline - 15:
            return None
        
        # Initialize: start from schoolbook and try to merge columns
        # Or start from a random sparse initialization
        
        # Strategy: Start with target_rank columns, each being a random combination
        # of the schoolbook columns, then optimize
        
        # Alternative: Greedy construction
        # Build up the decomposition column by column
        
        # Let's try a different approach: 
        # Use the fact that we can represent the tensor as a sum of rank-1 tensors
        # Each rank-1 tensor is U_r ⊗ V_r ⊗ W_r where:
        #   U_r[i*m+j] = u_i * v_j (outer product structure)
        #   V_r[j*p+k] = v_j * w_k
        #   W_r[k*n+i] = w_k * u_i
        # This gives: sum_r (u_i * v_j) * (v_j * w_k) * (w_k * u_i) = sum_r u_i^2 * v_j^2 * w_k^2
        # We need this to equal 1 for all i,j,k.
        
        # This is a special case. Let's try a more general approach.
        
        # Random initialization with small integers
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Initialize with sparse random values
        for r in range(target_rank):
            # Pick a few non-zero entries for each factor
            num_nonzero_u = random.randint(1, min(5, n * m))
            num_nonzero_v = random.randint(1, min(5, m * p))
            num_nonzero_w = random.randint(1, min(5, p * n))
            
            u_indices = random.sample(range(n * m), num_nonzero_u)
            v_indices = random.sample(range(m * p), num_nonzero_v)
            w_indices = random.sample(range(p * n), num_nonzero_w)
            
            for idx in u_indices:
                U[idx][r] = random.choice([-1, 1])
            for idx in v_indices:
                V[idx][r] = random.choice([-1, 1])
            for idx in w_indices:
                W[idx][r] = random.choice([-1, 1])
        
        current_max_err = _max_abs_error(U, V, W, n, m, p)
        
        # Local search: try to reduce error by adjusting individual entries
        for iteration in range(max_iterations):
            if time.time() > deadline - 15:
                break
            
            if current_max_err < 1e-9:
                return {"U": U, "V": V, "W": W}
            
            # Try to improve by adjusting one entry at a time
            improved = False
            for _ in range(100):  # Try 100 random moves per iteration
                # Randomly select which factor and which entry to modify
                factor = random.choice(['U', 'V', 'W'])
                
                if factor == 'U':
                    idx = random.randint(0, n * m - 1)
                    r = random.randint(0, target_rank - 1)
                    old_val = U[idx][r]
                    # Try changing to 0, 1, -1, 2, -2
                    new_val = random.choice([0, 1, -1, 2, -2, 3, -3])
                    if new_val == old_val:
                        continue
                    U[idx][r] = new_val
                    new_err = _max_abs_error(U, V, W, n, m, p)
                    if new_err < current_max_err:
                        current_max_err = new_err
                        improved = True
                    else:
                        U[idx][r] = old_val
                
                elif factor == 'V':
                    idx = random.randint(0, m * p - 1)
                    r = random.randint(0, target_rank - 1)
                    old_val = V[idx][r]
                    new_val = random.choice([0, 1, -1, 2, -2, 3, -3])
                    if new_val == old_val:
                        continue
                    V[idx][r] = new_val
                    new_err = _max_abs_error(U, V, W, n, m, p)
                    if new_err < current_max_err:
                        current_max_err = new_err
                        improved = True
                    else:
                        V[idx][r] = old_val
                
                else:  # W
                    idx = random.randint(0, p * n - 1)
                    r = random.randint(0, target_rank - 1)
                    old_val = W[idx][r]
                    new_val = random.choice([0, 1, -1, 2, -2, 3, -3])
                    if new_val == old_val:
                        continue
                    W[idx][r] = new_val
                    new_err = _max_abs_error(U, V, W, n, m, p)
                    if new_err < current_max_err:
                        current_max_err = new_err
                        improved = True
                    else:
                        W[idx][r] = old_val
            
            if not improved:
                # Stuck, try a random restart
                for r in range(target_rank):
                    num_nonzero_u = random.randint(1, min(5, n * m))
                    num_nonzero_v = random.randint(1, min(5, m * p))
                    num_nonzero_w = random.randint(1, min(5, p * n))
                    
                    u_indices = random.sample(range(n * m), num_nonzero_u)
                    v_indices = random.sample(range(m * p), num_nonzero_v)
                    w_indices = random.sample(range(p * n), num_nonzero_w)
                    
                    for idx in u_indices:
                        U[idx][r] = random.choice([-1, 1])
                    for idx in v_indices:
                        V[idx][r] = random.choice([-1, 1])
                    for idx in w_indices:
                        W[idx][r] = random.choice([-1, 1])
                
                current_max_err = _max_abs_error(U, V, W, n, m, p)
        
        if current_max_err < 1e-9:
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try to find lower rank decompositions
    # Start from base_rank - 1 and go down
    for target_rank in range(base_rank - 1, max(30, base_rank - 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random restarts
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            result = try_local_search(target_rank, max_iterations=200)
            if result is not None:
                # Verify the decomposition
                if _validate_decomposition(result['U'], result['V'], result['W']):
                    best_result = result
                    best_rank = target_rank
                    break
        
        if best_rank < base_rank:
            break
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above. This block is kept for structural compatibility.
# EVOLVE-BLOCK-END
