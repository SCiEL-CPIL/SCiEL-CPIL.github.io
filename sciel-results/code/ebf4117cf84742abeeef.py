"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def validate(U, V, W, rank_val):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Strategy 1: Try structured decomposition using Strassen-like ideas
    # For 2x2*2x2, Strassen rank 7. We can embed this.
    # For 3x3, we can use Laderman (rank 20 for 3x3*3x3).
    # For 3x3*3x8, try to build from smaller blocks.
    
    # Strategy 2: Random search with local improvement
    # Start from schoolbook and try to merge/eliminate terms
    
    # First, try a direct structured approach:
    # View B (3x8) as [B1 | B2] where B1, B2 are 3x4
    # C = A*B = [A*B1 | A*B2]
    # If we can decompose 3x3*3x4 with rank r, total is 2r
    # Schoolbook for 3x3*3x4 is 36, so 2*36=72 (same as before)
    # But maybe we can share: the A matrix is the same
    
    # Try to find a rank-r decomposition for 3x3*3x8 directly
    # Using the constraint that entries are in {0, ±1, ±0.5, ±1.5, ±2, ...}
    
    # Key insight: For exact rational decompositions, we can use the
    # following approach: fix U and V, solve for W (or vice versa)
    # The equations are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of bilinear equations. Let's try a more targeted search.
    
    # Approach: Try ranks from a reasonable lower bound upward
    # Lower bound: 24 (from max(n*m, m*p, p*n))
    # But realistically, for 3x3*3x8, the rank is probably between 40-72
    
    # Let's try a smarter random search:
    # 1. Initialize with small integer/half-integer values
    # 2. Use the structure: many entries should be 0 or ±1
    
    def random_decomposition(rank_val, sparsity=0.3):
        """Generate a random sparse decomposition."""
        vals = [0, 1, -1, 0.5, -0.5]
        U = [[random.choice(vals) if random.random() < (1-sparsity) else 0 
              for _ in range(rank_val)] for _ in range(n*m)]
        V = [[random.choice(vals) if random.random() < (1-sparsity) else 0 
              for _ in range(rank_val)] for _ in range(m*p)]
        W = [[random.choice(vals) if random.random() < (1-sparsity) else 0 
              for _ in range(rank_val)] for _ in range(p*n)]
        return U, V, W
    
    # Try various ranks, starting from a reasonable estimate
    # For 3x3*3x8, let's try from 40 to 71
    # Use time budget to allocate attempts across ranks
    
    remaining_budget = deadline - elapsed()
    
    # Allocate time: spend more time on lower ranks (more promising)
    # Try ranks 40-71
    ranks_to_try = list(range(40, 72))
    
    # Weight attempts: lower ranks get more attempts
    total_weight = sum(1.0/(r-39) for r in ranks_to_try)
    attempts_per_rank = {}
    for r in ranks_to_try:
        weight = 1.0/(r-39)
        attempts_per_rank[r] = max(5, int(500 * weight / total_weight))
    
    for target_rank in ranks_to_try:
        if elapsed() > deadline - 5:
            break
        
        max_attempts = attempts_per_rank.get(target_rank, 10)
        found = False
        
        for attempt in range(max_attempts):
            if elapsed() > deadline - 5:
                break
            
            U_r, V_r, W_r = random_decomposition(target_rank, sparsity=0.4)
            
            if validate(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # Strategy 3: Local search - start from a valid decomposition and try to reduce rank
    # by merging terms or finding alternative representations
    
    # Try a more systematic approach: use the fact that the tensor has structure
    # The matrix multiplication tensor for 3x3*3x8 can be seen as:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # Try to find a low-rank approximation that is exact
    # Using iterative refinement:
    
    def iterative_refinement(initial_rank, max_iter=100):
        """Try to reduce rank by local modifications."""
        if elapsed() > deadline - 10:
            return None, None, None, initial_rank
        
        # Start with schoolbook
        U, V, W, rank = schoolbook()
        
        # Try to eliminate terms by finding linear dependencies
        # For each column c, check if it can be expressed as a combination of others
        # This is complex; instead, try random perturbations and check
        
        best_U, best_V, best_W, best_r = U, V, W, rank
        
        for iter_num in range(max_iter):
            if elapsed() > deadline - 10:
                break
            
            # Try to remove one column and adjust others
            # Pick a random column to try to eliminate
            col_to_remove = random.randint(0, best_r - 1)
            
            # Create a new decomposition without this column
            new_rank = best_r - 1
            if new_rank < 24:
                break
            
            # Try to find new values for the remaining columns
            # that compensate for the removed column
            # This is a hard problem; use random search
            success = False
            for _ in range(50):
                if elapsed() > deadline - 10:
                    break
                
                # Randomly modify some entries
                U_new = [row[:] for row in best_U]
                V_new = [row[:] for row in best_V]
                W_new = [row[:] for row in best_W]
                
                # Remove column col_to_remove
                for row in range(n*m):
                    U_new[row] = U_new[row][:col_to_remove] + U_new[row][col_to_remove+1:]
                for row in range(m*p):
                    V_new[row] = V_new[row][:col_to_remove] + V_new[row][col_to_remove+1:]
                for row in range(p*n):
                    W_new[row] = W_new[row][:col_to_remove] + W_new[row][col_to_remove+1:]
                
                # Try small random adjustments
                for _ in range(10):
                    r_idx = random.randint(0, n*m-1)
                    c_idx = random.randint(0, new_rank-1)
                    U_new[r_idx][c_idx] += random.choice([-0.5, 0.5, -1, 1])
                    
                    r_idx = random.randint(0, m*p-1)
                    c_idx = random.randint(0, new_rank-1)
                    V_new[r_idx][c_idx] += random.choice([-0.5, 0.5, -1, 1])
                    
                    r_idx = random.randint(0, p*n-1)
                    c_idx = random.randint(0, new_rank-1)
                    W_new[r_idx][c_idx] += random.choice([-0.5, 0.5, -1, 1])
                
                if validate(U_new, V_new, W_new, new_rank):
                    best_U, best_V, best_W, best_r = U_new, V_new, W_new, new_rank
                    success = True
                    break
            
            if not success and iter_num > 50:
                break
        
        return best_U, best_V, best_W, best_r
    
    # Try iterative refinement if we have time
    if elapsed() < deadline - 30:
        U_ref, V_ref, W_ref, rank_ref = iterative_refinement(best_rank)
        if rank_ref < best_rank:
            best_U, best_V, best_W, best_rank = U_ref, V_ref, W_ref, rank_ref
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
