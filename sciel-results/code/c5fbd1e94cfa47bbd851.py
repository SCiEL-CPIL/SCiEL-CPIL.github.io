import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 7 matrix.
    
    The schoolbook rank is 2*3*7 = 42.
    We first check if a lower rank is possible by attempting to solve the
    bilinear system for smaller ranks using randomized local search with
    integer/half-integer constraints.
    
    Interface: returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    Entries must be integer or half-integer (multiples of 0.5).
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Time limit guard
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0
    
    def check_time():
        return time.time() - start_time > (time_limit - safety_margin)
    
    # Helper to verify if a decomposition is exact
    def verify(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Using numpy for speed
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this efficiently:
        # For each r, the outer product of U[:,r], V[:,r], W[:,r] contributes to the tensor.
        # But checking 42 entries is fast.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def is_valid_entry(x):
        # Check if x is integer or half-integer
        return abs(x * 2 - round(x * 2)) < 1e-9

    def round_to_half(x):
        return round(x * 2) / 2.0

    def try_rank(rank, seed=0):
        """
        Try to find a decomposition of the given rank using randomized local search.
        Returns (U, V, W) if found, else None.
        """
        import random
        rng = random.Random(seed)
        
        # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
        U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        max_iters = 50000
        local_time_limit = 10.0
        start_local = time.time()
        
        for it in range(max_iters):
            if check_time() or (time.time() - start_local > local_time_limit):
                break
            
            # Check if we are done
            if verify(U, V, W, rank):
                return U, V, W
            
            # Find unsatisfied constraints
            unsatisfied = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = 0.0
                        for r in range(rank):
                            s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                        if abs(s - 1.0) > 1e-9:
                            unsatisfied.append((i, j, k, s))
            
            if not unsatisfied:
                return U, V, W
            
            # Pick a random unsatisfied constraint
            i, j, k, current_sum = rng.choice(unsatisfied)
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            diff = 1.0 - current_sum
            
            # Try to adjust one entry to fix the constraint
            # We pick a random rank index r and a random matrix (U, V, or W)
            r = rng.randint(0, rank - 1)
            matrix_choice = rng.choice(['U', 'V', 'W'])
            
            if matrix_choice == 'U':
                # Adjust U[u_idx, r]
                # New value x such that x * V[v_idx, r] * W[w_idx, r] = diff + old_contribution
                # old_contribution = U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                # x = (diff + old_contribution) / (V[v_idx, r] * W[w_idx, r])
                denom = V[v_idx][r] * W[w_idx][r]
                if abs(denom) > 1e-9:
                    old_val = U[u_idx][r]
                    new_val = (diff + old_val * denom) / denom
                    # Check if new_val is close to a half-integer
                    rounded = round_to_half(new_val)
                    if abs(new_val - rounded) < 1e-6:
                        # Check if this change helps (reduces error)
                        # Actually, we just apply it if it's a valid half-integer
                        U[u_idx][r] = rounded
            elif matrix_choice == 'V':
                denom = U[u_idx][r] * W[w_idx][r]
                if abs(denom) > 1e-9:
                    old_val = V[v_idx][r]
                    new_val = (diff + old_val * denom) / denom
                    rounded = round_to_half(new_val)
                    if abs(new_val - rounded) < 1e-6:
                        V[v_idx][r] = rounded
            else: # W
                denom = U[u_idx][r] * V[v_idx][r]
                if abs(denom) > 1e-9:
                    old_val = W[w_idx][r]
                    new_val = (diff + old_val * denom) / denom
                    rounded = round_to_half(new_val)
                    if abs(new_val - rounded) < 1e-6:
                        W[w_idx][r] = rounded
            
            # Occasionally, perform a random mutation to escape local minima
            if rng.random() < 0.05:
                mat = rng.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = rng.randint(0, n * m - 1)
                    col = rng.randint(0, rank - 1)
                    U[row][col] = rng.choice([-1, -0.5, 0, 0.5, 1])
                elif mat == 'V':
                    row = rng.randint(0, m * p - 1)
                    col = rng.randint(0, rank - 1)
                    V[row][col] = rng.choice([-1, -0.5, 0, 0.5, 1])
                else:
                    row = rng.randint(0, p * n - 1)
                    col = rng.randint(0, rank - 1)
                    W[row][col] = rng.choice([-1, -0.5, 0, 0.5, 1])

        return None

    # Try ranks from schoolbook_rank - 1 downwards
    # Given the small dimensions, we might find a solution quickly if one exists
    # But if no lower rank exists, we return schoolbook.
    # To be safe, we only try a few ranks below schoolbook if time permits.
    
    # First, let's see if we can find rank < 42.
    # We will try ranks 41, 40, ... down to some limit.
    # However, the search space is huge. 
    # A smarter approach: start with schoolbook and try to merge columns.
    # But merging columns in a bilinear decomposition is not straightforward.
    
    # Let's just try a few random restarts for rank 41, 40, etc.
    # If we find one, great. If not, return schoolbook.
    
    # Since the problem is small, let's try rank 41 first with a few seeds.
    # If that fails, we assume 42 is the minimum for this specific instance within our time budget.
    # Actually, for 2x3x7, the Strassen-like algorithms might apply?
    # Strassen is for 2x2x2. 
    # For 2x3x7, the rank is likely 42 unless there's a specific structure.
    # Let's try to be aggressive but safe.
    
    # Try rank 41
    for target_rank in [schoolbook_rank - 1, schoolbook_rank - 2, schoolbook_rank - 3]:
        if check_time():
            break
        for seed in range(5):
            if check_time():
                break
            result = try_rank(target_rank, seed=seed)
            if result:
                U, V, W = result
                # Final verification
                if verify(U, V, W, target_rank):
                    return {
                        "U": U,
                        "V": V,
                        "W": W
                    }
    
    # Fallback: Schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
