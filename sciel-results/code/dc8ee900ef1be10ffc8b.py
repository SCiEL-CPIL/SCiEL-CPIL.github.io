import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                # Standard index mapping:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = get_schoolbook()
    start_time = time.time()
    time_limit = 145.0

    # Precompute indices for fast error calculation
    # We need to check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Let's organize the constraints.
    # There are n*m*p constraints (60).
    # We can vectorize the check somewhat by precomputing the indices.
    
    # For each constraint (i, j, k), we have a vector of length rank:
    # vec_c = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need sum(vec_c) == 1.
    
    # To speed up local search, we can precompute partial products.
    # However, since rank is small (we aim for < 60), and constraints are 60,
    # a simple loop is likely fast enough if we use NumPy for the dot product.
    
    # Let's define a fast error function.
    # U, V, W are NumPy arrays.
    
    def compute_error(U, V, W):
        # Returns sum of squared errors
        # Efficient way:
        # For each c, we form outer products? No, that's rank * nm * mp * pn which is huge.
        # Instead, loop over constraints.
        # err = 0
        # for i,j,k:
        #   s = dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        #   err += (s-1)^2
        
        # Vectorized approach:
        # Create arrays of indices for all constraints.
        # This might be overhead per call.
        # Given 60 constraints, explicit loop is fine.
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Dot product of row u_idx of U with element-wise product of row v_idx of V and row w_idx of W
                    s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    diff = s - 1.0
                    err += diff * diff
        return err

    # Strategy:
    # 1. Try to find a solution with rank < 60.
    # 2. Use a "structured" initialization inspired by Strassen-like blocks?
    #    For 2x5x6, we can try to decompose it into smaller blocks.
    #    But random search with good initialization might work better.
    #    
    #    Let's try a hybrid:
    #    - Start with random small integers.
    #    - Use local search (Simulated Annealing or simple Hill Climbing) to reduce error.
    #    - If error becomes 0, verify exactness.
    
    # Candidate values: integers and half-integers.
    # Let's restrict to integers first for simplicity, or {-1, 0, 1}.
    # If that fails, try {-1, -0.5, 0, 0.5, 1}.
    
    rng = random.Random(42)
    
    # Try ranks from 30 down to 20? Or up?
    # Lower rank is harder. Let's try to find *any* rank < 60.
    # Known bounds for 2x5x6?
    # Rank >= 2*5 + 5*6 - 2*6 = 10 + 30 - 12 = 28? (Alon-Boppana bound is for square)
    # Actually, lower bound is max(n*m, m*p, p*n) = 30? No, that's not a lower bound for rank.
    # The trivial lower bound is max(n*m, m*p, p*n) is NOT correct.
    # The rank of matmul tensor for n x m * m x p is at least n*m, m*p, p*n? No.
    # It is at least n*p? No.
    # For square n x n, rank >= n^2.
    # For 2x5x6, schoolbook is 60.
    # Let's try ranks 30 to 50.
    
    # To improve chances, we can use a "warm start" from a lower dimension if applicable,
    # but 2,5,6 are specific.
    
    # Let's try a few random restarts for each rank.
    
    for target_rank in range(30, 60):
        if time.time() - start_time > time_limit - 5:
            break
            
        found = False
        # More restarts for lower ranks?
        # Let's do 20 restarts per rank.
        for restart in range(20):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Initialize with random integers -1, 0, 1
            U = np.random.randint(-1, 2, size=(nm, target_rank))
            V = np.random.randint(-1, 2, size=(mp, target_rank))
            W = np.random.randint(-1, 2, size=(pn, target_rank))
            
            # Local Search
            current_err = compute_error(U, V, W)
            
            # If initial error is already 0, check validity
            if current_err < 1e-10:
                if check_valid(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

            max_steps = 1000
            for step in range(max_steps):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Pick a random variable to change
                var = rng.randint(3)
                if var == 0:
                    idx = rng.randint(nm)
                    c = rng.randint(target_rank)
                    old_val = U[idx, c]
                    # Change by +/- 1
                    delta = rng.choice([-1, 1])
                    U[idx, c] = old_val + delta
                elif var == 1:
                    idx = rng.randint(mp)
                    c = rng.randint(target_rank)
                    old_val = V[idx, c]
                    delta = rng.choice([-1, 1])
                    V[idx, c] = old_val + delta
                else:
                    idx = rng.randint(pn)
                    c = rng.randint(target_rank)
                    old_val = W[idx, c]
                    delta = rng.choice([-1, 1])
                    W[idx, c] = old_val + delta
                    
                new_err = compute_error(U, V, W)
                
                # Accept if better, or with small probability (annealing)
                if new_err < current_err:
                    current_err = new_err
                else:
                    # Revert
                    if var == 0:
                        U[idx, c] = old_val
                    elif var == 1:
                        V[idx, c] = old_val
                    else:
                        W[idx, c] = old_val
                
                if current_err < 1e-10:
                    if check_valid(U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result

def check_valid(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-6:
                    return False
    # Check integer/half-integer
    # In our search, we started with ints and added ints, so they are ints.
    # If we used half-integers, we would need to check.
    # For now, assume integers are sufficient.
    return True
# EVOLVE-BLOCK-END
