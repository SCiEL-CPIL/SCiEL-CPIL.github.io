"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal: T[a, b, c] = 0 unless indices align
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # Non-zero only if j==j', k==k', i==i'
    for a in range(n*m):
        for b in range(m*p):
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Try to find a lower-rank decomposition
    # Strategy: Use structured patterns and verify
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also, for bilinear maps, there are additional lower bounds
    
    # Let's try ranks from 60 down to 30
    # For each rank, try to construct a valid decomposition
    
    # First, let's try a simple structured approach for rank 60 (schoolbook)
    # Then try to improve
    
    # Key insight: For 2xm * mxp, we can think of it as:
    # Output[0,k] = sum_j A[0,j] * B[j,k]
    # Output[1,k] = sum_j A[1,j] * B[j,k]
    # 
    # If we could share some computation between the two rows, we could reduce rank.
    # But in exact decomposition, each (i,j,k) triple needs its own term unless
    # there's algebraic structure we can exploit.
    
    # Let's try a systematic search for rank < 60
    # We'll use a greedy approach: start with schoolbook and try to merge terms
    
    # For now, let's verify that schoolbook works
    U_test = best["U"]
    V_test = best["V"]
    W_test = best["W"]
    
    if _verify_rank(U_test, V_test, W_test):
        pass  # Schoolbook is valid
    else:
        # Should never happen, but just in case
        return best
    
    # Try to find a better decomposition
    # Approach: Use random initialization + refinement with vectorized checks
    
    import random
    random.seed(12345)
    
    # Define a larger set of possible values for integer/half-integer entries
    possible_values = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    def _random_candidate(rank):
        """Generate a random candidate with given rank."""
        U = np.array([[random.choice(possible_values) for _ in range(rank)] for _ in range(n*m)], dtype=float)
        V = np.array([[random.choice(possible_values) for _ in range(rank)] for _ in range(m*p)], dtype=float)
        W = np.array([[random.choice(possible_values) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        return U, V, W
    
    # Precompute indices for the "1" constraints
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We need to check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a sum of products of 3 vectors.
    # We can vectorize the check for all (i,j,k) at once?
    # U is (10, R), V is (30, R), W is (12, R)
    # We want to compute M[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # To speed up, we can just check the 60 "1" constraints first.
    # If they pass, we do the full check.
    
    def _check_ones(U, V, W, rank):
        """Check if the 60 diagonal entries are 1."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx]) # This is not right, it's sum of U*V*W
                    # Actually: sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    # We can compute this as:
                    # prod = U[u_idx] * V[v_idx]
                    # val = np.dot(prod, W[w_idx])
                    prod = U[u_idx] * V[v_idx]
                    val = np.dot(prod, W[w_idx])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Try ranks from 59 down to 30
    # We will try more attempts per rank
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        
        # Increase attempts as rank gets lower? Or keep constant?
        # Let's try 100 attempts per rank
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U, V, W = _random_candidate(target_rank)
            
            # Quick check: verify the 60 "1" entries
            if _check_ones(U, V, W, target_rank):
                # Do full verification
                if _verify_rank(U.tolist(), V.tolist(), W.tolist()):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break
    
    # Return the best found
    return best

# EVOLVE-BLOCK-END
