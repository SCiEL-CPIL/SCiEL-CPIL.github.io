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
    # Approach: Use random initialization + refinement
    
    import random
    random.seed(12345)
    
    def _random_candidate(rank):
        """Generate a random candidate with given rank."""
        U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W
    
    # Try a few ranks below 60
    for target_rank in range(best_rank - 1, max(29, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        # Try a few random candidates
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U, V, W = _random_candidate(target_rank)
            
            # Quick check: verify a few entries
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Do full verification
                if _verify_rank(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
    
    # Return the best found
    return best

# EVOLVE-BLOCK-END
