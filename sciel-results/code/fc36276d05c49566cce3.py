"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix using numerical optimization.
    """
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check a sample of zero entries
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i != i2 or j != j2 or k != k2:
                val = 0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val) > tol:
                    return False
        return True
    
    def full_verify(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True
    
    def try_rank(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition using alternating optimization."""
        if not check_time():
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Precompute the target tensor entries
        # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # And for inconsistent triples, sum should be 0
        
        for iteration in range(max_iter):
            if not check_time():
                break
            
            # Update U
            for a in range(n*m):
                i, j = a // m, a % m
                for r in range(r):
                    # Gradient of loss w.r.t. U[a,r]
                    grad = 0
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        target = 1.0
                        current = 0
                        for rr in range(r):
                            current += U[a, rr] * V[b, rr] * W[c, rr]
                        # Only this term contributes to U[a,r]
                        grad += (current - target) * V[b, r] * W[c, r]
                    # Also need to ensure zero entries are zero
                    # This is complex, so use a simplified update
                    # Use a step size
                    lr = 0.01
                    U[a, r] -= lr * grad
            
            # Update V
            for b in range(m*p):
                j, k = b // p, b % p
                for r in range(r):
                    grad = 0
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        target = 1.0
                        current = 0
                        for rr in range(r):
                            current += U[a, rr] * V[b, rr] * W[c, rr]
                        grad += (current - target) * U[a, r] * W[c, r]
                    lr = 0.01
                    V[b, r] -= lr * grad
            
            # Update W
            for c in range(p*n):
                k, i = c // n, c % n
                for r in range(r):
                    grad = 0
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        target = 1.0
                        current = 0
                        for rr in range(r):
                            current += U[a, rr] * V[b, rr] * W[c, rr]
                        grad += (current - target) * U[a, r] * V[b, r]
                    lr = 0.01
                    W[c, r] -= lr * grad
            
            # Periodic verification
            if iteration % 100 == 0:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if verify(U_list, V_list, W_list, tol=1e-4):
                    # Try to refine
                    if full_verify(U_list, V_list, W_list, tol=1e-6):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # First, get the schoolbook solution as fallback
    result = schoolbook()
    if not verify(result["U"], result["V"], result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    
    best_rank = n * m * p
    best_result = result
    
    # Try to find better ranks
    # Lower bound is max(n*m, m*p, p*n) = 24
    # Try ranks from 24 up to schoolbook
    for r in range(24, min(best_rank, 73)):
        if not check_time():
            break
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if not check_time():
                break
            res = try_rank(r, max_iter=3000, seed=seed)
            if res is not None:
                if full_verify(res["U"], res["V"], res["W"], tol=1e-6):
                    best_rank = r
                    best_result = res
                    found = True
                    break
        if found:
            # Try to go lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
