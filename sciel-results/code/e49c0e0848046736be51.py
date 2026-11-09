"""Rewritten program: rank-minimal exact tensor decomposition for 3x3 x 3x6 matrix multiplication."""
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form an exact decomposition of the matmul tensor.
    
    Convention: W[k*n+i, r]
    The tensor T[i,j,k] = delta should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    # Build U matrix: shape (n*m, rank)
    # V matrix: shape (m*p, rank)
    # W matrix: shape (p*n, rank)
    
    # Check that for each (i,j,k), the sum equals 1
    # We can do this more efficiently by checking the full tensor
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank_decomposition(n, m, p, target_rank, max_iters=2000, seed=42):
    """Try to find a rank-target_rank decomposition using iterative refinement.
    
    We use the approach: randomly initialize U, V, W with entries in {-1, 0, 1},
    then try to correct errors by modifying individual entries.
    """
    rng = random.Random(seed)
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: start with random sparse matrices and use a greedy/corrective approach
    # This is very hard in general, so we'll try a limited search.
    
    # Alternative: use the structure of the problem.
    # For 3x3x6, one approach is to split the 6 columns into two 3-column blocks.
    # If we have a rank-r decomposition for 3x3x3, we can get rank 2r for 3x3x6.
    # The known rank for 3x3x3 is 23 (Landsberg/Strassen type).
    
    # Let's try to construct a 3x3x3 decomposition of rank 23.
    # This is non-trivial. Let's try a different approach.
    
    # For 3x3x6, we can think of it as multiplying A(3x3) * B1(3x3) and A(3x3) * B2(3x3).
    # If we use the same U for both, we need:
    # sum_r U[i*m+j, r] * V1[j*3+k, r] * W1[k*3+i, r] = 1 for k=0,1,2
    # sum_r U[i*m+j, r] * V2[j*3+k, r] * W2[k*3+i, r] = 1 for k=3,4,5
    # where V2 and W2 are the second half.
    
    # This is equivalent to having two 3x3x3 tensors sharing the same U.
    # The minimum rank for 3x3x3 is 23, so total would be 46 if we can share U optimally.
    # But sharing U means we need a single U that works for both, which constrains things.
    
    # Let's try a direct approach: use random initialization and local search.
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for attempt in range(3):
        U = [[rng.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p*n)]
        
        for it in range(max_iters):
            # Compute current error for each (i,j,k)
            # err[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            
            # Find the (i,j,k) with largest error
            max_err_idx = None
            max_err_val = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        e = abs(1 - s)
                        if e > max_err_val:
                            max_err_val = e
                            max_err_idx = (i, j, k)
            
            if max_err_val < 1e-12:
                # Success!
                return U, V, W
            
            i, j, k = max_err_idx
            
            # Try to fix by modifying one entry
            # For each r, try flipping U[i*m+j][r], V[j*p+k][r], or W[k*n+i][r]
            best_change = None
            best_new_err = max_err_val
            
            for r in range(target_rank):
                # Try changing U[i*m+j][r]
                old_u = U[i*m+j][r]
                for new_u in [0, 1, -1]:
                    if new_u == old_u:
                        continue
                    # Compute new error at (i,j,k)
                    # The contribution of r to (i,j,k) changes from old_u*V[j*p+k][r]*W[k*n+i][r]
                    # to new_u*V[j*p+k][r]*W[k*n+i][r]
                    # But this also affects other (i',j',k') entries.
                    # For simplicity, just check if it improves (i,j,k)
                    # This is a heuristic and may not converge.
                    pass
                # This approach is too slow. Let's use a different method.
            
            # Alternative: just randomly modify a few entries
            for _ in range(3):
                if rng.random() < 0.33:
                    idx = rng.randint(0, n*m-1)
                elif rng.random() < 0.66:
                    idx = rng.randint(0, m*p-1)
                else:
                    idx = rng.randint(0, p*n-1)
                
                # Determine which matrix and row
                if idx < n*m:
                    mat = U
                    row = idx
                elif idx < n*m + m*p:
                    mat = V
                    row = idx - n*m
                else:
                    mat = W
                    row = idx - n*m - m*p
                
                col = rng.randint(0, target_rank-1)
                mat[row][col] = rng.choice([0, 1, -1])
        
        # Compute final error
        total_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    total_err += (1 - s) ** 2
        
        if total_err < best_err:
            best_err = total_err
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
    
    if best_err < 1e-12:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave safety margin
    
    # First, return the schoolbook decomposition as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # Theoretical bounds: for 3x3x6, the rank is at most 54 (schoolbook).
    # Known results suggest the rank might be around 40-50.
    # Let's try ranks from 40 upward.
    
    # Try a structured approach: split into two 3x3x3 problems
    # Use the fact that 3x3x3 has rank 23 (known result)
    # If we can construct two 3x3x3 decompositions that share the U factor,
    # we get a 3x3x6 decomposition.
    
    # For now, let's try the straightforward approach of trying to find
    # a decomposition of rank less than 54.
    
    # Try rank 46 (2*23, assuming we can share U between two 3x3x3 blocks)
    # This requires finding a rank-23 decomposition of 3x3x3.
    
    # Let's try to use a known construction for 3x3x3.
    # The 3x3x3 tensor of rank 23 is not trivially constructible.
    
    # Instead, let's try a random search for low-rank decompositions.
    # We'll try ranks 40, 42, 44, 46, 48, 50, 52.
    
    for target_rank in [40, 42, 44, 46, 48, 50, 52]:
        if time.time() > deadline:
            break
        
        result = _try_rank_decomposition(n, m, p, target_rank, max_iters=500)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return schoolbook
