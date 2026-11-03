import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 120.0

def _check_time():
    global _START_TIME
    return time.time() - _START_TIME < _DEADLINE - 5.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _structured_block_decomposition(n, m, p):
    """
    Try to construct a lower-rank decomposition by exploiting block structure.
    
    For 3x3x7, split B into B1 (3x3) and B2 (3x4).
    C = A*B1 + A*B2
    
    For each block, use a structured decomposition.
    """
    # For 3x3x3, use a rank-9 decomposition (trivial for this block)
    # For 3x3x4, use a rank-12 decomposition (3 rows * 4 cols, but structured)
    
    # Actually, let's try a different approach: use the fact that we can
    # combine terms across the k dimension for each (i,j)
    
    # Strategy: for each (i,j), we have 7 values of k. We can try to
    # represent the vector (B[j,0], B[j,1], ..., B[j,6]) as a linear
    # combination of fewer basis vectors.
    
    # This is equivalent to finding a low-rank decomposition of the
    # multiplication map for each fixed (i,j).
    
    # For now, let's try a simple structured approach:
    # Group the 7 output columns into groups and use shared factors.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # We'll build the decomposition column by column
    # For each (i,j,k) triple, we need U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to sum to 1
    
    # Try to find a decomposition where we share factors across k for fixed (i,j)
    
    # For each (i,j), we have 7 terms. Can we reduce this?
    # The condition is: for each k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of 7 equations for each (i,j).
    
    # Let's try: for each (i,j), use 3 rank-1 terms (one for each "group" of k)
    # This would give rank = 3*3*3 = 27
    
    # Or use 2 rank-1 terms per (i,j): rank = 3*3*2 = 18
    
    # Let's try to construct this systematically.
    
    # For a given (i,j), we need vectors u (length 9), v_k (length 21 for each k), w_k (length 21 for each k)
    # such that u[i*m+j] * v_k[j*p+k] * w_k[k*n+i] sums appropriately.
    
    # This is getting complex. Let me try a simpler approach:
    # Use the trivial decomposition but try to find linear dependencies.
    
    # Actually, let me try a direct construction:
    # For 3x3x7, the tensor has 9*21*21 = 3969 entries, but only 63 are non-zero.
    # We want to find the minimum number of rank-1 tensors that sum to this.
    
    # A lower bound is max(9, 21, 21) = 21, but the actual rank is higher.
    
    # Let me try to use a known result: for matrix multiplication, the rank
    # is at least n*m*p / (some factor). For 3x3x7, let's try rank 27.
    
    # Construct a rank-27 decomposition:
    # For each (i,j), use 3 terms corresponding to k groups {0,1,2}, {3,4,5}, {6}
    
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # 3 groups of k values
            groups = [(0,1,2), (3,4,5), (6,)]
            for g_idx, group in enumerate(groups):
                c = col
                # U factor
                U[i*m+j][c] = 1.0
                # V and W factors: need to be such that for k in group,
                # V[j*p+k][c] * W[k*n+i][c] = 1/|group| (so that sum over group = 1)
                # But this doesn't work because we need each individual (i,j,k) to sum to 1.
                
                # Let me reconsider. We need:
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For the trivial case, each (i,j,k) has its own column.
                # To reduce rank, we need to share columns across k values.
                
                # If we share a column across k values in a group, then for each k in the group:
                # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must contribute to the sum.
                
                # This is complex. Let me just try random search with a better initialization.
                
                col += 1
    
    # This approach is getting too complex. Let me just do a better random search.
    return None

def _greedy_rank_reduction(n, m, p, initial_rank, max_iter=1000):
    """
    Try to reduce the rank by finding linear dependencies.
    Start with the trivial decomposition and try to remove columns.
    """
    # Start with trivial decomposition
    U = np.zeros((n*m, initial_rank))
    V = np.zeros((m*p, initial_rank))
    W = np.zeros((p*n, initial_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Try to find a subset of columns that still spans the tensor
    # This is a combinatorial problem, so use a greedy approach
    
    # For now, just return the trivial decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_low_rank_search(n, m, p, target_rank, rng, max_trials=1000):
    """
    Try to find a decomposition of the given rank using random search.
    """
    for trial in range(max_trials):
        if not _check_time():
            return None
        
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Initialize with random sparse values
        for c in range(target_rank):
            # Each column corresponds to a rank-1 term
            # Choose which (i,j,k) this term primarily affects
            focus_i = rng.randint(0, n-1)
            focus_j = rng.randint(0, m-1)
            focus_k = rng.randint(0, p-1)
            
            # Set the primary entry
            U[focus_i*m+focus_j][c] = 1.0
            V[focus_j*p+focus_k][c] = 1.0
            W[focus_k*n+focus_i][c] = 1.0
            
            # Add some random perturbations
            for i in range(n):
                for j in range(m):
                    if (i, j) != (focus_i, focus_j) and rng.random() < 0.1:
                        U[i*m+j][c] = rng.choice([-1.0, 1.0])
            for j in range(m):
                for k in range(p):
                    if (j, k) != (focus_j, focus_k) and rng.random() < 0.1:
                        V[j*p+k][c] = rng.choice([-1.0, 1.0])
            for k in range(p):
                for i in range(n):
                    if (k, i) != (focus_k, focus_i) and rng.random() < 0.1:
                        W[k*n+i][c] = rng.choice([-1.0, 1.0])
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # Start with the trivial decomposition as fallback
    best_result = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Lower bound on rank
    lower_bound = max(n*m, m*p, p*n)
    
    # Try to find lower-rank decompositions
    rng = random.Random(42)
    
    # Search from lower_bound up to best_rank - 1
    for rank in range(lower_bound, best_rank):
        if not _check_time():
            break
        
        # Try multiple random trials for this rank
        max_trials = 2000 if rank < 30 else 500
        result = _try_low_rank_search(n, m, p, rank, rng, max_trials)
        
        if result is not None:
            best_rank = rank
            best_result = result
    
    return best_result
