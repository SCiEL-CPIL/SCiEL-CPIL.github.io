import numpy as np
from scipy.optimize import linprog
import itertools

DIMENSIONS = [3, 4, 6]

def _build_schoolbook(n, m, p):
    """Return U, V, W as lists of lists (row-major) for the schoolbook rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if this is the (i,j,k) entry of matmul tensor
    """
    rank = len(U[0])
    # The matmul tensor T[i,j,k] = 1 if we're looking at entry (i,k) of AB where B has row j
    # Actually, the tensor for matmul (n,m,p) has indices (i, j, k) where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (it's the identity tensor)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _try_rank_decomposition(n, m, p, target_rank, max_time=60):
    """Try to find a rank-decomposition using integer/half-integer coefficients.
    
    We use a randomized approach: randomly initialize U, V, W with small integers,
    then try to solve for one factor at a time using linear algebra.
    """
    import time
    start_time = time.time()
    
    num_rows_U = n * m
    num_rows_V = m * p
    num_rows_W = p * n
    
    # Number of constraints: n*m*p (one for each tensor entry)
    num_constraints = n * m * p
    
    # We'll try to find a decomposition where one of the factors is fixed
    # and we solve for the others.
    
    # Strategy: Fix W to be random with values in {-1, 0, 1}, then solve for U and V.
    # But that's a bilinear problem. Instead, let's try a different approach.
    
    # Approach: Use the fact that for each column r of the decomposition,
    # we need U_col, V_col, W_col such that outer product matches part of the tensor.
    # This is hard. Let's try a simpler heuristic.
    
    # Heuristic: Start from schoolbook and try to merge columns.
    # Two schoolbook columns (i1,j1,k1) and (i2,j2,k2) can be merged if...
    # Actually, this is complex. Let's try a different approach.
    
    # Approach: Random search with local optimization.
    # Initialize U, V, W randomly with values in {-1, 0, 1, 0.5, -0.5}
    # Then iteratively optimize.
    
    # For now, let's just try to find if rank < n*m*p is achievable by checking
    # if there's a known decomposition. For 3x4x6, let's try rank 60, 64, etc.
    
    # Actually, let's use a smarter approach: exploit structure.
    # For 3x4x6, we can view it as:
    # - 3x4x6 = 3x(4x6) or (3x4)x6
    # - We can try to use the fact that 4 = 2+2 and use Strassen for the 2x2 blocks
    
    # Let's try a block decomposition:
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x6) into B1 (2x6) and B2 (2x6)
    # C = A1*B1 + A2*B2
    
    # For each block, we have a 3x2x6 multiplication.
    # Rank of 3x2x6 schoolbook = 3*2*6 = 36
    # Total rank = 36 + 36 = 72 (same as schoolbook)
    
    # To do better, we need to find a decomposition of 3x2x6 with rank < 36.
    # Or find a way to share computations between the two blocks.
    
    # Let's try to find a rank decomposition for 3x2x6 first.
    # n=3, m=2, p=6, schoolbook rank = 36
    
    # For 2x2x2, Strassen gives rank 7 < 8.
    # For 3x2x6, can we do better than 36?
    
    # Let's try to construct a decomposition using the following idea:
    # The tensor for 3x2x6 has 3*2*6 = 36 non-zero entries.
    # We want to express it as a sum of rank-1 tensors.
    
    # One approach: use the fact that 6 = 2+2+2 and decompose into three 3x2x2 blocks.
    # Each 3x2x2 block has schoolbook rank 12.
    # Total: 3*12 = 36 (same as before).
    
    # To do better, we need to find a decomposition of 3x2x2 with rank < 12.
    # 3x2x2: n=3, m=2, p=2, schoolbook = 12.
    # Can we do better? The tensor rank of 3x2x2 matmul...
    
    # Actually, for small matrices, the best known ranks are:
    # 2x2x2: 7 (Strassen)
    # 3x3x3: 23 (best known)
    # 2x2x3: ?
    # 3x2x2: ?
    
    # Let's try to find a rank decomposition for 3x2x2.
    # n=3, m=2, p=2. Schoolbook rank = 12.
    
    # The tensor T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}.
    # We want T = sum_r U_r ⊗ V_r ⊗ W_r where:
    # U_r is a vector of length 6 (n*m = 3*2 = 6)
    # V_r is a vector of length 4 (m*p = 2*2 = 4)  
    # W_r is a vector of length 6 (p*n = 2*3 = 6)
    
    # This is a 6x4x6 tensor with all entries = 1.
    # Wait, no. The matmul tensor has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    # So the tensor is actually the "identity" in the sense that each (i,j,k) triple contributes 1.
    
    # For 3x2x2:
    # U has 3*2 = 6 rows
    # V has 2*2 = 4 rows
    # W has 2*3 = 6 rows
    
    # Let's try to find a rank-10 or rank-11 decomposition for 3x2x2.
    
    # Actually, let me just try a random search for a lower-rank decomposition.
    
    # For the overall 3x4x6 problem, let's try to use a combination of:
    # 1. Strassen-like tricks on sub-blocks
    # 2. Direct construction
    
    # Let's try a specific construction:
    # View 3x4x6 as combining a 3x2x6 and another 3x2x6.
    # For each 3x2x6, try to find a decomposition with rank < 36.
    
    # Alternatively, view 3x4x6 as 3x(2+2)x6 = (3x2x6) + (3x2x6).
    
    # Let's try to find a rank decomposition for 3x2x6 with rank 30 or so.
    
    # For now, let's just return schoolbook if we can't find something better.
    # But let's try a few random searches.
    
    # Random search for 3x4x6 with target rank
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() - start_time > max_time - 5:
            break
        
        # Randomly initialize with small integers
        U_rand = np.random.choice([-1, 0, 1], size=(num_rows_U, target_rank))
        V_rand = np.random.choice([-1, 0, 1], size=(num_rows_V, target_rank))
        W_rand = np.random.choice([-1, 0, 1], size=(num_rows_W, target_rank))
        
        # Check if this works
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_rand[i*m+j] * V_rand[j*p+k] * W_rand[k*n+i])
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition than schoolbook
    # Schoolbook rank is n*m*p = 72
    
    # Try a few target ranks below 72
    for target_rank in [70, 68, 66, 64, 60, 56]:
        result = _try_rank_decomposition(n, m, p, target_rank, max_time=10)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
