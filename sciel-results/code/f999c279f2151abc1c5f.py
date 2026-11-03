import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
_DEADLINE = 150

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U + V + W):
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_np[i * m + k, r] * V_np[k * p + j, r] * W_np[j * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_rank88():
    """Build the rank-88 decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = 3, 4, 8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Coefficients: (a_coeffs, b_coeffs, c_coeffs)
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the rank-96 schoolbook decomposition."""
    n, m, p = 3, 4, 8
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    return U, V, W

def _try_prune(U, V, W, n, m, p, max_time=30):
    """
    Try to prune the decomposition by finding linearly dependent rank terms.
    Returns a pruned decomposition if successful, else None.
    """
    start = time.time()
    rank = len(U[0])
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the constraint matrix: for each (i,k,j) triple, we need
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    
    # We can think of this as: for each r, the term is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # and we need the sum to equal the target tensor.
    
    # Strategy: try to find a subset of rank terms that still satisfies all constraints.
    # This is a combinatorial problem, so we'll use a greedy approach.
    
    # First, check if the current decomposition is valid
    if not _verify(U, V, W, n, m, p):
        return None
    
    # Try to remove terms one by one and check if the decomposition still holds
    # This is O(rank^2 * n*m*p) which is feasible for rank=88
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = rank
    
    # Try to remove each term and see if we can compensate
    for r_remove in range(rank):
        if time.time() - start > max_time:
            break
        
        # Create a copy without term r_remove
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Zero out column r_remove
        for i in range(len(U_test)):
            U_test[i][r_remove] = 0
        for i in range(len(V_test)):
            V_test[i][r_remove] = 0
        for i in range(len(W_test)):
            W_test[i][r_remove] = 0
        
        # Check if the remaining terms still satisfy all constraints
        valid = True
        U_np_test = np.array(U_test, dtype=np.float64)
        V_np_test = np.array(V_test, dtype=np.float64)
        W_np_test = np.array(W_test, dtype=np.float64)
        
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    actual = 0.0
                    for r in range(rank):
                        if r == r_remove:
                            continue
                        actual += U_np_test[i * m + k, r] * V_np_test[k * p + j, r] * W_np_test[j * n + i, r]
                    if abs(actual - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # We found a redundant term! Remove it and continue
            # Create a new decomposition without this term
            new_rank = rank - 1
            U_new = [[U[i][r] for r in range(rank) if r != r_remove] for i in range(len(U))]
            V_new = [[V[i][r] for r in range(rank) if r != r_remove] for i in range(len(V))]
            W_new = [[W[i][r] for r in range(rank) if r != r_remove] for i in range(len(W))]
            
            if _verify(U_new, V_new, W_new, n, m, p):
                U = U_new
                V = V_new
                W = W_new
                rank = new_rank
                best_U = U
                best_V = V
                best_W = W
                best_rank = rank
                # Restart the pruning process with the new rank
                return _try_prune(U, V, W, n, m, p, max_time - (time.time() - start))
    
    return (best_U, best_V, best_W) if best_rank < rank else None

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # First, try the rank-88 decomposition
    U88, V88, W88 = _build_rank88()
    
    if _verify(U88, V88, W88, n, m, p):
        # Try to prune it further
        pruned = _try_prune(U88, V88, W88, n, m, p, max_time=60)
        if pruned is not None:
            U_p, V_p, W_p = pruned
            if _verify(U_p, V_p, W_p, n, m, p):
                return {"U": U_p, "V": V_p, "W": W_p}
        
        # If pruning didn't help, return the rank-88 decomposition
        return {"U": U88, "V": V88, "W": W88}
    
    # Fallback: schoolbook rank 96
    U96, V96, W96 = _build_schoolbook()
    if _verify(U96, V96, W96, n, m, p):
        return {"U": U96, "V": V96, "W": W96}
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to construct a valid decomposition")
