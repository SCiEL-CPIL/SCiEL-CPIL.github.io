"""Matrix multiplication 3x4 by 4x6: Combined Strassen + optimization approach."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _construct_strassen_3x4x6():
    """
    Construct a Strassen-based decomposition for 3x4x6 matrix multiplication.
    
    Strategy: 
    - Split the 4x6 multiplication into blocks
    - Use Strassen's algorithm for 2x2x2 sub-blocks
    - Combine with schoolbook for remaining parts
    
    This should give rank significantly below 72 (schoolbook).
    """
    n, m, p = 3, 4, 6
    terms = []  # Each term: (u_dict, v_dict, w_dict)
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """
        Add Strassen decomposition for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1]
        -> C[i1:i2+1, k1:k2+1]
        
        For 2x2x2, Strassen uses 7 multiplications instead of 8.
        """
        # A indices: (i1,j1), (i1,j2), (i2,j1), (i2,j2)
        # B indices: (j1,k1), (j1,k2), (j2,k1), (j2,k2)
        # C indices: (i1,k1), (i1,k2), (i2,k1), (i2,k2)
        
        # M1 = (A11 + A22) * (B11 + B22)
        u1 = {(i1, j1): 1, (i2, j2): 1}
        v1 = {(j1, k1): 1, (j2, k2): 1}
        w1 = {(i1, k1): 1, (i1, k2): -1, (i2, k1): -1, (i2, k2): 1}
        terms.append((u1, v1, w1))
        
        # M2 = A22 * (B21 - B11)
        u2 = {(i2, j2): 1}
        v2 = {(j2, k1): 1, (j1, k1): -1}
        w2 = {(i2, k1): 1, (i2, k2): 1}
        terms.append((u2, v2, w2))
        
        # M3 = (A11 - A21) * B22
        u3 = {(i1, j1): 1, (i2, j1): -1}
        v3 = {(j2, k2): 1}
        w3 = {(i1, k1): 1, (i1, k2): 1}
        terms.append((u3, v3, w3))
        
        # M4 = A11 * (B12 - B22)
        u4 = {(i1, j1): 1}
        v4 = {(j1, k2): 1, (j2, k2): -1}
        w4 = {(i1, k2): 1, (i2, k2): 1}
        terms.append((u4, v4, w4))
        
        # M5 = (A12 + A22) * B11
        u5 = {(i1, j2): 1, (i2, j2): 1}
        v5 = {(j1, k1): 1}
        w5 = {(i1, k1): 1, (i2, k1): 1}
        terms.append((u5, v5, w5))
        
        # M6 = (A11 - A12) * (B12 + B22)
        u6 = {(i1, j1): 1, (i1, j2): -1}
        v6 = {(j1, k2): 1, (j2, k2): 1}
        w6 = {(i1, k2): 1}
        terms.append((u6, v6, w6))
        
        # M7 = (A21 - A11) * (B11 + B12)
        u7 = {(i2, j1): 1, (i1, j1): -1}
        v7 = {(j1, k1): 1, (j1, k2): 1}
        w7 = {(i2, k1): 1}
        terms.append((u7, v7, w7))
    
    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term A[i,j] * B[j,k] -> C[i,k]"""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(i, k): 1}
        terms.append((u, v, w))
    
    # For 3x4x6:
    # Split into blocks:
    # - Rows 0-1, cols 0-1, cols 0-1: use Strassen (7 terms)
    # - Rows 0-1, cols 2-3, cols 0-1: use Strassen (7 terms)
    # - Row 2, cols 0-3, cols 0-5: use schoolbook (12 terms)
    # Total: 7 + 7 + 12 = 26 terms... but this doesn't cover all cases
    
    # Let me try a different approach:
    # Split m=4 into two groups of 2
    # Split p=6 into three groups of 2
    # For each (m_block, p_block) pair, use Strassen if n_block >= 2
    
    # Actually, let's be more systematic:
    # We have n=3 rows, m=4 cols, p=6 rows of output
    
    # Strategy: 
    # - For rows 0-1 (2 rows), use Strassen on 2x2x2 blocks
    # - For row 2 (1 row), use schoolbook
    
    # Split m=4 into [0,1] and [2,3]
    # Split p=6 into [0,1], [2,3], [4,5]
    
    # For rows 0-1:
    #   For each m_block in [0-1, 2-3]:
    #     For each p_block in [0-1, 2-3, 4-5]:
    #       Use Strassen (7 terms)
    #   Total: 2 * 3 * 7 = 42 terms
    
    # For row 2:
    #   For each m in [0-3]:
    #     For each p in [0-5]:
    #       Use schoolbook (1 term)
    #   Total: 4 * 6 = 24 terms
    
    # Grand total: 42 + 24 = 66 terms
    
    # Let's implement this:
    
    # Rows 0-1, using Strassen
    for j_block in range(2):  # m blocks: [0-1], [2-3]
        j1 = j_block * 2
        j2 = j1 + 1
        for k_block in range(3):  # p blocks: [0-1], [2-3], [4-5]
            k1 = k_block * 2
            k2 = k1 + 1
            add_strassen_block(0, 1, j1, j2, k1, k2)
    
    # Row 2, using schoolbook
    for j in range(4):
        for k in range(6):
            add_schoolbook_term(2, j, k)
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    return U, V, W

def _try_rank_am(n, m, p, rank, max_iter=300, tol=1e-8, seed=None):
    """Try to find a decomposition of given rank using alternating minimization."""
    if _time_left() < 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    obj += (s - 1.0) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def _try_column_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by removing redundant columns.
    Returns (success, new_U, new_V, new_W, new_rank)
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try removing each column one at a time
    for remove_col in range(rank):
        if _time_left() < 5.0:
            break
        
        U_test = np.delete(U_np, remove_col, axis=1)
        V_test = np.delete(V_np, remove_col, axis=1)
        W_test = np.delete(W_np, remove_col, axis=1)
        
        # Check if the tensor equation still holds
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    val = 0.0
                    for c in range(rank - 1):
                        val += U_test[i*m+j, c] * V_test[j*p+k, c] * W_test[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            new_rank = rank - 1
            new_U = U_test.tolist()
            new_V = V_test.tolist()
            new_W = W_test.tolist()
            return True, new_U, new_V, new_W, new_rank
    
    return False, None, None, None, rank

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Primary: Strassen-based decomposition
    Secondary: Attempt to find lower-rank solution via optimization
    """
    n, m, p = 3, 4, 6
    
    # Build the Strassen-based solution
    U, V, W = _construct_strassen_3x4x6()
    rank = len(U[0])
    
    # Verify the solution is correct
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)
        rank = n * m * p
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Attempt 1: Column reduction
    success, new_U, new_V, new_W, new_rank = _try_column_reduction(U, V, W, n, m, p)
    if success and new_rank < best_rank:
        best_rank = new_rank
        best_U, best_V, best_W = new_U, new_V, new_W
    
    # Attempt 2: Try alternating minimization for lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But in practice, we expect something higher
    
    for target_rank in range(max(24, best_rank - 10), best_rank):
        if _time_left() < 10.0:
            break
        
        # Try a few random initializations
        for seed in range(3):
            if _time_left() < 5.0:
                break
            
            result = _try_rank_am(n, m, p, target_rank, max_iter=200, tol=1e-8, seed=seed)
            if result is not None:
                U_test, V_test, W_test = result
                # Verify
                if _verify(U_test, V_test, W_test, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_test, V_test, W_test
                    break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
