import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]
_START_TIME = time.time()
TIME_LIMIT = 120  # seconds

def _time_remaining():
    return TIME_LIMIT - 10 - (time.time() - _START_TIME)

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2: (A10+A11)B00
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1
    V[3][2] -= 1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4: A11(B10-B00)
    U[3][3] += 1
    V[2][3] += 1
    V[0][3] -= 1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5: (A00+A01)B11
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1
    U[0][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1
    U[3][6] -= 1
    V[2][6] += 1
    V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into:
    - Part 1: 2x2x2 (columns j=0,1; rows k=0,1) -> Strassen rank 7
    - Part 2: 2x2x1 (columns j=0,1; row k=2) -> schoolbook rank 4
    - Part 3: 2x1x3 (column j=2; rows k=0,1,2) -> schoolbook rank 6
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x2 block
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Part 2: 2x2x1 block (j in {0,1}, k=2)
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # Part 3: 2x1x3 block (j=2, k in {0,1,2})
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def _try_merge_terms(U, V, W, n, m, p, max_attempts=5000):
    """
    Try to merge pairs of rank-1 terms into a single term.
    If terms r1 and r2 can be replaced by a single term, we reduce rank by 1.
    """
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    rank = U_arr.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Try to merge each pair of columns
    for r1 in range(rank):
        if _time_remaining() < 5:
            break
        for r2 in range(r1 + 1, rank):
            # Check if the sum of two rank-1 tensors can be a rank-1 tensor
            # This is possible if the 3 vectors (u1+v1, v1+v2, w1+w2) are "compatible"
            # Actually, we need: (u1⊗v1⊗w1) + (u2⊗v2⊗w2) = u⊗v⊗w for some u,v,w
            # This is a non-trivial condition. Let's try a different approach:
            # Try to find a column that, when added to the existing sum, cancels
            # two terms.
            
            # Alternative: try to find a new term that, when added, allows removal
            # of two existing terms.
            pass
    
    return best_rank, best_U.tolist(), best_V.tolist(), best_W.tolist()

def _smooth_optimization_attempt(n, m, p, initial_rank, max_iter=10000):
    """
    Use smooth optimization to try to find a lower-rank decomposition.
    Start from the rank-17 solution and try to reduce rank by finding
    dependencies among the terms.
    """
    if _time_remaining() < 15:
        return None
    
    U, V, W = _build_decomposed_2x3x3()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # The tensor T is fixed. We want to find rank-1 tensors that sum to T.
    # Let's try to find if any subset of the 17 terms is linearly dependent
    # in the tensor space, allowing us to replace them with fewer terms.
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Flatten the tensor
    T_flat = T.flatten()
    
    # Each rank-1 term contributes a vector in R^(n*m*m*p*p*n)
    # Term r contributes: outer(U[:,r], outer(V[:,r], W[:,r])).flatten()
    
    # Check if the 17 term vectors are linearly independent
    term_vectors = []
    for r in range(rank):
        tv = np.outer(U_arr[:,r], np.outer(V_arr[:,r], W_arr[:,r])).flatten()
        term_vectors.append(tv)
    
    term_matrix = np.array(term_vectors).T  # shape: (n*m*m*p*p*n, rank)
    
    # Compute rank of term_matrix
    try:
        sv = np.linalg.svd(term_matrix, full_matrices=False)
        # Count singular values above tolerance
        tol = 1e-10
        effective_rank = np.sum(sv > tol)
    except np.linalg.LinAlgError:
        effective_rank = rank
    
    if effective_rank < rank:
        # There's a linear dependency! We can potentially reduce rank.
        # But this doesn't directly give us a lower-rank decomposition.
        # We'd need to find the actual null space vector and use it.
        pass
    
    return None

def _try_direct_construction():
    """
    Try to directly construct a rank-16 or lower decomposition.
    
    Key idea: The 2x3x3 tensor can be viewed as:
    - For each j, we have a 2x1x3 sub-tensor (fixing the middle index)
    - Or for each k, we have a 2x3x1 sub-tensor
    
    Let's try a different block split:
    - Block A: j=0, all k (2x1x3): 6 terms schoolbook
    - Block B: j=1, all k (2x1x3): 6 terms schoolbook  
    - Block C: j=2, all k (2x1x3): 6 terms schoolbook
    Total: 18 (schoolbook)
    
    Or:
    - Block A: k=0, all j (2x3x1): 6 terms
    - Block B: k=1, all j (2x3x1): 6 terms
    - Block C: k=2, all j (2x3x1): 6 terms
    Total: 18
    
    The Strassen trick works when we have a square block. For 2x3x3,
    the most natural square sub-block is 2x2x2 (using j=0,1 and k=0,1).
    
    Let me try yet another approach: use the fact that the tensor has
    a specific symmetry and try to find a rank-16 decomposition by
    solving a system of equations.
    """
    if _time_remaining() < 20:
        return None
    
    n, m, p = 2, 3, 3
    target_rank = 16
    
    # We need to find U (6x16), V (9x16), W (6x16) such that
    # for all (i,j,k): sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    
    # This is a system of 18 equations in 16*6 + 16*9 + 16*6 = 96+144+96 = 336 unknowns
    # But the equations are nonlinear (trilinear).
    
    # Let's try a heuristic: start from rank 17 and try to set one variable to 0
    # and see if the system can still be satisfied.
    
    U, V, W = _build_decomposed_2x3x3()
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    
    # Try to zero out one column and adjust others
    for col_to_remove in range(17):
        if _time_remaining() < 10:
            break
        
        # Save original
        U_orig = U_arr.copy()
        V_orig = V_arr.copy()
        W_orig = W_arr.copy()
        
        # Zero out the column
        U_arr[:, col_to_remove] = 0
        V_arr[:, col_to_remove] = 0
        W_arr[:, col_to_remove] = 0
        
        # Now try to adjust the other columns to compensate
        # This is a nonlinear problem. Let's use a simple iterative approach.
        
        # Compute the residual
        def compute_residual(U_a, V_a, W_a):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(U_a[a], V_a[b] * W_a[c])
                        res += (val - 1)**2
            return res
        
        # Simple gradient descent on the remaining columns
        lr = 0.01
        for iteration in range(200):
            if _time_remaining() < 5:
                break
            
            # Compute gradient numerically
            grad_U = np.zeros_like(U_arr)
            grad_V = np.zeros_like(V_arr)
            grad_W = np.zeros_like(W_arr)
            
            eps = 1e-7
            base_res = compute_residual(U_arr, V_arr, W_arr)
            
            # This is too slow for full gradient. Let's skip and just check
            # if zeroing out the column still gives a valid solution (it won't,
            # but let's verify).
            break
        
        # Restore
        U_arr = U_orig
        V_arr = V_orig
        W_arr = W_orig
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try the decomposed approach (rank 17)
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
