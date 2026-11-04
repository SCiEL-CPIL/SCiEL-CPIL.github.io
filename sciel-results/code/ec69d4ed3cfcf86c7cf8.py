"""Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition.

Strategy:
  Split A (2x4) into two 2x2 blocks A1=A[:,0:2], A2=A[:,2:4].
  Split B (4x4) into four 2x2 blocks:
    B11=B[0:2,0:2], B12=B[0:2,2:4]
    B21=B[2:4,0:2], B22=B[2:4,2:4]
  
  C = A1*B11 + A2*B21 + A1*B12 + A2*B22
  
  Use Strassen's 7-multiplication algorithm independently on each of the four
  2x2 block products. Total rank: 4 * 7 = 28.
  
  The key insight: we compute each 2x2 block product using Strassen's 7-multiplication
  algorithm, and the results are added to form C.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    
    For C = A*B where A,B,C are 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    Strassen's 7 products:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)(b11)
    M3 = (a11)(b12-b22)
    M4 = (a22)(b21-b11)
    M5 = (a11+a12)(b22)
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4, M1-M2+M3+M6]]
    
    Returns list of 7 dicts with 'a', 'b', 'c' coefficient dicts.
    A entries indexed as: a11=0, a12=1, a21=2, a22=3
    B entries indexed as: b11=0, b12=1, b21=2, b22=3
    C entries indexed as: c11=0, c12=1, c21=2, c22=3
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _try_rank_reduction(U, V, W, n=2, m=4, p=4, target_rank=27):
    """Try to reduce rank by finding if one column can be eliminated.
    
    For each column r, check if the rank-1 tensor u_r ⊗ v_r ⊗ w_r can be
    expressed as a linear combination of the other rank-1 tensors.
    
    This is equivalent to checking if the 28 vectors (flattened rank-1 tensors)
    are linearly dependent, and if so, finding the dependency.
    """
    rank = U.shape[1]
    nmp = n * m * p  # 32
    
    # Build matrix where each column is a flattened rank-1 tensor
    M = np.zeros((nmp, rank))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i * m + j, r]
                if u_val == 0:
                    continue
                for k in range(p):
                    M[i * m * p + j * p + k, r] = u_val * V[j * p + k, r] * W[k * n + i, r]
    
    # Check rank
    try:
        s_vals = np.linalg.svd(M, full_matrices=False)[1]
    except:
        return None
    
    tol = 1e-9
    effective_rank = int(np.sum(s_vals > tol))
    
    if effective_rank >= rank:
        return None  # No reduction possible
    
    # Find null space vectors
    # Use SVD: M = U_svd @ diag(s) @ Vt
    U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
    null_vecs = Vt[effective_rank:]  # shape (rank - effective_rank, rank)
    
    # For each null space vector, try to eliminate the column with largest coefficient
    for ns in null_vecs:
        max_idx = int(np.argmax(np.abs(ns)))
        if abs(ns[max_idx]) < tol:
            continue
        
        # The relation is: sum_r ns[r] * M[:,r] = 0
        # So M[:,max_idx] = -sum_{r != max_idx} (ns[r]/ns[max_idx]) * M[:,r]
        
        # This means the flattened tensor for column max_idx is a linear combination
        # of the others. But we need to express this in terms of U, V, W.
        
        # For a valid rank-1 decomposition reduction, we need to find NEW rank-1 terms.
        # The simple approach doesn't work because the linear combination of rank-1 tensors
        # is not generally a rank-1 tensor.
        
        # However, if we can find coefficients c_r such that:
        # u_max ⊗ v_max ⊗ w_max = sum_{r != max} c_r * u_r ⊗ v_r ⊗ w_r
        # This is generally NOT possible unless the tensors have special structure.
        
        # So linear dependence of flattened vectors doesn't directly give rank reduction.
        # We need a different approach.
        
        pass
    
    return None


def _search_for_rank27():
    """Search for a rank-27 decomposition by trying to merge Strassen terms.
    
    Key insight: In the 2x4x4 problem, A has only 2 rows. The Strassen terms
    that involve row 0 of A and row 1 of A might have dependencies when
    considered across the four blocks.
    
    Specifically, consider the M1 terms from all four blocks:
    - Block 0: (A[0,0]+A[0,1]) * (B[0,0]+B[0,1]+B[2,0]+B[2,1]) ... 
    Actually, let me be more careful.
    
    Let me try a different approach: assume the rank is 27 and set up equations.
    """
    n, m, p = DIMENSIONS
    
    # The true tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # We need: sum_{r=1}^{27} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of 32 equations. With 27 rank-1 terms, we have many unknowns.
    # This is underdetermined, so a solution likely exists. But finding it is hard.
    
    # Let me try a specific ansatz: take the 28 Strassen terms and try to eliminate one
    # by expressing it as a combination that preserves the rank-1 structure.
    
    # Actually, let me check if the problem has a known lower bound.
    # The tensor rank of M_{2,4,4} is at least max(2*4, 4*4, 2*4) = 16 by trivial bounds.
    # Better bounds exist. The current best known might be 27 or 28.
    
    # For now, let's just verify that rank 28 works and return it.
    return None


def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm on each of the four 2x2 block products:
      A1*B11, A2*B21, A1*B12, A2*B22
    
    Total rank: 4 * 7 = 28.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)  
      W shape: (p*n, rank) = (8, 28)
    
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i (trace convention W[k*n+i])
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank))   # 8 x 28
    V = np.zeros((m * p, rank))   # 16 x 28
    W = np.zeros((p * n, rank))   # 8 x 28
    
    strassen_patterns = _strassen_2x2()
    
    # Block definitions: (a_base_j, b_base_j, b_base_k, c_base_k)
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U[global_i * m + global_j, col] = coeff
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V[global_j * p + global_k, col] = coeff
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                W[global_k * n + global_i, col] = coeff
            col += 1
    
    # Verify the decomposition
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i * m + j, r]
                if u_val == 0:
                    continue
                for k in range(p):
                    T_decomp[i, j, k] += u_val * V[j * p + k, r] * W[k * n + i, r]
    
    if not np.allclose(T_decomp, 1.0, atol=1e-9):
        raise ValueError("Decomposition verification failed")
    
    # Convert to list of lists with integer values
    U_list = [[int(U[i, j]) if abs(U[i, j] - round(U[i, j])) < 1e-9 else U[i, j] for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) if abs(V[i, j] - round(V[i, j])) < 1e-9 else V[i, j] for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) if abs(W[i, j] - round(W[i, j])) < 1e-9 else W[i, j] for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
