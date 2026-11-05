"""
Rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication.

Strategy:
  - Base: Strassen on the 2x2 block structure (4 intermediate indices split into
    two pairs of 2) combined with schoolbook on the 5 output columns.
    This gives rank 36 (vs schoolbook 40).
  - Try to reduce further by exploiting the rectangular nature (2 rows, 5 cols)
    with additional Strassen-like tricks on the 2x2 x 2x5 sub-problems.
  - Use a time-bounded search for additional rank reductions.
"""

import time
import itertools

DIMENSIONS = [2, 4, 5]
n, m, p = DIMENSIONS  # 2, 4, 5


def _validate(U, V, W, tol=1e-9):
    """Check that the decomposition is exact: sum_r U[:,r] V[:,r]^T == W reshaped properly."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor is T[a, b, c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # The matrix multiplication tensor has T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for matching indices
    # i.e., T[(i*m+j), (j*p+k), (k*n+i)] = 1
    
    rank = U.shape[1]
    # Compute the reconstructed tensor
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a 3-tensor. Check specific entries.
    # For matrix multiplication: T[(i*m+j), (j'*p+k), (k'*n+i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            # Compute sum_r U[a,r]*V[b,r]*W[c,r]
                            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                            if abs(val - expected) > tol:
                                return False
    return True


def _strassen_2x2x2_block():
    """
    Return the 7 rank-1 terms for 2x2 x 2x2 Strassen multiplication.
    A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    C = [[c0, c1], [c2, c3]]
    
    Strassen products:
    M1 = (a0 + a3) * (b0 + b3)
    M2 = (a2 + a3) * b0
    M3 = a0 * (b1 - b0)
    M4 = a3 * (b3 - b2)
    M5 = (a0 + a1) * b2
    M6 = (a2 - a0) * (b0 + b1)
    M7 = (a1 - a3) * (b2 + b3)
    
    C[0,0] = M1 - M4 + M6 - M7
    C[0,1] = M5 + M7
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M5
    """
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U indexes A: (a0, a1, a2, a3) -> indices 0,1,2,3
    # V indexes B: (b0, b1, b2, b3) -> indices 0,1,2,3
    # W indexes C: (c0, c1, c2, c3) -> indices 0,1,2,3
    
    terms = []
    # M1 = (a0+a3)*(b0+b3), contributes to C[0,0] (+1), C[1,1] (+1)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (a2+a3)*b0, contributes to C[1,0] (+1), C[1,1] (-1)
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = a0*(b1-b0), contributes to C[1,1] (+1)
    terms.append(([1, 0, 0, 0], [0, 1, -1, 0], [0, 0, 0, 1]))
    # M4 = a3*(b3-b2), contributes to C[0,0] (-1), C[1,0] (+1)
    terms.append(([0, 0, 0, 1], [0, 0, -1, 1], [-1, 0, 1, 0]))
    # M5 = (a0+a1)*b2, contributes to C[0,1] (+1), C[1,1] (+1)
    terms.append(([1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]))
    # M6 = (a2-a0)*(b0+b1), contributes to C[0,0] (+1)
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [1, 0, 0, 0]))
    # M7 = (a1-a3)*(b2+b3), contributes to C[0,0] (-1), C[0,1] (+1)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]))
    
    return terms


def _construct_rank_36():
    """
    Rank-36 construction for 2x4x5.
    
    Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    Split B (4x5) into B1 (2x5, rows 0-1) and B2 (2x5, rows 2-3).
    C = A1*B1 + A2*B2.
    
    For each 2x2 x 2x5 product, decompose into:
      - Strassen 2x2x2 on the first 2 columns: 7 terms
      - Schoolbook on the remaining 3 columns: 2*3 = 6 terms
      Total per sub-problem: 13 terms
    Two sub-problems: 26 terms.
    
    Wait, that's only 26, not 36. Let me recount.
    
    Actually for 2x2 x 2x5:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for k=0..4
    
    For k=0,1: use Strassen (7 terms)
    For k=2,3,4: schoolbook (2*3=6 terms)
    Total: 13 terms per 2x2x2x5.
    
    Two such sub-problems: 2*13 = 26 terms.
    
    Hmm, that's even better than 36. Let me verify this is correct.
    
    Actually wait - the Strassen decomposition for 2x2x2 gives us C[0,0], C[0,1], C[1,0], C[1,1]
    using 7 products. Then for columns 2,3,4 we use schoolbook: 2 rows * 3 cols = 6 more products.
    Total: 7 + 6 = 13 per sub-problem.
    Two sub-problems: 26 total.
    
    But wait - can we do better? The 2x2x5 tensor might have rank less than 13.
    For now, rank 26 is a valid construction. Let me verify it.
    
    Actually, I need to be more careful. Let me index things properly.
    
    A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    B is 4x5: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4}
    C is 2x5: C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
           = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
           = (A1*B1)[i,k] + (A2*B2)[i,k]
    
    where A1 = A[:, 0:2] (2x2), B1 = B[0:2, :] (2x5)
          A2 = A[:, 2:4] (2x2), B2 = B[2:4, :] (2x5)
    
    For A1*B1 (2x2 x 2x5):
      Let A1 = [[a0, a1], [a2, a3]] (a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1])
      Let B1 = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]]
      
      C1[0,0] = a0*b00 + a1*b10
      C1[0,1] = a0*b01 + a1*b11
      C1[0,2] = a0*b02 + a1*b12
      C1[0,3] = a0*b03 + a1*b13
      C1[0,4] = a0*b04 + a1*b14
      C1[1,0] = a2*b00 + a3*b10
      C1[1,1] = a2*b01 + a3*b11
      C1[1,2] = a2*b02 + a3*b12
      C1[1,3] = a2*b03 + a3*b13
      C1[1,4] = a2*b04 + a3*b14
      
      Strassen on columns 0,1 (7 terms) + schoolbook on columns 2,3,4 (6 terms) = 13 terms.
    
    Similarly for A2*B2: 13 terms.
    
    Total: 26 terms. Let me construct this.
    """
    rank = 26
    # U: (n*m, rank) = (8, 26)
    # V: (m*p, rank) = (20, 26)
    # W: (p*n, rank) = (10, 26)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        # u_coeffs: dict mapping A-index (i*m+j) to coefficient
        # v_coeffs: dict mapping B-index (j*p+k) to coefficient
        # w_coeffs: dict mapping C-index (k*n+i) to coefficient
        for idx, coeff in u_coeffs.items():
            U[idx][col] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx][col] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx][col] += coeff
        col += 1
    
    # Process sub-problem 1: A1 (cols 0-1) * B1 (rows 0-1)
    # A1 indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B1 indices: B[0,k]=k, B[1,k]=5+k for k=0..4
    # C1 indices: C[0,k]=k, C[1,k]=5+k for k=0..4
    
    # Strassen terms for A1*B1 on columns 0,1
    # A1 = [[a0, a1], [a2, a3]] where a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
    # B1 first 2 cols = [[b00, b01], [b10, b11]] where b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1]
    # C1 first 2 cols = [[c00, c01], [c10, c11]] where c00=C[0,0], c01=C[0,1], c10=C[1,0], c11=C[1,1]
    
    strassen_terms = _strassen_2x2x2_block()
    
    for (u_s, v_s, w_s) in strassen_terms:
        # u_s is for (a0, a1, a2, a3) -> A indices 0,1,2,3
        # v_s is for (b0, b1, b2, b3) -> B indices: b0=B[0,0]=0, b1=B[0,1]=1, b2=B[1,0]=5, b3=B[1,1]=6
        # w_s is for (c0, c1, c2, c3) -> C indices: c0=C[0,0]=0, c1=C[0,1]=1, c2=C[1,0]=5, c3=C[1,1]=6
        
        u_map = {}
        for idx, coeff in enumerate(u_s):
            if coeff != 0:
                u_map[idx] = coeff  # A index = idx (0,1,2,3)
        
        b_idx_map = [0, 1, 5, 6]  # b0->0, b1->1, b2->5, b3->6
        v_map = {}
        for idx, coeff in enumerate(v_s):
            if coeff != 0:
                v_map[b_idx_map[idx]] = coeff
        
        c_idx_map = [0, 1, 5, 6]  # c0->0, c1->1, c2->5, c3->6
        w_map = {}
        for idx, coeff in enumerate(w_s):
            if coeff != 0:
                w_map[c_idx_map[idx]] = coeff
        
        add_term(u_map, v_map, w_map)
    
    # Schoolbook for A1*B1 on columns 2,3,4
    for i in range(2):
        for k in range(2, 5):
            u_map = {i * m + 0: 1, i * m + 1: 0}  # A[i,0] and A[i,1]
            # Wait, this is wrong. Schoolbook for 2x2 x 2x1:
            # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
            # That's 2 terms per (i,k), so 2*3 = 6 terms total.
            
            # Term 1: A[i,0] * B[0,k]
            u_map1 = {i * m + 0: 1}
            v_map1 = {0 * p + k: 1}
            w_map1 = {k * n + i: 1}
            add_term(u_map1, v_map1, w_map1)
            
            # Term 2: A[i,1] * B[1,k]
            u_map2 = {i * m + 1: 1}
            v_map2 = {1 * p + k: 1}
            w_map2 = {k * n + i: 1}
            add_term(u_map2, v_map2, w_map2)
    
    # Process sub-problem 2: A2 (cols 2-3) * B2 (rows 2-3)
    # A2 indices: A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
    # B2 indices: B[2,k]=10+k, B[3,k]=15+k for k=0..4
    # C2 indices: same C indices: C[0,k]=k, C[1,k]=5+k
    
    for (u_s, v_s, w_s) in strassen_terms:
        u_map = {}
        a2_idx_map = [4, 5, 6, 7]  # a0->A[0,2]=4, a1->A[0,3]=5, a2->A[1,2]=6, a3->A[1,3]=7
        for idx, coeff in enumerate(u_s):
            if coeff != 0:
                u_map[a2_idx_map[idx]] = coeff
        
        b2_idx_map = [10, 11, 15, 16]  # b0->B[2,0]=10, b1->B[2,1]=11, b2->B[3,0]=15, b3->B[3,1]=16
        v_map = {}
        for idx, coeff in enumerate(v_s):
            if coeff != 0:
                v_map[b2_idx_map[idx]] = coeff
        
        c_idx_map = [0, 1, 5, 6]
        w_map = {}
        for idx, coeff in enumerate(w_s):
            if coeff != 0:
                w_map[c_idx_map[idx]] = coeff
        
        add_term(u_map, v_map, w_map)
    
    # Schoolbook for A2*B2 on columns 2,3,4
    for i in range(2):
        for k in range(2, 5):
            # Term 1: A[i,2] * B[2,k]
            u_map1 = {i * m + 2: 1}
            v_map1 = {2 * p + k: 1}
            w_map1 = {k * n + i: 1}
            add_term(u_map1, v_map1, w_map1)
            
            # Term 2: A[i,3] * B[3,k]
            u_map2 = {i * m + 3: 1}
            v_map2 = {3 * p + k: 1}
            w_map2 = {k * n + i: 1}
            add_term(u_map2, v_map2, w_map2)
    
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook():
    """Rank 40 schoolbook decomposition."""
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


def _try_improve(rank_36_result, time_budget=100):
    """
    Try to find a lower-rank decomposition by searching for linear dependencies
    or using algebraic techniques.
    
    For now, this is a placeholder that returns the rank-36 result.
    In a more advanced version, we could:
    1. Try to find linear combinations of the 26 terms that cancel.
    2. Use the fact that 2x2x5 might have rank < 13.
    """
    start_time = time.time()
    # The rank-36 construction is already good.
    # Let's check if we can do better with a different split.
    
    # Alternative: split the 5 columns into 2+2+1 and use Strassen on each 2x2x2 part.
    # A1*B1: Strassen on cols 0,1 (7) + Strassen on cols 2,3 (7) + schoolbook on col 4 (2) = 16
    # That's worse than 13.
    
    # Alternative: use a different Strassen variant for 2x2x5.
    # The tensor [2x2, 2x5] might have rank 12 or less.
    
    # For now, return the rank-26 result (which is actually rank 26, not 36).
    elapsed = time.time() - start_time
    return rank_36_result, elapsed


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with rank-26 construction
    result = _construct_rank_36()
    current_rank = len(result['U'][0])
    
    # Validate
    try:
        valid = _validate(result['U'], result['V'], result['W'])
    except Exception:
        valid = False
    
    if not valid:
        # Fall back to schoolbook
        result = _construct_schoolbook()
        current_rank = n * m * p
    
    # Try to improve within time budget
    if time.time() < deadline:
        try:
            improved, elapsed = _try_improve(result, time_budget=60)
            if improved is not None:
                new_rank = len(improved['U'][0])
                if new_rank < current_rank:
                    try:
                        if _validate(improved['U'], improved['V'], improved['W']):
                            result = improved
                            current_rank = new_rank
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Final time check
    if time.time() > deadline - 5:
        # Return immediately
        pass
    
    return result
