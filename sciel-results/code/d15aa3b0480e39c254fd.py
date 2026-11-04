import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W matrices (4x7 each) for 2x2x2 Strassen decomposition."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: (A00+A11)(B00+B11) -> C00+C11
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # c=1: (A10+A11)B00 -> C10-C11
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # c=2: A00(B01-B11) -> C01+C11
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # c=3: A11(B10-B00) -> C00+C10
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # c=4: (A00+A01)B11 -> -C00+C01
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # c=5: (A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # c=6: (A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_4x4x4_rank49():
    """
    Build rank-49 decomposition of 4x4x4 matrix multiplication tensor.
    Returns list of (u, v, w) triples, each of length 16.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Block Strassen for 4x4x4:
    # Partition A, B, C into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]], etc.
    
    # 7 products P_k, each computed via 2x2x2 Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    # P_k definitions: (A_eff, B_eff) where A_eff, B_eff are linear combos of 2x2 blocks
    # A_block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B_block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    terms = []
    
    for kp in range(7):
        for ms in range(7):
            u = np.zeros(16)
            v = np.zeros(16)
            w = np.zeros(16)
            
            # Fill u from A_specs[kp]
            for coeff, a_block in A_specs[kp]:
                bi, bj = divmod(a_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = U2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            u[gi * 4 + gj] += coeff * val
            
            # Fill v from B_specs[kp]
            for coeff, b_block in B_specs[kp]:
                bi, bj = divmod(b_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = V2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            v[gi * 4 + gj] += coeff * val
            
            # Fill w from C_specs[kp]
            for coeff, c_block in C_specs[kp]:
                bi, bj = divmod(c_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = W2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            w[gj * 4 + gi] += coeff * val  # W[k*n+i], so k=gj, i=gi
            
            if np.any(u) and np.any(v) and np.any(w):
                terms.append((u, v, w))
    
    return terms

def _try_improved_decomposition(terms_444):
    """
    Attempt to reduce rank by exploiting structure in 4x4x8 problem.
    
    Key insight: B is 4x8, C is 4x8. We can potentially share work
    between the left and right halves of B if the A-side linear combinations
    are the same.
    
    Strategy: For each of the 7 Strassen products P_k, we have 7 sub-products.
    For each sub-product ms, the A-side vector u is the same for both halves.
    The B-side vectors v_L and v_R differ only in column offset.
    The C-side vectors w_L and w_R differ only in column offset.
    
    We can potentially merge terms where u is the same and v/w have a simple
    relationship. But in general, each (kp, ms) pair gives two terms
    (one for left half, one for right half).
    
    However, we can try to find linear dependencies among the 98 terms.
    If the rank of the tensor is actually less than 98, we can find
    a smaller decomposition.
    
    For now, we'll just check if we can reduce by exploiting the
    fact that A is the same for both halves.
    """
    n, m, p = 4, 4, 8
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_list = []
    V_list = []
    W_list = []
    
    for half in range(2):
        b_col_off = half * 4
        
        for u, v, w in terms_444:
            U_list.append(u.copy())
            
            v_shifted = np.zeros(32)
            for j in range(4):
                for k in range(4):
                    v_shifted[j * 8 + (k + b_col_off)] = v[j * 4 + k]
            V_list.append(v_shifted)
            
            w_shifted = np.zeros(32)
            for k in range(4):
                for i in range(4):
                    w_shifted[(k + b_col_off) * 4 + i] = w[k * 4 + i]
            W_list.append(w_shifted)
    
    U = np.array(U_list).T  # (16, 98)
    V = np.array(V_list).T  # (32, 98)
    W = np.array(W_list).T  # (32, 98)
    
    return U, V, W

def _reduce_rank_via_svd(U, V, W, tol=1e-10):
    """
    Try to reduce rank by finding linear dependencies.
    
    We want to find the minimal rank r such that there exist
    U' (16 x r), V' (32 x r), W' (32 x r) with
    sum_r U'[i*m+j, r] * V'[j*p+k, r] * W'[k*n+i, r] = 1 for all i,j,k
    
    This is equivalent to saying the tensor T_{ijk} = delta_ik (for matrix mult)
    has rank r.
    
    One approach: check if columns of U, V, W are linearly dependent.
    If a column is a linear combination of others, we can potentially
    merge it.
    
    For now, we'll just compute the rank of each matrix and see if
    we can reduce.
    """
    n, m, p = 4, 4, 8
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank_U = np.linalg.matrix_rank(U, tol=tol)
    rank_V = np.linalg.matrix_rank(V, tol=tol)
    rank_W = np.linalg.matrix_rank(W, tol=tol)
    
    # If any matrix has rank less than the number of columns,
    # we have linear dependencies.
    
    # Try to find a basis for the column space of each matrix
    # and express the columns as linear combinations.
    
    # For U:
    U_svd = np.linalg.svd(U, full_matrices=False)
    U_u, U_s, U_vt = U_svd
    r_U = np.sum(U_s > tol)
    
    # For V:
    V_svd = np.linalg.svd(V, full_matrices=False)
    V_u, V_s, V_vt = V_svd
    r_V = np.sum(V_s > tol)
    
    # For W:
    W_svd = np.linalg.svd(W, full_matrices=False)
    W_u, W_s, W_vt = W_svd
    r_W = np.sum(W_s > tol)
    
    # The minimal rank is at most max(r_U, r_V, r_W) if the tensor
    # can be expressed with that rank, but this is not guaranteed.
    
    # For now, return the original decomposition.
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32
    
    # Build 4x4x4 rank-49 decomposition
    terms_444 = _build_4x4x4_rank49()
    
    # For 4x4x8, split B into two 4x4 halves: B_L (cols 0-3) and B_R (cols 4-7)
    # C = A * B = [A*B_L, A*B_R]
    # So we need two 4x4x4 decompositions, one for each half.
    
    U, V, W = _try_improved_decomposition(terms_444)
    
    # Verify correctness
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-10:
                    error += abs(val - 1)
    
    if error > 1e-8:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school))
        V_school = np.zeros((mp, rank_school))
        W_school = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    rank = U.shape[1]
    
    # Check if we can reduce rank
    U_reduced, V_reduced, W_reduced = _reduce_rank_via_svd(U, V, W)
    
    if U_reduced.shape[1] < rank:
        # Verify reduced decomposition
        error_red = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_reduced[i*m+j, :], V_reduced[j*p+k, :] * W_reduced[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        error_red += abs(val - 1)
        
        if error_red < 1e-8:
            return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
