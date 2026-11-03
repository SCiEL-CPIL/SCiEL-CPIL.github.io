"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(len(U[0])):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    For A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B:
    C00 = ae+bg, C01=af+bh, C10=ce+dg, C11=cf+dh
    
    The 7 products are:
    P1 = (a+d)(e+h)
    P2 = (c+d)(e)
    P3 = (a+b)(h)
    P4 = (c+d)(f-h)
    P5 = (a+d)(h)
    P6 = (b-d)(e)
    P7 = (a-c)(g+h)
    
    C00 = P1 + P4 - P5 - P7
    C01 = P5 + P6
    C10 = P2 + P3
    C11 = P1 - P2 + P3 + P4
    """
    # Each term is (u_vec, v_vec, w_vec) where:
    # u_vec has 4 entries (for a,b,c,d), v_vec has 4 entries (for e,f,g,h),
    # w_vec has 4 entries (for C00,C01,C10,C11)
    # Indexing: A = [[a,b],[c,d]] -> indices 0,1,2,3
    #           B = [[e,f],[g,h]] -> indices 0,1,2,3
    #           C = [[C00,C01],[C10,C11]] -> indices 0,1,2,3
    
    terms = []
    # P1 = (a+d)(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # P2 = (c+d)(e)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # P3 = (a+b)(h)
    terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,0]))
    # P4 = (c+d)(f-h)
    terms.append(([0,0,1,1], [0,1,0,-1], [1,0,0,1]))
    # P5 = (a+d)(h)
    terms.append(([1,0,0,1], [0,0,0,1], [0,1,0,0]))
    # P6 = (b-d)(e)
    terms.append(([0,1,0,-1], [1,0,0,0], [0,1,0,0]))
    # P7 = (a-c)(g+h)
    terms.append(([1,0,-1,0], [0,0,1,1], [-1,0,0,0]))
    
    return terms

def _embed_strassen(n, m, p, row_pair, col_pair, bcol_pair):
    """
    Embed a Strassen 2x2x2 block into the n x m x p problem.
    row_pair: (r0, r1) - rows of A
    col_pair: (c0, c1) - columns of A (which are rows of B)
    bcol_pair: (k0, k1) - columns of B (which are columns of C)
    
    Returns a list of (U_row, V_row, W_row, U_val, V_val, W_val) tuples
    representing the 7 Strassen terms to add.
    """
    r0, r1 = row_pair
    c0, c1 = col_pair
    k0, k1 = bcol_pair
    
    terms = _strassen_2x2x2()
    
    # Map local indices to global
    # A local index: 0->(r0,c0), 1->(r0,c1), 2->(r1,c0), 3->(r1,c1)
    # B local index: 0->(c0,k0), 1->(c0,k1), 2->(c1,k0), 3->(c1,k1)
    # C local index: 0->(k0,r0), 1->(k0,r1), 2->(k1,r0), 3->(k1,r1)
    
    a_map = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    b_map = [(c0, k0), (c0, k1), (c1, k0), (c1, k1)]
    c_map = [(k0, r0), (k0, r1), (k1, r0), (k1, r1)]
    
    # Return the 7 terms as (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    strassen_terms = []
    for u_vec, v_vec, w_vec in terms:
        u_idx = [a_map[li][0]*m + a_map[li][1] for li in range(4) if u_vec[li] != 0]
        u_val = [u_vec[li] for li in range(4) if u_vec[li] != 0]
        v_idx = [b_map[li][0]*p + b_map[li][1] for li in range(4) if v_vec[li] != 0]
        v_val = [v_vec[li] for li in range(4) if v_vec[li] != 0]
        w_idx = [c_map[li][0]*n + c_map[li][1] for li in range(4) if w_vec[li] != 0]
        w_val = [w_vec[li] for li in range(4) if w_vec[li] != 0]
        strassen_terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    return strassen_terms

def _build_with_strassen(n, m, p, placements):
    """
    Build a decomposition using schoolbook for most terms and Strassen for specified placements.
    placements: list of (row_pair, col_pair, bcol_pair)
    """
    # Mark which schoolbook terms are replaced by Strassen
    replaced = set()
    for row_pair, col_pair, bcol_pair in placements:
        r0, r1 = row_pair
        c0, c1 = col_pair
        k0, k1 = bcol_pair
        for i in [r0, r1]:
            for j in [c0, c1]:
                for k in [k0, k1]:
                    replaced.add((i, j, k))
    
    # Total rank = (n*m*p - 8*len(placements)) + 7*len(placements) = n*m*p - len(placements)
    rank = n * m * p - len(placements)
    
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    # Add schoolbook terms (excluding replaced ones)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in replaced:
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    
    # Add Strassen terms
    for row_pair, col_pair, bcol_pair in placements:
        strassen_terms = _embed_strassen(n, m, p, row_pair, col_pair, bcol_pair)
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in strassen_terms:
            for idx, val in zip(u_idx, u_val):
                U[idx][col] += val
            for idx, val in zip(v_idx, v_val):
                V[idx][col] += val
            for idx, val in zip(w_idx, w_val):
                W[idx][col] += val
            col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # Try to use Strassen on 2x2 sub-blocks.
    # We can partition rows {0,1}, cols {0,1} of A, and column pairs of B.
    # 4 column pairs: (0,1), (2,3), (4,5), (6,7)
    # Each 2x2x2 block saves 1 term, so total savings = 4, rank = 72 - 4 = 68
    
    placements = [
        ((0, 1), (0, 1), (0, 1)),
        ((0, 1), (0, 1), (2, 3)),
        ((0, 1), (0, 1), (4, 5)),
        ((0, 1), (0, 1), (6, 7)),
    ]
    
    U, V, W = _build_with_strassen(n, m, p, placements)
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Check time
    if time.time() - start_time > deadline - 5:
        pass  # We're already at the result, just return
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
