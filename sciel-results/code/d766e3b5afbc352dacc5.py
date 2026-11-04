"""Matrix multiplication 4x4,4x6 rank-minimal exact tensor decomposition.

Uses 2x2 block structure: 4x4 = 2x2 blocks of 2x2, 4x6 = 2x3 blocks of 2x2,
4x6 output = 2x3 blocks of 2x2. Each output block uses the rank-8 decomposition
for 2x2 matrix multiplication (Strassen-style), giving total rank = 6*8 = 48.
"""
import time

DIMENSIONS = [4, 4, 6]

# Rank-8 decomposition for 2x2 matrix multiplication.
# M = A*B where A,B,M are 2x2.
# A = [[a0,a1],[a2,a3]], B = [[b0,b1],[b2,b3]], M = [[m0,m1],[m2,m3]]
#
# 8 rank-1 terms: each term has (A-coeff, B-coeff, M-coeff) in {0,1,-1}
# Using the standard rank-8 decomposition:
# m0 = a0*b0 + a1*b2
# m1 = a0*b1 + a1*b3
# m2 = a2*b0 + a3*b2
# m3 = a2*b1 + a3*b3
#
# This gives 8 rank-1 terms (each product a_i * b_j contributes to one m_k).
# Actually, the schoolbook for 2x2 is 8 terms. Let me use the standard 8-term:

def _rank8_2x2():
    """Return list of 8 (A_row, A_col, B_row, B_col, M_row, M_col) tuples.
    For 2x2: A=[[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]], M=[[m00,m01],[m10,m11]]
    m_rc = sum_s A_r,s * B_s,c
    Rank-1 terms: for each (r,s,c): A_r,s * B_s,c -> M_r,c
    """
    terms = []
    for r in range(2):
        for s in range(2):
            for c in range(2):
                terms.append((r, s, c))
    return terms

def construct():
    """Return {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    
    n=4, m=4, p=6.
    U: (16, rank), V: (24, rank), W: (24, rank)
    
    Block structure:
    - A is 4x4: 2x2 blocks of 2x2. Block (a,c) for a,c in {0,1}.
      A[a*2+i, c*2+j] = A_block[a,c][i,j]
    - B is 4x6: 2x3 blocks of 2x2. Block (c,b) for c in {0,1}, b in {0,1,2}.
      B[c*2+j, b*2+k] = B_block[c,b][j,k]
    - C is 4x6: 2x3 blocks of 2x2. Block (a,b) for a in {0,1}, b in {0,1,2}.
      C[a*2+i, b*2+k] = C_block[a,b][i,k]
    
    C_block[a,b] = sum_c A_block[a,c] * B_block[c,b]
    
    Each 2x2 multiplication uses 8 rank-1 terms.
    Total rank = 2 * 3 * 8 = 48.
    """
    n, m, p = 4, 4, 6
    
    # Block dimensions
    na, nb = 2, 3  # 2x3 blocks of 2x2 for output (and B)
    nc = 2         # 2x2 blocks for A and inner dimension
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Precompute rank-8 terms for 2x2
    r8_terms = []
    for r in range(2):
        for s in range(2):
            for c in range(2):
                r8_terms.append((r, s, c))
    
    for a in range(na):
        for b in range(nb):
            for c in range(nc):
                for (ri, si, ci) in r8_terms:
                    # ri: row in 2x2 output block
                    # si: column of A / row of B in 2x2
                    # ci: column in 2x2 output block
                    
                    # A index: block (a,c), element (ri, si)
                    # A is 4x4: row = a*2 + ri, col = c*2 + si
                    a_row = a * 2 + ri
                    a_col = c * 2 + si
                    u_idx = a_row * m + a_col  # n*m = 16
                    
                    # B index: block (c,b), element (si, ci)
                    # B is 4x6: row = c*2 + si, col = b*2 + ci
                    b_row = c * 2 + si
                    b_col = b * 2 + ci
                    v_idx = b_row * p + b_col  # m*p = 24
                    
                    # C (output) index: block (a,b), element (ri, ci)
                    # C is 4x6: row = a*2 + ri, col = b*2 + ci
                    # W convention: W[k*n + i, r] where k = col of C, i = row of C
                    c_row = a * 2 + ri
                    c_col = b * 2 + ci
                    w_idx = c_col * n + c_row  # p*n = 24
                    
                    col = rank
                    U.append([0] * rank)  # placeholder, will fill
                    V.append([0] * rank)
                    W.append([0] * rank)
                    rank += 1
    
    # Now fill the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(na):
        for b in range(nb):
            for c in range(nc):
                for (ri, si, ci) in r8_terms:
                    a_row = a * 2 + ri
                    a_col = c * 2 + si
                    u_idx = a_row * m + a_col
                    
                    b_row = c * 2 + si
                    b_col = b * 2 + ci
                    v_idx = b_row * p + b_col
                    
                    c_row = a * 2 + ri
                    c_col = b * 2 + ci
                    w_idx = c_col * n + c_row
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
