import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over r gives the tensor.
      Here n=m=p=4, so:
        U index: i*4 + k  (A row i, A col k)
        V index: k*4 + j  (B row k, B col j)
        W index: j*4 + i  (C col j, C row i)
    
    Strassen at the 4x4 level: split into 2x2 blocks of 2x2 matrices.
    Each 2x2 sub-multiplication uses Strassen (7 scalar mults).
    Total rank: 7 * 7 = 49.
    """
    n = m = p = 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Index helpers
    def u_idx(a_row, a_col):
        return a_row * m + a_col
    
    def v_idx(b_row, b_col):
        return b_row * p + b_col
    
    def w_idx(c_row, c_col):
        # W[k*n+i, r] where k is B's col (= C's col), i is A's row (= C's row)
        # So W index = c_col * n + c_row
        return c_col * n + c_row
    
    # Outer Strassen terms for 4x4:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each outer term t, define:
    # L_t: 2x2 matrix of linear forms in A elements
    # R_t: 2x2 matrix of linear forms in B elements
    # C_contrib: dict mapping (block_row, block_col) to coefficient
    
    # We represent each element of L_t and R_t as a dict:
    # {(a_row, a_col): coeff} for A, {(b_row, b_col): coeff} for B
    
    # Outer term definitions:
    outer_terms = []
    
    # Term 0: M1 = (A11 + A22)(B11 + B22)
    # L[0,0] = A[0,0]+A[2,2], L[0,1] = A[0,1]+A[2,3], L[1,0] = A[1,0]+A[3,2], L[1,1] = A[1,1]+A[3,3]
    # R similarly for B
    L0 = {}
    R0 = {}
    for r in range(2):
        for c in range(2):
            L0[(r,c)] = {(r,c): 1, (r+2,c+2): 1}
            R0[(r,c)] = {(r,c): 1, (r+2,c+2): 1}
    C0 = {(0,0): 1, (1,1): 1}
    outer_terms.append((L0, R0, C0))
    
    # Term 1: M2 = (A21 + A22)B11
    L1 = {}
    R1 = {}
    for r in range(2):
        for c in range(2):
            L1[(r,c)] = {(r+2,c): 1, (r+2,c+2): 1}
            R1[(r,c)] = {(r,c): 1}
    C1 = {(1,0): 1, (1,1): -1}
    outer_terms.append((L1, R1, C1))
    
    # Term 2: M3 = A11(B12 - B22)
    L2 = {}
    R2 = {}
    for r in range(2):
        for c in range(2):
            L2[(r,c)] = {(r,c): 1}
            R2[(r,c)] = {(r,c+2): 1, (r+2,c+2): -1}
    C2 = {(0,1): 1, (1,1): 1}
    outer_terms.append((L2, R2, C2))
    
    # Term 3: M4 = A22(B21 - B11)
    L3 = {}
    R3 = {}
    for r in range(2):
        for c in range(2):
            L3[(r,c)] = {(r+2,c+2): 1}
            R3[(r,c)] = {(r+2,c): 1, (r,c): -1}
    C3 = {(0,0): 1, (1,0): 1}
    outer_terms.append((L3, R3, C3))
    
    # Term 4: M5 = (A11 + A12)B22
    L4 = {}
    R4 = {}
    for r in range(2):
        for c in range(2):
            L4[(r,c)] = {(r,c): 1, (r,c+2): 1}
            R4[(r,c)] = {(r+2,c+2): 1}
    C4 = {(0,0): -1, (0,1): 1}
    outer_terms.append((L4, R4, C4))
    
    # Term 5: M6 = (A21 - A11)(B11 + B12)
    L5 = {}
    R5 = {}
    for r in range(2):
        for c in range(2):
            L5[(r,c)] = {(r+2,c): 1, (r,c): -1}
            R5[(r,c)] = {(r,c): 1, (r,c+2): 1}
    C5 = {(1,1): 1}
    outer_terms.append((L5, R5, C5))
    
    # Term 6: M7 = (A12 - A22)(B21 + B22)
    L6 = {}
    R6 = {}
    for r in range(2):
        for c in range(2):
            L6[(r,c)] = {(r,c+2): 1, (r+2,c+2): -1}
            R6[(r,c)] = {(r+2,c): 1, (r+2,c+2): 1}
    C6 = {(0,0): 1}
    outer_terms.append((L6, R6, C6))
    
    # Inner Strassen: for each outer term, compute 7 scalar products
    # Inner Strassen scalar products for L * R (both 2x2):
    # s1 = (L[0,0] + L[1,1]) * (R[0,0] + R[1,1])
    # s2 = (L[1,0] + L[1,1]) * R[0,0]
    # s3 = L[0,0] * (R[0,1] - R[1,1])
    # s4 = L[1,1] * (R[1,0] - R[0,0])
    # s5 = (L[0,0] + L[0,1]) * R[1,1]
    # s6 = (L[1,0] - L[0,0]) * (R[0,0] + R[0,1])
    # s7 = (L[0,1] - L[1,1]) * (R[1,0] + R[1,1])
    
    # P[0,0] = s1 + s4 - s5 + s7
    # P[0,1] = s3 + s5
    # P[1,0] = s2 + s4
    # P[1,1] = s1 - s2 + s3 + s6
    
    # For each inner scalar product s_k, we need:
    # - L-side: dict {(r,c): coeff} indicating which L elements to combine
    # - R-side: dict {(r,c): coeff} indicating which R elements to combine
    # - P_contrib: dict {(r,c): coeff} indicating how s_k contributes to P
    
    inner_strassen = [
        # s1: (L[0,0]+L[1,1]) * (R[0,0]+R[1,1])
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s2: (L[1,0]+L[1,1]) * R[0,0]
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s3: L[0,0] * (R[0,1]-R[1,1])
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s4: L[1,1] * (R[1,0]-R[0,0])
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s5: (L[0,0]+L[0,1]) * R[1,1]
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s6: (L[1,0]-L[0,0]) * (R[0,0]+R[0,1])
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s7: (L[0,1]-L[1,1]) * (R[1,0]+R[1,1])
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Now generate all 49 rank terms
    for t in range(7):
        L_t, R_t, C_contrib = outer_terms[t]
        
        for s in range(7):
            r_idx = t * 7 + s
            L_coeffs, R_coeffs, P_contrib = inner_strassen[s]
            
            # Build full U vector (length 16)
            # For each element (lr, lc) in L_coeffs, and each element (ar, ac) in L_t[(lr,lc)]:
            # U[u_idx(ar, ac)] += L_coeffs[(lr,lc)] * L_t[(lr,ac)][(ar,ac)]
            u_vec = np.zeros(16)
            for (lr, lc), l_coeff in L_coeffs.items():
                if l_coeff == 0:
                    continue
                # L_t[(lr,lc)] is a dict of A element coefficients
                for (ar, ac), a_coeff in L_t[(lr, lc)].items():
                    idx = u_idx(ar, ac)
                    u_vec[idx] += l_coeff * a_coeff
            
            # Build full V vector (length 16)
            v_vec = np.zeros(16)
            for (rr, rc), r_coeff in R_coeffs.items():
                if r_coeff == 0:
                    continue
                # R_t[(rr,rc)] is a dict of B element coefficients
                for (br, bc), b_coeff in R_t[(rr, rc)].items():
                    idx = v_idx(br, bc)
                    v_vec[idx] += r_coeff * b_coeff
            
            # Build full W vector (length 16)
            # P_contrib maps (pr, pc) to coefficient
            # C_contrib maps (block_row, block_col) to coefficient
            # For each (pr, pc) in P_contrib and (block_row, block_col) in C_contrib:
            # The element P[pr,pc] contributes to C block (block_row, block_col) at position (pr, pc)
            # Global C index: row = block_row*2 + pr, col = block_col*2 + pc
            w_vec = np.zeros(16)
            for (pr, pc), p_coeff in P_contrib.items():
                if p_coeff == 0:
                    continue
                for (block_row, block_col), c_coeff in C_contrib.items():
                    total_coeff = p_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    c_row = block_row * 2 + pr
                    c_col = block_col * 2 + pc
                    idx = w_idx(c_row, c_col)
                    w_vec[idx] += total_coeff
            
            U[:, r_idx] = u_vec
            V[:, r_idx] = v_vec
            W[:, r_idx] = w_vec
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
