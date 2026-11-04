import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

def _construct_block_strassen_rank65():
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((NM, total_rank))
    V = np.zeros((MP, total_rank))
    W = np.zeros((PN, total_rank))
    
    col_offset = 0
    
    # Strassen 2x2 basis (rank 7)
    # u_s, v_s, w_s are 4-vectors (2x2 matrices)
    # T = sum u_s v_s w_s
    # Z = X Y
    # Z_00 = M1 + M4 - M5 + M7
    # Z_01 = M3 + M5
    # Z_10 = M2 + M4
    # Z_11 = M1 - M2 + M3 + M6
    
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y00
    # M3: X00(Y01-Y11)
    # M4: X11(Y10-Y00)
    # M5: (X00+X01)Y11
    # M6: (X10-X00)(Y00+Y01)
    # M7: (X01-X11)(Y10+Y11)
    
    # We define the 7 terms (u, v, w) for the 2x2x2 tensor.
    # Note: The tensor is sum of rank-1 terms.
    # The "scalar multiplications" M1..M7 are not the rank-1 terms themselves,
    # but the Strassen algorithm *is* the decomposition.
    # Wait, if we use Strassen, we are *using* the rank 7 decomposition.
    # So for each block product, we instantiate the 7 rank-1 terms of the 2x2x2 tensor.
    
    # Let's get the 7 rank-1 terms for 2x2x2.
    # This is a known set.
    # Ref: https://en.wikipedia.org/wiki/Strassen_algorithm
    # The 7 multiplications are scalar products of linear combinations.
    # In the tensor decomposition sense, each multiplication corresponds to a term?
    # No, the rank of the tensor is 7. So there are 7 terms.
    # Each term is a rank-1 tensor $u \otimes v \otimes w$.
    
    # Let's hardcode the 7 terms for 2x2x2.
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    # Term 1 (M1): (x0+x3)(y0+y3)
    # This is not a rank-1 tensor. It's a product of sums.
    # The rank-1 decomposition is different.
    # Actually, the Strassen algorithm *is* the decomposition into 7 terms.
    # But the terms are not "scalar multiplications" in the tensor decomposition sense?
    # Yes they are. The 7 scalar multiplications are the rank-1 components.
    
    # Let's verify.
    # $Z = \sum_{k=1}^7 M_k w_k$?
    # $M_k = u_k \cdot X, v_k \cdot Y$.
    # $Z = \sum_k (u_k \cdot X) (v_k \cdot Y) w_k$.
    # This is $\sum_k (u_k \otimes v_k \otimes w_k)$.
    # Yes.
    
    # We need the vectors u_k, v_k, w_k for k=1..7.
    # From Strassen:
    # M1: u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,1,1,1]? No.
    # M1 contributes to all 4 entries of Z.
    # Z00 += M1, Z01 += M1? No.
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # So w1 has 1 at 00, 1 at 11? No, M1 is a scalar.
    # The term is $M_1 \times \text{matrix } W_1$.
    # $W_1$ is the matrix of coefficients of M1 in Z.
    # $W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$?
    # Z00 has +1. Z11 has +1. Z01 has 0. Z10 has 0.
    # So w1 = [1, 0, 0, 1] (flattened 2x2).
    
    # u1 = [1, 0, 0, 1] (X00 + X11)
    # v1 = [1, 0, 0, 1] (Y00 + Y11)
    
    # M2: (X10+X11)Y00
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # Z10 has +1. Z11 has -1.
    # w2 = [0, 0, 1, -1]
    
    # M3: X00(Y01-Y11)
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # Z01 has +1. Z11 has +1.
    # w3 = [0, 1, 0, 1]
    
    # M4: X11(Y10-Y00)
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 0, 1, -1]? No, Y10-Y00.
    # v4 = [ -1, 0, 1, 0 ]? Indices: 0:00, 1:01, 2:10, 3:11.
    # Y10 is idx 2. Y00 is idx 0.
    # v4 = [-1, 0, 1, 0]
    # Z00 has +1. Z10 has +1.
    # w4 = [1, 0, 1, 0]
    
    # M5: (X00+X01)Y11
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # Z00 has -1. Z01 has +1.
    # w5 = [-1, 1, 0, 0]
    
    # M6: (X10-X00)(Y00+Y01)
    # u6 = [-1, 0, 1, 0]
    # v6 = [1, 1, 0, 0]
    # Z11 has +1.
    # w6 = [0, 0, 0, 1]
    
    # M7: (X01-X11)(Y10+Y11)
    # u7 = [0, 1, 0, -1]
    # v7 = [0, 0, 1, 1]
    # Z00 has +1.
    # w7 = [1, 0, 0, 0]
    
    strassen_2x2_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Block Strassen for 4x4
    # X_k, Y_k, Z_k (coefficients for C blocks)
    # k=0 (M1): X=A00+A11, Y=B00+B11, C: C00+=1, C11+=1
    # k=1 (M2): X=A10+A11, Y=B00, C: C10+=1, C11+=-1
    # k=2 (M3): X=A00, Y=B01-B11, C: C01+=1, C11+=1
    # k=3 (M4): X=A11, Y=B10-B00, C: C00+=1, C10+=1
    # k=4 (M5): X=A00+A01, Y=B11, C: C00+=-1, C01+=1
    # k=5 (M6): X=A10-A00, Y=B00+B01, C: C11+=1
    # k=6 (M7): X=A01-A11, Y=B10+B11, C: C00+=1
    
    # alpha[k][block_idx]
    # beta[k][block_idx]
    # gamma[k][block_idx]
    
    # Block indices: 0:00, 1:01, 2:10, 3:11
    
    alphas = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    betas = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [-1, 0, 1, 0],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    gammas = [
        [1, 0, 0, 1], # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],# M2: C10 += 1, C11 += -1
        [0, 1, 0, 1], # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0], # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],# M5: C00 += -1, C01 += 1
        [0, 0, 0, 1], # M6: C11 += 1
        [1, 0, 0, 0]  # M7: C00 += 1
    ]
    
    # Iterate over 7 block products
    for k in range(7):
        for s in range(7):
            u_s, v_s, w_s = strassen_2x2_terms[s]
            
            # Add rank-1 term
            col = col_offset
            
            # U
            for ab in range(4):
                if alphas[k][ab] == 0: continue
                a = ab // 2
                b = ab % 2
                for pq in range(4):
                    if u_s[pq] == 0: continue
                    p = pq // 2
                    q = pq % 2
                    global_i = 2*a + p
                    global_j = 2*b + q
                    U[global_i*m + global_j, col] += alphas[k][ab] * u_s[pq]
            
            # V
            for cd in range(4):
                if betas[k][cd] == 0: continue
                c = cd // 2
                d = cd % 2
                for rq in range(4):
                    if v_s[rq] == 0: continue
                    r = rq // 2
                    q = rq % 2
                    global_j = 2*c + r # B row index is j in A? No.
                    # B is 4x4. Indices (row, col).
                    # In the 2x2 block, row r, col q.
                    # Global row = 2*c + r. Global col = 2*d + q.
                    # V index is (j, k). j is global row, k is global col.
                    global_row_b = 2*c + r
                    global_col_b = 2*d + q
                    V[global_row_b*p + global_col_b, col] += betas[k][cd] * v_s[rq]
            
            # W
            for ij in range(4):
                if gammas[k][ij] == 0: continue
                i = ij // 2
                j = ij % 2
                for rc in range(4):
                    if w_s[rc] == 0: continue
                    r = rc // 2
                    c = rc % 2
                    # C block (i,j). Local (r,c).
                    # Global C row = 2*i + r. Global C col = 2*j + c.
                    # W index is (k, i) in the problem?
                    # Problem: W[k*n+i].
                    # Here C is 4x4 (k=0..3).
                    # Global col of C is 2*j + c.
                    # Global row of C is 2*i + r.
                    global_col_c = 2*j + c
                    global_row_c = 2*i + r
                    # W index: global_col_c * n + global_row_c
                    W[global_col_c*n + global_row_c, col] += gammas[k][ij] * w_s[rc]
            
            col_offset += 1
            
    # 4x4x1 part
    for i in range(4):
        for j in range(4):
            col = col_offset
            U[i*m + j, col] = 1.0
            V[j*p + 4, col] = 1.0
            W[4*n + i, col] = 1.0
            col_offset += 1
            
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    U, V, W = _construct_block_strassen_rank65()
    return {"U": U, "V": V, "W": W}

