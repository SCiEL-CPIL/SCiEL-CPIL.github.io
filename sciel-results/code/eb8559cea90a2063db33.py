import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        # i, j, k are 0-indexed global indices
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        U[i * m + j][col] += u_val
        V[j * p + k][col] += v_val
        W[k * n + i][col] += w_val
        col += 1

    # Strassen terms for 2x2x2
    # Local indices 0,1
    # u, v, w are 4x7 matrices (flattened 2x2)
    # u[r][i*2+j], v[r][j*2+k], w[r][k*2+i]
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We need U, V, W such that sum U V W = 1
    # This is equivalent to the standard Strassen decomposition of the tensor.
    
    # Let's generate the 7 terms explicitly.
    # Term 1: (a00+a11)(b00+b11)
    # U: a00=1, a11=1. Others 0.
    # V: b00=1, b11=1. Others 0.
    # W: c00=1, c01=1, c10=1, c11=1? No.
    # The W part comes from the combination.
    
    # It is easier to use the known rank-1 decomposition of the Strassen tensor.
    # The 7 rank-1 tensors for 2x2x2 multiplication.
    
    # I will hardcode the 7 terms for U, V, W for the 2x2x2 block.
    # u, v, w are vectors of length 4.
    
    # Term 1
    u1 = [1, 0, 0, 1] # a00, a01, a10, a11
    v1 = [1, 0, 0, 1] # b00, b01, b10, b11
    w1 = [1, 1, 1, 1] # c00, c01, c10, c11? 
    # Check: sum u1[i,j] v1[j,k] w1[k,i]
    # i=0,j=0,k=0: 1*1*1 = 1
    # i=0,j=0,k=1: 1*0*w = 0
    # i=0,j=1,k=0: 0*1*w = 0
    # i=0,j=1,k=1: 0*1*w = 0
    # This gives 1 at (0,0,0) only. We need 1 everywhere.
    
    # The Strassen algorithm computes C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The tensor decomposition is:
    # T = sum_r u_r v_r w_r
    # where u_r, v_r are the coefficients of A and B in M_r,
    # and w_r is the vector of coefficients for C in the combination.
    
    # M1 = (a00+a11)(b00+b11)
    # u1 = [1,0,0,1]
    # v1 = [1,0,0,1]
    # w1 = [1, 0, 0, 1] # C00 += 1, C11 += 1?
    # C00 = ... + 1*M1 ...
    # C11 = ... + 1*M1 ...
    # So w1 has 1 at (0,0) and (1,1).
    # w1 = [1, 0, 0, 1] (flattened C: c00, c01, c10, c11)
    
    # M2 = (a10+a11)b00
    # u2 = [0,0,1,1]
    # v2 = [1,0,0,0]
    # C10 += 1*M2
    # w2 = [0,0,1,0]
    
    # M3 = a00(b01-b11)
    # u3 = [1,0,0,0]
    # v3 = [0,1,0,-1]
    # C01 += 1*M3
    # C11 += 1*M3
    # w3 = [0,1,0,1]
    
    # M4 = a11(b10-b00)
    # u4 = [0,0,0,1]
    # v4 = [0,0,1,-1]
    # C00 += 1*M4
    # C10 += 1*M4
    # w4 = [1,0,1,0]
    
    # M5 = (a00+a01)b11
    # u5 = [1,1,0,0]
    # v5 = [0,0,0,1]
    # C00 -= 1*M5
    # C01 += 1*M5
    # w5 = [-1,1,0,0]
    
    # M6 = (a10-a00)(b00+b01)
    # u6 = [-1,0,1,0]
    # v6 = [1,1,0,0]
    # C11 += 1*M6
    # w6 = [0,0,0,1]
    
    # M7 = (a01-a11)(b10+b11)
    # u7 = [0,1,0,-1]
    # v7 = [0,0,1,1]
    # C00 += 1*M7
    # w7 = [1,0,0,0]
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0])
    ]
    
    # Schoolbook terms for 1x2x2
    # i=0 only.
    # j=0,1. k=0,1.
    schoolbook_1x2x2_terms = [
        ([1,0], [1,0], [1]), # j=0, k=0. U[0,0]=1, V[0,0]=1, W[0,0]=1
        ([1,0], [0,1], [0]), # j=0, k=1. U[0,0]=1, V[0,1]=1, W[1,0]=1
        ([0,1], [1,0], [1]), # j=1, k=0. U[0,1]=1, V[1,0]=1, W[0,0]=1
        ([0,1], [0,1], [0])  # j=1, k=1. U[0,1]=1, V[1,1]=1, W[1,0]=1
    ]
    # Note: W index is k*n+i. Here n=1 for the block?
    # In the block, i is local 0.
    # W[k, 0].
    # So w is vector of length 2 (k=0,1).
    # w[0] corresponds to k=0. w[1] to k=1.
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Global i: 0,1. j: 0,1. k: 0,1.
    for r in range(7):
        u, v, w = strassen_terms[r]
        for li in range(2):
            for lj in range(2):
                if u[li*2+lj] != 0:
                    gi = li
                    gj = lj
                    add_term(gi, gj, 0, 0, 0, 0) # placeholder, need to loop k too?
                    # No, u is fixed for the term.
                    # We add u[li*2+lj] to U[gi, gj, col]
                    # But we are in a loop for r.
                    # We should add the whole vector at once.
                    
    # Let's refactor add_term to take vectors.
    
    def add_strassen_term(i_offset, j_offset, k_offset, u, v, w):
        nonlocal col
        # u, v, w are lists of length 4 (for 2x2) or 2 (for 1x2)
        # Here we assume 2x2 for u, v and 2x2 for w (flattened)
        # u[li*2+lj], v[lj*2+lk], w[lk*2+li]
        for li in range(2):
            for lj in range(2):
                val_u = u[li*2+lj]
                if val_u != 0:
                    U[(i_offset+li)*m + (j_offset+lj)][col] += val_u
        for lj in range(2):
            for lk in range(2):
                val_v = v[lj*2+lk]
                if val_v != 0:
                    V[(j_offset+lj)*p + (k_offset+lk)][col] += val_v
        for lk in range(2):
            for li in range(2):
                val_w = w[lk*2+li]
                if val_w != 0:
                    W[(k_offset+lk)*n + (i_offset+li)][col] += val_w
        col += 1

    def add_schoolbook_1x2x2_term(i_offset, j_offset, k_offset, u, v, w):
        nonlocal col
        # u length 2 (lj), v length 2 (lk), w length 2 (lk)
        for lj in range(2):
            if u[lj] != 0:
                U[i_offset*m + (j_offset+lj)][col] += u[lj]
        for lj in range(2):
            for lk in range(2):
                if v[lj*2+lk] != 0:
                    V[(j_offset+lj)*p + (k_offset+lk)][col] += v[lj*2+lk]
        for lk in range(2):
            if w[lk] != 0:
                W[(k_offset+lk)*n + i_offset][col] += w[lk]
        col += 1

    # Apply to blocks
    # i_offset = 0, j_offset = 0, k_offset = 0
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_strassen_term(0, 0, 0, u, v, w)
        
    # i_offset = 0, j_offset = 0, k_offset = 2
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_strassen_term(0, 0, 2, u, v, w)
        
    # i_offset = 0, j_offset = 0, k_offset = 4
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_strassen_term(0, 0, 4, u, v, w)
        
    # i_offset = 2, j_offset = 0, k_offset = 0
    for r in range(4):
        u, v, w = schoolbook_1x2x2_terms[r]
        add_schoolbook_1x2x2_term(2, 0, 0, u, v, w)
        
    # i_offset = 2, j_offset = 0, k_offset = 2
    for r in range(4):
        u, v, w = schoolbook_1x2x2_terms[r]
        add_schoolbook_1x2x2_term(2, 0, 2, u, v, w)
        
    # i_offset = 2, j_offset = 0, k_offset = 4
    for r in range(4):
        u, v, w = schoolbook_1x2x2_terms[r]
        add_schoolbook_1x2x2_term(2, 0, 4, u, v, w)
        
    # We have used j_offset = 0 for all.
    # We need to handle j_offset = 2 as well!
    # The decomposition was C = A[:,0:2] B[0:2,:] + A[:,2:4] B[2:4,:]
    # So we need to add terms for j_offset = 2.
    
    # i_offset = 0, j_offset = 2, k_offset = 0
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_strassen_term(0, 2, 0, u, v, w)
        
    # i_offset = 0, j_offset = 2, k_offset = 2
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_strassen_term(0, 2, 2, u, v, w)
        
    # i_offset = 0, j_offset = 2, k_offset = 4
    for r in range(7):
        u, v, w = strassen_terms[r]
        add_strassen_term(0, 2, 4, u, v, w)
        
    # i_offset = 2, j_offset = 2, k_offset = 0
    for r in range(4):
        u, v, w = schoolbook_1x2x2_terms[r]
        add_schoolbook_1x2x2_term(2, 2, 0, u, v, w)
        
    # i_offset = 2, j_offset = 2, k_offset = 2
    for r in range(4):
        u, v, w = schoolbook_1x2x2_terms[r]
        add_schoolbook_1x2x2_term(2, 2, 2, u, v, w)
        
    # i_offset = 2, j_offset = 2, k_offset = 4
    for r in range(4):
        u, v, w = schoolbook_1x2x2_terms[r]
        add_schoolbook_1x2x2_term(2, 2, 4, u, v, w)
        
    return {"U": U, "V": V, "W": W}
