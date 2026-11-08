"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We use 8 Strassen blocks of rank 7 each. Total rank 56.
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 by 2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Define the 8 blocks
    # Each block is defined by:
    # a_idx, b_idx, c_idx, d_idx (indices into U, which is A)
    # e_idx, f_idx, g_idx, h_idx (indices into V, which is B)
    # c00_idx, c01_idx, c10_idx, c11_idx (indices into W, which is C)
    
    # A is 2x4. Index i*m + j. i in {0,1}, j in {0,1,2,3}.
    # B is 4x8. Index j*p + k. j in {0,1,2,3}, k in {0..7}.
    # C is 2x8. Index k*n + i. k in {0..7}, i in {0,1}.
    # Note: W[k*n+i] corresponds to C[i,k].
    
    blocks = []
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C00=C[0,0], C01=C[0,1], C10=C[1,0], C11=C[1,1]
    blocks.append({
        'a': 0*4+0, 'b': 0*4+1, 'c': 1*4+0, 'd': 1*4+1,
        'e': 0*8+0, 'f': 0*8+1, 'g': 1*8+0, 'h': 1*8+1,
        'c00': 0*2+0, 'c01': 1*2+0, 'c10': 0*2+1, 'c11': 1*2+1
    })
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,2], f=B[0,3], g=B[1,2], h=B[1,3]
    # C00=C[0,2], C01=C[0,3], C10=C[1,2], C11=C[1,3]
    blocks.append({
        'a': 0*4+0, 'b': 0*4+1, 'c': 1*4+0, 'd': 1*4+1,
        'e': 0*8+2, 'f': 0*8+3, 'g': 1*8+2, 'h': 1*8+3,
        'c00': 2*2+0, 'c01': 3*2+0, 'c10': 2*2+1, 'c11': 3*2+1
    })
    
    # Block 3: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3]
    # e=B[2,0], f=B[2,1], g=B[3,0], h=B[3,1]
    # C00=C[0,0], C01=C[0,1], C10=C[1,0], C11=C[1,1]
    blocks.append({
        'a': 0*4+2, 'b': 0*4+3, 'c': 1*4+2, 'd': 1*4+3,
        'e': 2*8+0, 'f': 2*8+1, 'g': 3*8+0, 'h': 3*8+1,
        'c00': 0*2+0, 'c01': 1*2+0, 'c10': 0*2+1, 'c11': 1*2+1
    })
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3]
    # e=B[2,2], f=B[2,3], g=B[3,2], h=B[3,3]
    # C00=C[0,2], C01=C[0,3], C10=C[1,2], C11=C[1,3]
    blocks.append({
        'a': 0*4+2, 'b': 0*4+3, 'c': 1*4+2, 'd': 1*4+3,
        'e': 2*8+2, 'f': 2*8+3, 'g': 3*8+2, 'h': 3*8+3,
        'c00': 2*2+0, 'c01': 3*2+0, 'c10': 2*2+1, 'c11': 3*2+1
    })
    
    # Block 5: A[:, 0:2] * B[0:2, 4:6] -> C[:, 4:6]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,4], f=B[0,5], g=B[1,4], h=B[1,5]
    # C00=C[0,4], C01=C[0,5], C10=C[1,4], C11=C[1,5]
    blocks.append({
        'a': 0*4+0, 'b': 0*4+1, 'c': 1*4+0, 'd': 1*4+1,
        'e': 0*8+4, 'f': 0*8+5, 'g': 1*8+4, 'h': 1*8+5,
        'c00': 4*2+0, 'c01': 5*2+0, 'c10': 4*2+1, 'c11': 5*2+1
    })
    
    # Block 6: A[:, 0:2] * B[0:2, 6:8] -> C[:, 6:8]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,6], f=B[0,7], g=B[1,6], h=B[1,7]
    # C00=C[0,6], C01=C[0,7], C10=C[1,6], C11=C[1,7]
    blocks.append({
        'a': 0*4+0, 'b': 0*4+1, 'c': 1*4+0, 'd': 1*4+1,
        'e': 0*8+6, 'f': 0*8+7, 'g': 1*8+6, 'h': 1*8+7,
        'c00': 6*2+0, 'c01': 7*2+0, 'c10': 6*2+1, 'c11': 7*2+1
    })
    
    # Block 7: A[:, 2:4] * B[2:4, 4:6] -> C[:, 4:6]
    # a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3]
    # e=B[2,4], f=B[2,5], g=B[3,4], h=B[3,5]
    # C00=C[0,4], C01=C[0,5], C10=C[1,4], C11=C[1,5]
    blocks.append({
        'a': 0*4+2, 'b': 0*4+3, 'c': 1*4+2, 'd': 1*4+3,
        'e': 2*8+4, 'f': 2*8+5, 'g': 3*8+4, 'h': 3*8+5,
        'c00': 4*2+0, 'c01': 5*2+0, 'c10': 4*2+1, 'c11': 5*2+1
    })
    
    # Block 8: A[:, 2:4] * B[2:4, 6:8] -> C[:, 6:8]
    # a=A[0,2], b=A[0,3], c=A[1,2], d=A[1,3]
    # e=B[2,6], f=B[2,7], g=B[3,6], h=B[3,7]
    # C00=C[0,6], C01=C[0,7], C10=C[1,6], C11=C[1,7]
    blocks.append({
        'a': 0*4+2, 'b': 0*4+3, 'c': 1*4+2, 'd': 1*4+3,
        'e': 2*8+6, 'f': 2*8+7, 'g': 3*8+6, 'h': 3*8+7,
        'c00': 6*2+0, 'c01': 7*2+0, 'c10': 6*2+1, 'c11': 7*2+1
    })
    
    current_col = 0
    for blk in blocks:
        a, b, c, d = blk['a'], blk['b'], blk['c'], blk['d']
        e, f, g, h = blk['e'], blk['f'], blk['g'], blk['h']
        c00, c01, c10, c11 = blk['c00'], blk['c01'], blk['c10'], blk['c11']
        
        # m1 = (a+d)(e+h)
        U[a][current_col] += 1
        U[d][current_col] += 1
        V[e][current_col] += 1
        V[h][current_col] += 1
        W[c00][current_col] += 1
        W[c11][current_col] += 1
        current_col += 1
        
        # m2 = (c+d)e
        U[c][current_col] += 1
        U[d][current_col] += 1
        V[e][current_col] += 1
        W[c10][current_col] += 1
        W[c11][current_col] += -1
        current_col += 1
        
        # m3 = a(f-h)
        U[a][current_col] += 1
        V[f][current_col] += 1
        V[h][current_col] += -1
        W[c01][current_col] += 1
        W[c11][current_col] += 1
        current_col += 1
        
        # m4 = d(g-e)
        U[d][current_col] += 1
        V[g][current_col] += 1
        V[e][current_col] += -1
        W[c00][current_col] += 1
        W[c10][current_col] += 1
        current_col += 1
        
        # m5 = (a+b)h
        U[a][current_col] += 1
        U[b][current_col] += 1
        V[h][current_col] += 1
        W[c00][current_col] += -1
        W[c01][current_col] += 1
        current_col += 1
        
        # m6 = (c-a)(e+f)
        U[c][current_col] += 1
        U[a][current_col] += -1
        V[e][current_col] += 1
        V[f][current_col] += 1
        W[c11][current_col] += 1
        current_col += 1
        
        # m7 = (b-d)(g+h)
        U[b][current_col] += 1
        U[d][current_col] += -1
        V[g][current_col] += 1
        V[h][current_col] += 1
        W[c00][current_col] += 1
        current_col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
