"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-7 Strassen decomposition for 2x2 matrices
    # U2[i, r], V2[j, r], W2[k, r]
    # i, j, k are flattened 2x2 indices (0..3)
    # r is term index (0..6)
    
    # Indices for 2x2:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: W[k*2+i, r] -> 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # M3 = A00(B10-B11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # M4 = A11(B01-B00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4 decomposition (Rank 49)
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Map global index to (block_index, local_index)
    # Global index g = r*4 + c
    # r = 2*r1 + r0, c = 2*c1 + c0
    # block_idx = r1*2 + c1 (for A/U and B/V)
    # local_idx = r0*2 + c0
    
    def get_indices(g, type):
        # type 'U': g = i*4+j -> i=2*i1+i0, j=2*j1+j0 -> block=2*i1+j1, local=2*i0+j0
        # type 'V': g = j*4+k -> j=2*j1+j0, k=2*k1+k0 -> block=2*j1+k1, local=2*j0+k0
        # type 'W': g = k*4+i -> k=2*k1+k0, i=2*i1+i0 -> block=2*k1+i1, local=2*k0+i0
        
        r = g // 4
        c = g % 4
        
        if type == 'U':
            r1, r0 = divmod(r, 2)
            c1, c0 = divmod(c, 2)
            block = 2*r1 + c1
            local = 2*r0 + c0
        elif type == 'V':
            r1, r0 = divmod(r, 2) # r is j
            c1, c0 = divmod(c, 2) # c is k
            block = 2*r1 + c1
            local = 2*r0 + c0
        elif type == 'W':
            r1, r0 = divmod(r, 2) # r is k
            c1, c0 = divmod(c, 2) # c is i
            block = 2*r1 + c1
            local = 2*r0 + c0
        else:
            raise ValueError("Invalid type")
        
        return block, local

    for g_u in range(16):
        bu, lu = get_indices(g_u, 'U')
        for g_v in range(16):
            bv, lv = get_indices(g_v, 'V')
            for g_w in range(16):
                bw, lw = get_indices(g_w, 'W')
                
                # We need to fill U, V, W such that sum_r U[g_u, r] V[g_v, r] W[g_w, r] = T[g_u, g_v, g_w]
                # T is 1 if the multiplication term exists.
                # For Kronecker product, we iterate over r1, r0
                # U[g_u, r] = U2[bu, r1] * U2[lu, r0]
                # V[g_v, r] = V2[bv, r1] * V2[lv, r0]
                # W[g_w, r] = W2[bw, r1] * W2[lw, r0]
                
                # This is not quite right. The decomposition is defined by the factors.
                # We just need to construct the factors U, V, W directly.
                
                pass # We don't need to iterate over all g_u, g_v, g_w to build factors.
                    # We build factors by iterating over their indices and term indices.

    # Re-initialize and build correctly
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r1 in range(7):
        for r0 in range(7):
            r = 7*r1 + r0
            
            for g_u in range(16):
                bu, lu = get_indices(g_u, 'U')
                val_u = U2[bu][r1] * U2[lu][r0]
                U[g_u][r] = val_u
                
            for g_v in range(16):
                bv, lv = get_indices(g_v, 'V')
                val_v = V2[bv][r1] * V2[lv][r0]
                V[g_v][r] = val_v
                
            for g_w in range(16):
                bw, lw = get_indices(g_w, 'W')
                val_w = W2[bw][r1] * W2[lw][r0]
                W[g_w][r] = val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
