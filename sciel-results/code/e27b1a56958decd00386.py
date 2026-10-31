"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2 matrices
    # U, V, W are of shape (4, 7)
    # Indices: i*2+j, r
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Factor 1: (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Factor 2: (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Factor 3: A11(B12-B22)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Factor 4: A22(B21-B11)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Factor 5: (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Factor 6: (A21-A11)(B11+B12)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Factor 7: (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Build 4x4 factors using Kronecker product
    # Rank = 7 * 7 = 49
    # Size = 16
    rank = 49
    U4 = [[0]*rank for _ in range(16)]
    V4 = [[0]*rank for _ in range(16)]
    W4 = [[0]*rank for _ in range(16)]
    
    for a in range(2):
        for b in range(2):
            for ap in range(2):
                for bp in range(2):
                    # Indices for U4
                    # i = 2*a + ap, j = 2*b + bp
                    # idx = i*4 + j
                    i = 2*a + ap
                    j = 2*b + bp
                    idx_u = i*4 + j
                    
                    # Indices for V4
                    # j = 2*b + bp, k = 2*bp + ? No.
                    # V is indexed by j, k.
                    # j is col of A, row of B.
                    # k is col of B.
                    # So V index is (j, k).
                    # j = 2*b + bp? No.
                    # In U, j is the second index.
                    # In V, j is the first index.
                    # So for V, the first index is j.
                    # Let k = 2*c + cp.
                    # We need to iterate over c, cp as well?
                    # The Kronecker product structure:
                    # U4[i,j, r1,r2] = U2[a,b, r1] * U2[ap,bp, r2]
                    # V4[j,k, r1,r2] = V2[b,c, r1] * V2[bp,cp, r2]
                    # W4[k,i, r1,r2] = W2[c,a, r1] * W2[cp,ap, r2]
                    
                    # We need to sum over r1, r2?
                    # No, the rank index is (r1, r2).
                    # So for each (r1, r2), we set the value.
                    
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            
                            val_u = U2[a*2+b][r1] * U2[ap*2+bp][r2]
                            if val_u != 0:
                                U4[idx_u][r] = val_u
                            
                            # V index: j, k
                            # j = 2*b + bp
                            # k = 2*c + cp
                            # We need to iterate c, cp
                            # This loop is inside the a,b,ap,bp loop.
                            # But V depends on b, c, bp, cp.
                            # So we must iterate c, cp.
                            
                            # It is more efficient to nest the loops properly.
                            # Let's restructure the loops to avoid redundant work.
                            
                            # Actually, the Kronecker product is:
                            # U4 = U2 (x) U2
                            # The indices are (i1, i2, j1, j2) for U?
                            # No, U is (i, j). i=(a, ap), j=(b, bp).
                            # So U4[(a, ap), (b, bp), (r1, r2)] = U2[a, b, r1] * U2[ap, bp, r2]
                            
                            # V is (j, k). j=(b, bp), k=(c, cp).
                            # V4[(b, bp), (c, cp), (r1, r2)] = V2[b, c, r1] * V2[bp, cp, r2]
                            
                            # W is (k, i). k=(c, cp), i=(a, ap).
                            # W4[(c, cp), (a, ap), (r1, r2)] = W2[c, a, r1] * W2[cp, ap, r2]
                            
                            # So we need to iterate over all a,b,c,ap,bp,cp.
                            
                            # Let's move the r1, r2 loop outside or inside?
                            # It's better to iterate over indices and then r.
                            
                            # I will rewrite the construction part.
                            pass
    
    # Correct construction
    U4 = [[0]*rank for _ in range(16)]
    V4 = [[0]*rank for _ in range(16)]
    W4 = [[0]*rank for _ in range(16)]
    
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for ap in range(2):
                    for bp in range(2):
                        for cp in range(2):
                            idx_u = (2*a + ap)*4 + (2*b + bp)
                            idx_v = (2*b + bp)*4 + (2*c + cp)
                            idx_w = (2*c + cp)*4 + (2*a + ap)
                            
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1*7 + r2
                                    
                                    u_val = U2[a*2+b][r1] * U2[ap*2+bp][r2]
                                    v_val = V2[b*2+c][r1] * V2[bp*2+cp][r2]
                                    w_val = W2[c*2+a][r1] * W2[cp*2+ap][r2]
                                    
                                    if u_val != 0:
                                        U4[idx_u][r] = u_val
                                    if v_val != 0:
                                        V4[idx_v][r] = v_val
                                    if w_val != 0:
                                        W4[idx_w][r] = w_val
                                        
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
