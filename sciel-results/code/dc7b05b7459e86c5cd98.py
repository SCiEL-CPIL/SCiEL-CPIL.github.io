import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We split the p=10 columns into 5 blocks of size 2.
    # Each block is a 3x2 matrix B_i.
    # We compute A (2x3) * B_i (3x2) using rank 11.
    # Total rank = 5 * 11 = 55.
    
    rank_total = 55
    U = [[0] * rank_total for _ in range(n * m)] # 6 x 55
    V = [[0] * rank_total for _ in range(m * p)] # 30 x 55
    W = [[0] * rank_total for _ in range(p * n)] # 20 x 55
    
    # Strassen 2x2 decomposition (rank 7)
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Tail decomposition (rank 4)
    # a13 b31, a13 b32, a23 b31, a23 b32
    # C11 += a13 b31
    # C12 += a13 b32
    # C21 += a23 b31
    # C22 += a23 b32
    
    col_offset = 0
    for block_idx in range(5):
        # Columns in B for this block: 2*block_idx, 2*block_idx + 1
        # Rows in B: 0, 1, 2
        # Cols in C: 2*block_idx, 2*block_idx + 1
        
        # --- Strassen Part (7 terms) ---
        
        # M1
        # U: a11 (0), a22 (3) -> indices in U: 0, 3
        # V: b11 (0), b22 (2) -> indices in V: 0, 2 (relative to block start)
        # W: C11 (0), C22 (1) -> indices in W: 0, 1 (relative to block start)
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w, offset):
            for u_idx, c_u in zip(u_indices, coeffs_u):
                for v_idx, c_v in zip(v_indices, coeffs_v):
                    for w_idx, c_w in zip(w_indices, coeffs_w):
                        U[u_idx][offset] += c_u
                        V[v_idx][offset] += c_v
                        W[w_idx][offset] += c_w
            return offset + 1

        # Indices in U (A is 2x3, flattened row-major: a11=0, a12=1, a21=2, a22=3, a13=4, a23=5)
        # Indices in V (B is 3x2 per block, flattened: b11=0, b12=1, b21=2, b22=3, b31=4, b32=5 relative to block)
        # Indices in W (C is 2x2 per block, flattened: c11=0, c12=1, c21=2, c22=3 relative to block)
        
        # Note: V and W indices are global in the final arrays.
        # V row index: row_in_B * 2 + col_in_block
        # W row index: col_in_C * 2 + row_in_C ? No, W[k*n+i].
        # W index for C[r, c] is c * n + r = c * 2 + r.
        
        # Let's map local indices to global indices for this block.
        # A indices are always 0..5.
        # B local indices 0..5 map to global V indices:
        # b11 (local 0) -> V[0*2 + 2*block_idx] = V[4*block_idx]
        # b12 (local 1) -> V[0*2 + 2*block_idx + 1] = V[4*block_idx + 1]
        # b21 (local 2) -> V[1*2 + 2*block_idx] = V[4*block_idx + 2]
        # b22 (local 3) -> V[1*2 + 2*block_idx + 1] = V[4*block_idx + 3]
        # b31 (local 4) -> V[2*2 + 2*block_idx] = V[4*block_idx + 4]
        # b32 (local 5) -> V[2*2 + 2*block_idx + 1] = V[4*block_idx + 5]
        
        # C local indices 0..3 map to global W indices:
        # c11 (local 0) -> W[0*2 + 0] = W[2*block_idx]
        # c12 (local 1) -> W[1*2 + 0] = W[2*block_idx + 1]
        # c21 (local 2) -> W[0*2 + 1] = W[2*block_idx + 2]
        # c22 (local 3) -> W[1*2 + 1] = W[2*block_idx + 3]
        
        # We will manually add the 11 terms for each block.
        
        # Term 1: M1 = (a11+a22)(b11+b22)
        # U: 0 (a11), 3 (a22)
        # V: 4*b (b11), 4*b+2 (b22)
        # W: 2*b (c11), 2*b+3 (c22)
        # Coeffs: U[0]=1, U[3]=1; V[b11]=1, V[b22]=1; W[c11]=1, W[c22]=1
        # This is a sum of 4 products? No, it's (u1+u2)(v1+v2) = u1v1 + u1v2 + u2v1 + u2v2.
        # In tensor decomposition, one rank-1 term is u * v * w.
        # Here we have one multiplication M1.
        # So we add 1 column.
        # U[0] = 1, U[3] = 1
        # V[4b] = 1, V[4b+2] = 1
        # W[2b] = 1, W[2b+3] = 1
        
        # Term 2: M2 = (a21+a22)b11
        # U: 2 (a21), 3 (a22)
        # V: 4b (b11)
        # W: 2b+2 (c21)
        # Coeffs: U[2]=1, U[3]=1; V[4b]=1; W[2b+2]=1
        
        # Term 3: M3 = a11(b12-b22)
        # U: 0 (a11)
        # V: 4b+1 (b12), 4b+2 (b22)
        # W: 2b+1 (c12)
        # Coeffs: U[0]=1; V[4b+1]=1, V[4b+2]=-1; W[2b+1]=1
        
        # Term 4: M4 = a22(b21-b11)
        # U: 3 (a22)
        # V: 4b+2 (b21), 4b (b11)
        # W: 2b (c11), 2b+2 (c21)
        # Coeffs: U[3]=1; V[4b+2]=1, V[4b]=-1; W[2b]=1, W[2b+2]=1
        
        # Term 5: M5 = (a11+a12)b22
        # U: 0 (a11), 1 (a12)
        # V: 4b+2 (b22)
        # W: 2b (c11), 2b+1 (c12)
        # Coeffs: U[0]=1, U[1]=1; V[4b+2]=1; W[2b]=-1, W[2b+1]=1
        
        # Term 6: M6 = (a21-a11)(b11+b12)
        # U: 2 (a21), 0 (a11)
        # V: 4b (b11), 4b+1 (b12)
        # W: 2b+3 (c22)
        # Coeffs: U[2]=1, U[0]=-1; V[4b]=1, V[4b+1]=1; W[2b+3]=1
        
        # Term 7: M7 = (a12-a22)(b21+b22)
        # U: 1 (a12), 3 (a22)
        # V: 4b+2 (b21), 4b+2 (b22) -> Wait, b21 is index 2, b22 is index 2?
        # V indices: b11=0, b12=1, b21=2, b22=3, b31=4, b32=5.
        # b21 is 4b+2. b22 is 4b+3.
        # Correction: b21 index is 2, b22 index is 3.
        # So V: 4b+2 (b21), 4b+3 (b22).
        # W: 2b (c11), 2b+3 (c22)
        # Coeffs: U[1]=1, U[3]=-1; V[4b+2]=1, V[4b+3]=1; W[2b]=1, W[2b+3]=1
        
        # Term 8: T1 = a13 b31
        # U: 4 (a13)
        # V: 4b+4 (b31)
        # W: 2b (c11)
        # Coeffs: U[4]=1; V[4b+4]=1; W[2b]=1
        
        # Term 9: T2 = a13 b32
        # U: 4 (a13)
        # V: 4b+5 (b32)
        # W: 2b+1 (c12)
        # Coeffs: U[4]=1; V[4b+5]=1; W[2b+1]=1
        
        # Term 10: T3 = a23 b31
        # U: 5 (a23)
        # V: 4b+4 (b31)
        # W: 2b+2 (c21)
        # Coeffs: U[5]=1; V[4b+4]=1; W[2b+2]=1
        
        # Term 11: T4 = a23 b32
        # U: 5 (a23)
        # V: 4b+5 (b32)
        # W: 2b+3 (c22)
        # Coeffs: U[5]=1; V[4b+5]=1; W[2b+3]=1
        
        # Now apply these to the arrays
        b = block_idx
        c = col_offset
        
        # 1
        U[0][c] += 1; U[3][c] += 1
        V[4*b][c] += 1; V[4*b+2][c] += 1
        W[2*b][c] += 1; W[2*b+3][c] += 1
        c += 1
        
        # 2
        U[2][c] += 1; U[3][c] += 1
        V[4*b][c] += 1
        W[2*b+2][c] += 1
        c += 1
        
        # 3
        U[0][c] += 1
        V[4*b+1][c] += 1; V[4*b+2][c] -= 1
        W[2*b+1][c] += 1
        c += 1
        
        # 4
        U[3][c] += 1
        V[4*b+2][c] += 1; V[4*b][c] -= 1
        W[2*b][c] += 1; W[2*b+2][c] += 1
        c += 1
        
        # 5
        U[0][c] += 1; U[1][c] += 1
        V[4*b+2][c] += 1
        W[2*b][c] -= 1; W[2*b+1][c] += 1
        c += 1
        
        # 6
        U[2][c] += 1; U[0][c] -= 1
        V[4*b][c] += 1; V[4*b+1][c] += 1
        W[2*b+3][c] += 1
        c += 1
        
        # 7
        U[1][c] += 1; U[3][c] -= 1
        V[4*b+2][c] += 1; V[4*b+3][c] += 1
        W[2*b][c] += 1; W[2*b+3][c] += 1
        c += 1
        
        # 8
        U[4][c] += 1
        V[4*b+4][c] += 1
        W[2*b][c] += 1
        c += 1
        
        # 9
        U[4][c] += 1
        V[4*b+5][c] += 1
        W[2*b+1][c] += 1
        c += 1
        
        # 10
        U[5][c] += 1
        V[4*b+4][c] += 1
        W[2*b+2][c] += 1
        c += 1
        
        # 11
        U[5][c] += 1
        V[4*b+5][c] += 1
        W[2*b+3][c] += 1
        c += 1
        
    return {"U": U, "V": V, "W": W}
