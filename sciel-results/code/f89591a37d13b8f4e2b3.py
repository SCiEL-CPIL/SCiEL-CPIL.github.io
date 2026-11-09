import numpy as np

def construct():
    n, m, p = 2, 4, 5
    # We decompose M_{2,4,5} into two M_{2,2,5} blocks.
    # Block 1: A[:, 0:2] * B[0:2, :]
    # Block 2: A[:, 2:4] * B[2:4, :]
    
    # First, define the factors for M_{2,2,5} with rank 13.
    # Indices for M_{2,2,5}:
    # i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # U shape (4, 13), V shape (10, 13), W shape (10, 13)
    
    def get_225_factors():
        rank_225 = 13
        U_225 = [[0]*rank_225 for _ in range(4)] # i*2+j
        V_225 = [[0]*rank_225 for _ in range(10)] # j*5+k
        W_225 = [[0]*rank_225 for _ in range(10)] # k*2+i
        
        col = 0
        # Strassen for first 2 columns (k=0,1)
        # M1 = A00 * (B00 + B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A01 + A11) * B00
        # M4 = A11 * (B00 - B11)
        # M5 = (A00 + A01) * B10
        # M6 = (A10 - A00) * B01
        # M7 = (A01 - A10) * B10
        
        # U indices: 00, 01, 10, 11
        # V indices: 00, 01, 02, 03, 04, 10, 11, 12, 13, 14
        # W indices: 00, 01, 10, 11, 20, 21, 30, 31, 40, 41
        
        # M1
        U_225[0][col] = 1
        V_225[0][col] = 1 # B00
        V_225[1][col] = 1 # B11
        W_225[0][col] = 1 # C00
        W_225[1][col] = -1 # C11? No, Strassen formulas for C.
        # C00 = M1 - M2 - M4 + M7
        # C01 = M3 + M4
        # C10 = M1 + M2
        # C11 = M3 - M1 + M5 - M6
        
        # Let's use the standard Strassen mapping.
        # M1: A00, B00+B11 -> C00 (+1), C11 (-1)
        U_225[0][col] = 1
        V_225[0][col] = 1
        V_225[1][col] = 1
        W_225[0][col] = 1 # C00
        W_225[1][col] = -1 # C11
        col += 1
        
        # M2: A00+A10, B11 -> C00 (-1), C10 (+1)
        U_225[0][col] = 1
        U_225[2][col] = 1
        V_225[1][col] = 1
        W_225[0][col] = -1
        W_225[2][col] = 1 # C10
        col += 1
        
        # M3: A01+A11, B00 -> C01 (+1), C11 (+1)
        U_225[1][col] = 1
        U_225[3][col] = 1
        V_225[0][col] = 1
        W_225[1][col] = 1 # C01
        W_225[3][col] = 1 # C11
        col += 1
        
        # M4: A11, B00-B11 -> C01 (-1), C11 (-1)
        U_225[3][col] = 1
        V_225[0][col] = 1
        V_225[1][col] = -1
        W_225[1][col] = -1
        W_225[3][col] = -1
        col += 1
        
        # M5: A00+A01, B10 -> C11 (+1)
        U_225[0][col] = 1
        U_225[1][col] = 1
        V_225[6][col] = 1 # B10
        W_225[3][col] = 1
        col += 1
        
        # M6: A10-A00, B01 -> C11 (-1)
        U_225[2][col] = 1
        U_225[0][col] = -1
        V_225[7][col] = 1 # B01
        W_225[3][col] = -1
        col += 1
        
        # M7: A01-A10, B10 -> C00 (+1)
        U_225[1][col] = 1
        U_225[2][col] = -1
        V_225[6][col] = 1
        W_225[0][col] = 1
        col += 1
        
        # Extra columns k=2,3,4
        # For each k, C0k = A00*B0k + A01*B1k
        # C1k = A10*B0k + A11*B1k
        # We use 2 multiplications: P1 = A00*B0k, P2 = A01*B1k?
        # No, we can just use the terms directly.
        # Term 1: A00 * B0k -> C0k (+1)
        # Term 2: A01 * B1k -> C0k (+1)
        # Term 3: A10 * B0k -> C1k (+1)
        # Term 4: A11 * B1k -> C1k (+1)
        # This is 4 terms. Can we reduce?
        # Rank of 2x2x1 is 2? No, rank is 2?
        # A00 B0k + A01 B1k. This is dot product.
        # It requires 2 multiplications if we don't share?
        # Actually, for a single column, the tensor is rank 2?
        # $T = A_{00} \otimes B_{0k} \otimes C_{0k} + A_{01} \otimes B_{1k} \otimes C_{0k} + A_{10} \otimes B_{0k} \otimes C_{1k} + A_{11} \otimes B_{1k} \otimes C_{1k}$.
        # This can be decomposed into 2 terms:
        # $M_1 = (A_{00} B_{0k} + A_{01} B_{1k})$? No, scalar mult.
        # We need $A_{00} B_{0k}$ and $A_{01} B_{1k}$.
        # If we compute $X = A_{00} B_{0k}$ and $Y = A_{01} B_{1k}$, then $C_{0k} = X+Y$.
        # Similarly $Z = A_{10} B_{0k}$ and $W = A_{11} B_{1k}$, $C_{1k} = Z+W$.
        # This is 4 multiplications.
        # Can we do it in 2?
        # $U_1 = A_{00}+A_{01}$, $V_1 = B_{0k}+B_{1k}$?
        # $U_2 = A_{00}-A_{01}$, $V_2 = B_{0k}-B_{1k}$?
        # $U_1 V_1 - U_2 V_2 = 2(A_{00}B_{0k} + A_{01}B_{1k})$.
        # This gives $C_{0k}$ (scaled).
        # But we also need $C_{1k} = A_{10}B_{0k} + A_{11}B_{1k}$.
        # This involves different A indices.
        # So we cannot share the Strassen-like trick for the second row easily without more terms.
        # Thus, for each extra column, we likely need 4 terms?
        # If so, Rank(2,2,5) = 7 + 3*4 = 19.
        # Total Rank = 38. Still better than 40.
        
        # Let's stick to 4 terms per extra column for safety.
        for k in [2, 3, 4]:
            # Term 1: A00 * B0k -> C0k
            U_225[0][col] = 1
            V_225[k][col] = 1 # B0k is index k in V?
            # V index j*5+k. j=0, so k.
            W_225[k*2+0][col] = 1 # C0k is index k*2+0
            col += 1
            
            # Term 2: A01 * B1k -> C0k
            U_225[1][col] = 1
            V_225[5+k][col] = 1 # B1k is index 5+k
            W_225[k*2+0][col] = 1
            col += 1
            
            # Term 3: A10 * B0k -> C1k
            U_225[2][col] = 1
            V_225[k][col] = 1
            W_225[k*2+1][col] = 1
            col += 1
            
            # Term 4: A11 * B1k -> C1k
            U_225[3][col] = 1
            V_225[5+k][col] = 1
            W_225[k*2+1][col] = 1
            col += 1
            
        return U_225, V_225, W_225

    U_225, V_225, W_225 = get_225_factors()
    rank_225 = len(U_225[0])
    
    # Now assemble for 2x4x5
    rank_total = 2 * rank_225
    U = [[0]*rank_total for _ in range(8)] # 2*4
    V = [[0]*rank_total for _ in range(20)] # 4*5
    W = [[0]*rank_total for _ in range(10)] # 5*2
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    # A indices: i in {0,1}, j in {0,1}. U index i*4+j.
    # B indices: j in {0,1}, k in {0..4}. V index j*5+k.
    # C indices: k in {0..4}, i in {0,1}. W index k*2+i.
    
    for c in range(rank_225):
        for u_idx in range(4):
            i = u_idx // 2
            j_local = u_idx % 2
            j_global = j_local # j=0,1
            U[i*4 + j_global][c] = U_225[u_idx][c]
            
        for v_idx in range(10):
            j_local = v_idx // 5
            k = v_idx % 5
            j_global = j_local # j=0,1
            V[j_global*5 + k][c] = V_225[v_idx][c]
            
        for w_idx in range(10):
            k = w_idx // 2
            i = w_idx % 2
            W[k*2 + i][c] = W_225[w_idx][c]
            
    # Block 2: A[:, 2:4] * B[2:4, :]
    # A indices: i in {0,1}, j in {2,3}. U index i*4+j.
    # B indices: j in {2,3}, k in {0..4}. V index j*5+k.
    # C indices: same.
    
    offset = rank_225
    for c in range(rank_225):
        for u_idx in range(4):
            i = u_idx // 2
            j_local = u_idx % 2
            j_global = 2 + j_local # j=2,3
            U[i*4 + j_global][c + offset] = U_225[u_idx][c]
            
        for v_idx in range(10):
            j_local = v_idx // 5
            k = v_idx % 5
            j_global = 2 + j_local # j=2,3
            V[j_global*5 + k][c + offset] = V_225[v_idx][c]
            
        for w_idx in range(10):
            k = w_idx // 2
            i = w_idx % 2
            W[k*2 + i][c + offset] = W_225[w_idx][c]
            
    return {"U": U, "V": V, "W": W}
