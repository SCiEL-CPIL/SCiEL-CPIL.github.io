"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a schoolbook decomposition for n, m, p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen-like decomposition for 2, 2, 2
    def strassen_222():
        # Rank 7
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Strassen's 7 multiplications
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Indices: A00=0, A01=1, A10=2, A11=3
        # B00=0, B01=1, B10=2, B11=3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W

    # Helper to create decomposition for 2, 1, 2 (Rank 4)
    def dec_212():
        rank = 4
        U = np.zeros((2, rank))
        V = np.zeros((2, rank))
        W = np.zeros((4, rank))
        
        # C = A * B, A is 2x1, B is 1x2
        # C00 = A0*B0, C01 = A0*B1, C10 = A1*B0, C11 = A1*B1
        # T = e0⊗e0⊗e00 + e0⊗e1⊗e01 + e1⊗e0⊗e10 + e1⊗e1⊗e11
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W

    # Helper to create decomposition for 2, 2, 1 (Rank 4)
    def dec_221():
        rank = 4
        U = np.zeros((4, rank))
        V = np.zeros((2, rank))
        W = np.zeros((2, rank))
        
        # C = A * B, A is 2x2, B is 2x1
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # T = e0⊗e0⊗e0 + e1⊗e0⊗e0 + e2⊗e1⊗e1 + e3⊗e1⊗e1
        # Wait, indices: A is 2x2 (4 entries), B is 2x1 (2 entries), C is 2x1 (2 entries).
        # A00=0, A01=1, A10=2, A11=3
        # B0=0, B1=1
        # C0=0, C1=1
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Helper to create decomposition for 1, 2, 2 (Rank 4)
    def dec_122():
        rank = 4
        U = np.zeros((2, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # A is 1x2, B is 2x2, C is 1x2
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Helper to create decomposition for 2, 1, 1 (Rank 2)
    def dec_211():
        rank = 2
        U = np.zeros((2, rank))
        V = np.zeros((1, rank))
        W = np.zeros((2, rank))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
        
        return U, V, W

    # Helper to create decomposition for 1, 2, 1 (Rank 2)
    def dec_121():
        rank = 2
        U = np.zeros((2, rank))
        V = np.zeros((2, rank))
        W = np.zeros((2, rank))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        
        return U, V, W

    # Helper to create decomposition for 1, 1, 2 (Rank 2)
    def dec_112():
        rank = 2
        U = np.zeros((1, rank))
        V = np.zeros((2, rank))
        W = np.zeros((2, rank))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        
        return U, V, W

    # Helper to create decomposition for 1, 1, 1 (Rank 1)
    def dec_111():
        rank = 1
        U = np.ones((1, rank))
        V = np.ones((1, rank))
        W = np.ones((1, rank))
        return U, V, W

    # Construct M_333 using block decomposition
    def construct_333():
        # Blocks:
        # A22 B22 (2,2,2) -> Rank 7
        # A21 B12 (2,1,2) -> Rank 4
        # A22 B21 (2,2,1) -> Rank 4
        # A12 B22 (1,2,2) -> Rank 4
        # A21 B11 (2,1,1) -> Rank 2
        # A12 B21 (1,2,1) -> Rank 2
        # A11 B12 (1,1,2) -> Rank 2
        # A11 B11 (1,1,1) -> Rank 1
        
        U_list = []
        V_list = []
        W_list = []
        
        # 1. A22 B22
        U1, V1, W1 = strassen_222()
        # Map indices to 3x3 blocks
        # A22 is rows 0,1 and cols 0,1 of A? No.
        # A is 3x3. A22 is top-left 2x2?
        # Let's define A indices:
        # 0,0 0,1 0,2
        # 1,0 1,1 1,2
        # 2,0 2,1 2,2
        # A22: rows 0,1; cols 0,1.
        # A21: rows 0,1; col 2.
        # A12: row 2; cols 0,1.
        # A11: row 2; col 2.
        
        # B is 3x3.
        # B22: rows 0,1; cols 0,1.
        # B21: rows 0,1; col 2.
        # B12: row 2; cols 0,1.
        # B11: row 2; col 2.
        
        # C is 3x3.
        # C22: rows 0,1; cols 0,1.
        # C21: rows 0,1; col 2.
        # C12: row 2; cols 0,1.
        # C11: row 2; col 2.
        
        # We need to embed the small decompositions into the 3x3x3 space.
        
        def embed(U_small, V_small, W_small, u_rows, v_rows, w_rows):
            # u_rows: list of indices in U (9)
            # v_rows: list of indices in V (9)
            # w_rows: list of indices in W (9)
            rank = U_small.shape[1]
            U_full = np.zeros((9, rank))
            V_full = np.zeros((9, rank))
            W_full = np.zeros((9, rank))
            
            for c in range(rank):
                for i, u_idx in enumerate(u_rows):
                    U_full[u_idx, c] = U_small[i, c]
                for i, v_idx in enumerate(v_rows):
                    V_full[v_idx, c] = V_small[i, c]
                for i, w_idx in enumerate(w_rows):
                    W_full[w_idx, c] = W_small[i, c]
            return U_full, V_full, W_full

        # 1. A22 B22 (2,2,2)
        u_rows = [0, 1, 3, 4] # A00, A01, A10, A11
        v_rows = [0, 1, 3, 4] # B00, B01, B10, B11
        w_rows = [0, 1, 3, 4] # C00, C01, C10, C11
        U1, V1, W1 = embed(U1, V1, W1, u_rows, v_rows, w_rows)
        U_list.append(U1); V_list.append(V1); W_list.append(W1)
        
        # 2. A21 B12 (2,1,2)
        # A21: rows 0,1; col 2. Indices 2, 5.
        # B12: row 2; cols 0,1. Indices 6, 7.
        # C22: rows 0,1; cols 0,1. Indices 0, 1, 3, 4.
        # Wait, C22 is the result of A21*B12?
        # C = A B.
        # C_ij = sum_k A_ik B_kj.
        # C_00 = A_00 B_00 + A_01 B_10 + A_02 B_20.
        # My block notation:
        # A22 is top-left 2x2.
        # A21 is top-right 2x1.
        # A12 is bottom-left 1x2.
        # A11 is bottom-right 1x1.
        
        # B22 is top-left 2x2.
        # B21 is top-right 2x1.
        # B12 is bottom-left 1x2.
        # B11 is bottom-right 1x1.
        
        # C22 = A22 B22 + A21 B12.
        # C21 = A22 B21 + A21 B11.
        # C12 = A12 B22 + A11 B12.
        # C11 = A12 B21 + A11 B11.
        
        # So for term A21 B12:
        # U indices (A21): 2, 5.
        # V indices (B12): 6, 7.
        # W indices (C22): 0, 1, 3, 4.
        
        u_rows = [2, 5]
        v_rows = [6, 7]
        w_rows = [0, 1, 3, 4]
        U2, V2, W2 = dec_212()
        U2, V2, W2 = embed(U2, V2, W2, u_rows, v_rows, w_rows)
        U_list.append(U2); V_list.append(V2); W_list.append(W2)
        
        # 3. A22 B21 (2,2,1)
        # U indices (A22): 0, 1, 3, 4.
        # V indices (B21): 2, 5.
        # W indices (C21): 2, 5.
        
        u_rows = [0, 1, 3, 4]
        v_rows = [2, 5]
        w_rows = [2, 5]
        U3, V3, W3 = dec_221()
        U3, V3, W3 = embed(U3, V3, W3, u_rows, v_rows, w_rows)
        U_list.append(U3); V_list.append(V3); W_list.append(W3)
        
        # 4. A12 B22 (1,2,2)
        # U indices (A12): 6, 7.
        # V indices (B22): 0, 1, 3, 4.
        # W indices (C12): 6, 7.
        
        u_rows = [6, 7]
        v_rows = [0, 1, 3, 4]
        w_rows = [6, 7]
        U4, V4, W4 = dec_122()
        U4, V4, W4 = embed(U4, V4, W4, u_rows, v_rows, w_rows)
        U_list.append(U4); V_list.append(V4); W_list.append(W4)
        
        # 5. A21 B11 (2,1,1)
        # U indices (A21): 2, 5.
        # V indices (B11): 8.
        # W indices (C21): 2, 5.
        
        u_rows = [2, 5]
        v_rows = [8]
        w_rows = [2, 5]
        U5, V5, W5 = dec_211()
        U5, V5, W5 = embed(U5, V5, W5, u_rows, v_rows, w_rows)
        U_list.append(U5); V_list.append(V5); W_list.append(W5)
        
        # 6. A12 B21 (1,2,1)
        # U indices (A12): 6, 7.
        # V indices (B21): 2, 5.
        # W indices (C12): 6, 7.
        
        u_rows = [6, 7]
        v_rows = [2, 5]
        w_rows = [6, 7]
        U6, V6, W6 = dec_121()
        U6, V6, W6 = embed(U6, V6, W6, u_rows, v_rows, w_rows)
        U_list.append(U6); V_list.append(V6); W_list.append(W6)
        
        # 7. A11 B12 (1,1,2)
        # U indices (A11): 8.
        # V indices (B12): 6, 7.
        # W indices (C12): 6, 7.
        
        u_rows = [8]
        v_rows = [6, 7]
        w_rows = [6, 7]
        U7, V7, W7 = dec_112()
        U7, V7, W7 = embed(U7, V7, W7, u_rows, v_rows, w_rows)
        U_list.append(U7); V_list.append(V7); W_list.append(W7)
        
        # 8. A11 B11 (1,1,1)
        # U indices (A11): 8.
        # V indices (B11): 8.
        # W indices (C11): 8.
        
        u_rows = [8]
        v_rows = [8]
        w_rows = [8]
        U8, V8, W8 = dec_111()
        U8, V8, W8 = embed(U8, V8, W8, u_rows, v_rows, w_rows)
        U_list.append(U8); V_list.append(V8); W_list.append(W8)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        return U, V, W

    # Construct M_331 using schoolbook
    def construct_331():
        return schoolbook(3, 3, 1)

    # Combine for M_337: 3 + 3 + 1
    U333, V333, W333 = construct_333()
    U331, V331, W331 = construct_331()
    
    # We need to map the indices for the second 333 block and the 331 block.
    # The full V is 3x7 = 21 rows.
    # The full W is 7x3 = 21 rows.
    # The full U is 3x3 = 9 rows.
    
    # Block 1: M_333 (cols 0-2 of B, cols 0-2 of C)
    # Block 2: M_333 (cols 3-5 of B, cols 3-5 of C)
    # Block 3: M_331 (col 6 of B, col 6 of C)
    
    U_full = np.zeros((9, 61))
    V_full = np.zeros((21, 61))
    W_full = np.zeros((21, 61))
    
    # Block 1
    U_full[:, 0:26] = U333
    V_full[0:9, 0:26] = V333
    W_full[0:9, 0:26] = W333
    
    # Block 2
    U_full[:, 26:52] = U333
    V_full[9:18, 26:52] = V333
    W_full[9:18, 26:52] = W333
    
    # Block 3
    U_full[:, 52:61] = U331
    V_full[18:21, 52:61] = V331
    W_full[18:21, 52:61] = W331
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# EVOLVE-BLOCK-END
