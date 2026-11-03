"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)

    # Helper to create a Strassen decomposition for 2x2x2
    def strassen_2x2x2(offset_a=0, offset_b=0, offset_c=0):
        """
        Returns U, V, W for 2x2x2 multiplication.
        Indices:
        a: 0,1 (local i) -> global i
        b: 0,1 (local j) -> global j
        c: 0,1 (local k) -> global k
        
        Global indices:
        A_idx = offset_a + i*2 + j
        B_idx = offset_b + j*2 + k  (Wait, B is j x k. j is row, k is col)
        C_idx = offset_c + k*2 + i
        
        Strassen formulas:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
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
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1
        
        # Apply offsets
        # U index: i*2 + j. Local i,j in 0..1.
        # Global A index: offset_a + i*2 + j? 
        # Wait, A is 2x2. i is row, j is col.
        # In our global mapping, A is n x m.
        # Here we are decomposing a sub-problem.
        # We just return local U, V, W. The caller will embed them.
        return U, V, W

    def schoolbook_2x2x1(offset_a=0, offset_b=0, offset_c=0):
        """
        Rank 4 decomposition for 2x2x1.
        C00 = A00 B00
        C01 = A01 B00 (Wait, B is 2x1. B00, B10)
        C10 = A10 B00
        C11 = A11 B00
        """
        U = np.zeros((4, 4))
        V = np.zeros((2, 4)) # B is 2x1, so 2 entries
        W = np.zeros((2, 4)) # C is 2x1, so 2 entries
        
        # M1: A00 B00 -> C00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # M2: A01 B00 -> C01? No.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # For 2x2x1:
        # C_{00} = A_{00}B_{00} + A_{01}B_{10}
        # C_{10} = A_{10}B_{00} + A_{11}B_{10}
        # This is rank 4.
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # A00 B00 -> C00
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1 # A01 B10 -> C00
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1 # A10 B00 -> C10
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1 # A11 B10 -> C10
        
        return U, V, W

    def build_2x2x7():
        """
        Build rank 25 decomposition for 2x2x7.
        Split k into 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6 (Schoolbook).
        """
        U_list = []
        V_list = []
        W_list = []
        
        # Block 1: k=0,1. Strassen.
        U1, V1, W1 = strassen_2x2x2()
        # Embed into global 2x2x7
        # U is 4x7. V is 14x7 (2x7). W is 14x7 (7x2).
        # Local U: 4x7. Global U: 4x7.
        # Local V: 4x7. Global V: 14x7.
        #   Local V row j*2+k.
        #   Global V row j*7+k.
        # Local W: 4x7. Global W: 14x7.
        #   Local W row k*2+i.
        #   Global W row k*2+i? No, Global W is p*n = 7*2 = 14.
        #   Global W row k*n+i = k*2+i.
        
        U_g1 = np.zeros((4, 7))
        U_g1[:, :] = U1
        
        V_g1 = np.zeros((14, 7))
        for j in range(2):
            for k in range(2):
                V_g1[j*7+k, :] = V1[j*2+k, :]
        
        W_g1 = np.zeros((14, 7))
        for k in range(2):
            for i in range(2):
                W_g1[k*2+i, :] = W1[k*2+i, :]
                
        U_list.append(U_g1)
        V_list.append(V_g1)
        W_list.append(W_g1)
        
        # Block 2: k=2,3. Strassen.
        U2, V2, W2 = strassen_2x2x2()
        U_g2 = np.zeros((4, 7))
        U_g2[:, :] = U2
        
        V_g2 = np.zeros((14, 7))
        for j in range(2):
            for k in range(2):
                V_g2[j*7+(k+2), :] = V2[j*2+k, :]
        
        W_g2 = np.zeros((14, 7))
        for k in range(2):
            for i in range(2):
                W_g2[(k+2)*2+i, :] = W2[k*2+i, :]
                
        U_list.append(U_g2)
        V_list.append(V_g2)
        W_list.append(W_g2)
        
        # Block 3: k=4,5. Strassen.
        U3, V3, W3 = strassen_2x2x2()
        U_g3 = np.zeros((4, 7))
        U_g3[:, :] = U3
        
        V_g3 = np.zeros((14, 7))
        for j in range(2):
            for k in range(2):
                V_g3[j*7+(k+4), :] = V3[j*2+k, :]
        
        W_g3 = np.zeros((14, 7))
        for k in range(2):
            for i in range(2):
                W_g3[(k+4)*2+i, :] = W3[k*2+i, :]
                
        U_list.append(U_g3)
        V_list.append(V_g3)
        W_list.append(W_g3)
        
        # Block 4: k=6. Schoolbook 2x2x1.
        U4, V4, W4 = schoolbook_2x2x1()
        # U4 is 4x4. V4 is 2x4. W4 is 2x4.
        U_g4 = np.zeros((4, 4))
        U_g4[:, :] = U4
        
        V_g4 = np.zeros((14, 4))
        for j in range(2):
            V_g4[j*7+6, :] = V4[j, :]
            
        W_g4 = np.zeros((14, 4))
        for k in range(1): # k=6
            for i in range(2):
                W_g4[6*2+i, :] = W4[i, :] # W4 row i corresponds to C_{i0}
                
        U_list.append(U_g4)
        V_list.append(V_g4)
        W_list.append(W_g4)
        
        U_total = np.hstack(U_list)
        V_total = np.hstack(V_list)
        W_total = np.hstack(W_list)
        
        return U_total, V_total, W_total

    # Build 2x4x7
    # Split j into 0,1 and 2,3.
    # Part 1: j=0,1. 2x2x7.
    # Part 2: j=2,3. 2x2x7.
    
    U_p1, V_p1, W_p1 = build_2x2x7()
    # Embed Part 1 into 2x4x7
    # U: 4x25. Global U: 8x25.
    #   Local U row i*2+j. Global U row i*4+j.
    U_g1 = np.zeros((8, 25))
    for i in range(2):
        for j in range(2):
            U_g1[i*4+j, :] = U_p1[i*2+j, :]
            
    # V: 14x25. Global V: 28x25.
    #   Local V row j*7+k. Global V row j*7+k.
    V_g1 = np.zeros((28, 25))
    for j in range(2):
        for k in range(7):
            V_g1[j*7+k, :] = V_p1[j*7+k, :]
            
    # W: 14x25. Global W: 14x25.
    #   Local W row k*2+i. Global W row k*2+i.
    W_g1 = np.zeros((14, 25))
    for k in range(7):
        for i in range(2):
            W_g1[k*2+i, :] = W_p1[k*2+i, :]
            
    # Part 2: j=2,3.
    U_p2, V_p2, W_p2 = build_2x2x7()
    U_g2 = np.zeros((8, 25))
    for i in range(2):
        for j in range(2):
            U_g2[i*4+(j+2), :] = U_p2[i*2+j, :]
            
    V_g2 = np.zeros((28, 25))
    for j in range(2):
        for k in range(7):
            V_g2[(j+2)*7+k, :] = V_p2[j*7+k, :]
            
    W_g2 = np.zeros((14, 25))
    for k in range(7):
        for i in range(2):
            W_g2[k*2+i, :] = W_p2[k*2+i, :]
            
    U_final = np.hstack([U_g1, U_g2])
    V_final = np.hstack([V_g1, V_g2])
    W_final = np.hstack([W_g1, W_g2])
    
    # Verify
    T_rec = tensor_from_factors(U_final, V_final, W_final)
    if not np.allclose(T_rec, T, atol=1e-8):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
