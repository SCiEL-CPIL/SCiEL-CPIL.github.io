import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 4, 6
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication (Strassen)."""
        # Strassen's 7 products for 2x2 matrices
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        #
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 - M4 + M5 - M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M3 + M6 - M7
        #
        # U has shape (2*2, 7) = (4, 7), indexed by (i*m+j, r) where m=2
        # V has shape (2*2, 7) = (4, 7), indexed by (j*p+k, r) where p=2
        # W has shape (2*2, 7) = (4, 7), indexed by (k*n+i, r) where n=2
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (a11+a22)(b11+b22)
        # U: a11 at pos 0, a22 at pos 3
        U[0, 0] = 1; U[3, 0] = 1
        # V: b11 at pos 0, b22 at pos 3
        V[0, 0] = 1; V[3, 0] = 1
        # W: contributes to c11 (pos 0) and c22 (pos 3)
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # c21
        
        # M3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # c12
        
        # M4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1  # c11 gets -M4
        W[2, 3] = 1   # c21 gets +M4
        
        # M5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1  # c11
        W[1, 4] = 1  # c12
        
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # c22
        
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1  # c11
        W[3, 6] = -1  # c22
        
        return U, V, W
    
    def schoolbook_1xMxP(M, P):
        """Return U, V, W for 1xMxP matrix multiplication (schoolbook)."""
        # A is 1xM, B is MxP, C is 1xP
        # C[0,k] = sum_j A[0,j] B[j,k]
        # U has shape (1*M, M*P) = (M, M*P)
        # V has shape (M*P, M*P) = (M*P, M*P)
        # W has shape (P*1, M*P) = (P, M*P)
        
        rank = M * P
        U = np.zeros((M, rank), dtype=float)
        V = np.zeros((M * P, rank), dtype=float)
        W = np.zeros((P, rank), dtype=float)
        
        col = 0
        for j in range(M):
            for k in range(P):
                U[j, col] = 1
                V[j * P + k, col] = 1
                W[k, col] = 1
                col += 1
        
        return U, V, W
    
    # Build 2x4x6 using Strassen for the 2x2 blocks
    # Decompose into two 2x2x? problems
    # Actually, let's use the approach: R(2,4,6) = 2 * R(2,2,6)
    # And R(2,2,6) = R(2,2,4) + R(2,2,2) = 2*R(2,2,2) + R(2,2,2) = 3*7 = 21
    
    # Let's build 2x2x6 first
    def build_2x2x6():
        """Build 2x2x6 using two Strassen decompositions."""
        # Split B into B1 (2x4) and B2 (2x2)
        # C1 = A * B1 (2x4), C2 = A * B2 (2x2)
        # C = [C1, C2]
        
        # For C1 = A * B1, we need 2x2x4
        # 2x2x4 can be done with 2 Strassen blocks (each 2x2x2)
        # But actually, 2x2x4: A is 2x2, B is 2x4, C is 2x4
        # We can split B into two 2x2 blocks: B = [B_left, B_right]
        # C_left = A * B_left, C_right = A * B_right
        # Each is 2x2x2, so each uses 7 products
        # Total: 14 products
        
        U_222, V_222, W_222 = strassen_2x2()
        
        # For 2x2x4:
        # U is (4, 14), V is (8, 14), W is (8, 14)
        U_224 = np.zeros((4, 14), dtype=float)
        V_224 = np.zeros((8, 14), dtype=float)
        W_224 = np.zeros((8, 14), dtype=float)
        
        # First half: B_left (cols 0,1 of B)
        U_224[:, 0:7] = U_222
        V_224[0:4, 0:7] = V_222  # B_left has shape 2x2, indexed by j*2+k for j in 0,1, k in 0,1
        W_224[0:4, 0:7] = W_222  # C_left has shape 2x2, indexed by k*2+i for k in 0,1, i in 0,1
        
        # Second half: B_right (cols 2,3 of B)
        U_224[:, 7:14] = U_222
        V_224[4:8, 7:14] = V_222  # B_right: j*2+k for j in 0,1, k in 2,3 -> offset by 4
        W_224[4:8, 7:14] = W_222  # C_right: k*2+i for k in 2,3, i in 0,1 -> offset by 4
        
        # Now 2x2x6:
        # Split B (2x6) into B1 (2x4) and B2 (2x2)
        # C1 = A * B1 (2x4), C2 = A * B2 (2x2)
        # C = [C1, C2]
        
        U_226 = np.zeros((4, 21), dtype=float)
        V_226 = np.zeros((12, 21), dtype=float)
        W_226 = np.zeros((12, 21), dtype=float)
        
        # First part: 2x2x4
        U_226[:, 0:14] = U_224
        V_226[0:8, 0:14] = V_224  # B1: j*6+k for j in 0,1, k in 0..3 -> rows 0..7
        W_226[0:8, 0:14] = W_224  # C1: k*2+i for k in 0..3, i in 0,1 -> rows 0..7
        
        # Second part: 2x2x2
        U_226[:, 14:21] = U_222
        V_226[8:12, 14:21] = V_222  # B2: j*6+k for j in 0,1, k in 4,5 -> rows 8..11
        W_226[8:12, 14:21] = W_222  # C2: k*2+i for k in 4,5, i in 0,1 -> rows 8..11
        
        return U_226, V_226, W_226
    
    # Build 2x4x6
    def build_2x4x6():
        """Build 2x4x6 using two 2x2x6 decompositions."""
        U_226, V_226, W_226 = build_2x2x6()
        
        # A is 2x4, split into A_left (2x2) and A_right (2x2)
        # C_left = A_left * B (2x6), C_right = A_right * B (2x6)
        # C = [C_left; C_right] (stacked vertically)
        
        U_246 = np.zeros((8, 42), dtype=float)
        V_246 = np.zeros((24, 42), dtype=float)
        W_246 = np.zeros((12, 42), dtype=float)
        
        # First part: A_left (rows 0,1 of A)
        U_246[0:4, 0:21] = U_226
        V_246[0:12, 0:21] = V_226  # B: j*6+k for j in 0,1, k in 0..5 -> rows 0..11
        W_246[0:12, 0:21] = W_226  # C_left: k*2+i for k in 0..5, i in 0,1 -> rows 0..11
        
        # Second part: A_right (rows 2,3 of A)
        U_246[4:8, 21:42] = U_226
        V_246[12:24, 21:42] = V_226  # B: j*6+k for j in 2,3, k in 0..5 -> rows 12..23
        W_246[0:12, 21:42] = W_226  # C_right: k*2+i for k in 0..5, i in 2,3 -> rows 12..23? 
        # Wait, W is indexed by k*n+i where n=2 for the 2x2x6 case, but here n=2 for the whole 2x4x6
        # Actually, for 2x4x6, W has shape (6*2, 42) = (12, 42), indexed by k*2+i for k in 0..5, i in 0,1
        # C_left corresponds to i=0,1, C_right to i=2,3? No, that's not right.
        
        # Let me reconsider. For 2x4x6:
        # A is 2x4, B is 4x6, C is 2x6
        # C[i,k] = sum_j A[i,j] B[j,k]
        # U is (2*4, rank) = (8, rank), indexed by i*4+j
        # V is (4*6, rank) = (24, rank), indexed by j*6+k
        # W is (6*2, rank) = (12, rank), indexed by k*2+i
        
        # If we split A into A_left (2x2, cols 0,1) and A_right (2x2, cols 2,3):
        # C = A_left * B_left + A_right * B_right
        # where B_left is B[:, 0:2] (4x2) and B_right is B[:, 2:4] (4x4)? No.
        
        # Actually, let's split differently. Let's split B into two 2x6 blocks:
        # B_top = B[0:2, :] (2x6), B_bottom = B[2:4, :] (2x6)
        # C = A_top * B_top + A_bottom * B_bottom
        # where A_top = A[:, 0:2] (2x2) and A_bottom = A[:, 2:4] (2x2)
        
        # So:
        # C = A_top * B_top + A_bottom * B_bottom
        # Each term is 2x2 * 2x6 = 2x6, which is a 2x2x6 problem
        
        # For A_top * B_top:
        # U: A_top entries at i*4+j for i in 0,1, j in 0,1 -> rows 0,1,4,5
        # V: B_top entries at j*6+k for j in 0,1, k in 0..5 -> rows 0..11
        # W: C entries at k*2+i for k in 0..5, i in 0,1 -> rows 0..11
        
        # For A_bottom * B_bottom:
        # U: A_bottom entries at i*4+j for i in 0,1, j in 2,3 -> rows 2,3,6,7
        # V: B_bottom entries at j*6+k for j in 2,3, k in 0..5 -> rows 12..23
        # W: C entries at k*2+i for k in 0..5, i in 0,1 -> rows 0..11
        
        U_246 = np.zeros((8, 42), dtype=float)
        V_246 = np.zeros((24, 42), dtype=float)
        W_246 = np.zeros((12, 42), dtype=float)
        
        # First part: A_top * B_top
        U_246[0, 0:21] = U_226[0, :]  # A[0,0]
        U_246[1, 0:21] = U_226[1, :]  # A[0,1]
        U_246[4, 0:21] = U_226[2, :]  # A[1,0]
        U_246[5, 0:21] = U_226[3, :]  # A[1,1]
        V_246[0:12, 0:21] = V_226  # B_top
        W_246[0:12, 0:21] = W_226  # C
        
        # Second part: A_bottom * B_bottom
        U_246[2, 21:42] = U_226[0, :]  # A[0,2]
        U_246[3, 21:42] = U_226[1, :]  # A[0,3]
        U_246[6, 21:42] = U_226[2, :]  # A[1,2]
        U_246[7, 21:42] = U_226[3, :]  # A[1,3]
        V_246[12:24, 21:42] = V_226  # B_bottom
        W_246[0:12, 21:42] = W_226  # C
        
        return U_246, V_246, W_246
    
    # Build 1x4x6 (schoolbook)
    U_146, V_146, W_146 = schoolbook_1xMxP(4, 6)
    
    # Build 3x4x6 = 2x4x6 + 1x4x6
    U_246, V_246, W_246 = build_2x4x6()
    
    rank = 42 + 24  # 42 from 2x4x6, 24 from 1x4x6
    U = np.zeros((12, rank), dtype=float)  # 3*4 = 12
    V = np.zeros((24, rank), dtype=float)  # 4*6 = 24
    W = np.zeros((18, rank), dtype=float)  # 6*3 = 18
    
    # First part: 2x4x6
    U[0:8, 0:42] = U_246
    V[0:24, 0:42] = V_246
    W[0:12, 0:42] = W_246  # C for 2x4x6: k*2+i for k in 0..5, i in 0,1 -> rows 0..11
    
    # Second part: 1x4x6
    U[8:12, 42:66] = U_146
    V[0:24, 42:66] = V_146
    W[12:18, 42:66] = W_146  # C for 1x4x6: k*1+i for k in 0..5, i in 0 -> rows 0..5, but offset by 12
    
    # Wait, W for 3x4x6 is indexed by k*3+i for k in 0..5, i in 0,1,2
    # For the 2x4x6 part, i is 0,1, so W[k*3+i] for k in 0..5, i in 0,1 -> rows 0,1, 3,4, 6,7, 9,10, 12,13, 15,16
    # For the 1x4x6 part, i is 2, so W[k*3+2] for k in 0..5 -> rows 2, 5, 8, 11, 14, 17
    
    # Let me redo this with proper indexing
    
    U = np.zeros((12, 66), dtype=float)
    V = np.zeros((24, 66), dtype=float)
    W = np.zeros((18, 66), dtype=float)
    
    # 2x4x6 part: i in 0,1
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:42] = U_246[i*4+j, :]
    V[0:24, 0:42] = V_246
    for k in range(6):
        for i in range(2):
            W[k*3+i, 0:42] = W_246[k*2+i, :]
    
    # 1x4x6 part: i=2
    for j in range(4):
        U[2*4+j, 42:66] = U_146[j, :]
    V[0:24, 42:66] = V_146
    for k in range(6):
        W[k*3+2, 42:66] = W_146[k, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
