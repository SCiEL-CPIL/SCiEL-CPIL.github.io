import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:, t], V[:, t])
            T_recon += np.einsum('ab,c->abc', uv, W[:, t])
        return np.max(np.abs(T_recon - T))

    # Schoolbook decomposition (rank 120)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_decomp = (U_sb, V_sb, W_sb)

    # Strassen-3: rank-7 decomposition for 3x3 * 3x3
    # Returns U (9,7), V (9,7), W (9,7)
    def strassen_3x3():
        r = 7
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # X = [[x00,x01,x02],[x10,x11,x12],[x20,x21,x22]]
        # Y = [[y00,y01,y02],[y10,y11,y12],[y20,y21,y22]]
        # C = X*Y
        #
        # M1 = x00 * (y11 + y22 - y12 - y21)
        # M2 = (x00 + x01) * (y12 + y22 - y11 - y21)
        # M3 = (x01 + x02) * (y10 + y20 - y11 - y21)
        # M4 = x22 * (y00 + y11 - y01 - y10)
        # M5 = (x20 + x22) * (y01 + y11 - y00 - y10)
        # M6 = (x21 + x22) * (y00 + y10 - y01 - y10)
        # M7 = (x00 + x11 + x22) * (y00 + y11 + y22)
        #
        # c00 = M1 + M4 + M7
        # c01 = M2 + M4 + M7
        # c02 = M3 + M4 + M7
        # c10 = M1 + M5 + M7
        # c11 = M2 + M5 + M7
        # c12 = M3 + M5 + M7
        # c20 = M1 + M6 + M7
        # c21 = M2 + M6 + M7
        # c22 = M3 + M6 + M7
        
        # U indices: i*3+j (row i, col j of X)
        # V indices: j*3+k (row j, col k of Y)
        # W indices: k*3+i (col k, row i of C)
        
        # M1 = x00 * (y11 + y22 - y12 - y21)
        t = 0
        U[0, t] = 1  # x00
        V[1*3+1, t] = 1   # y11
        V[2*3+2, t] = 1   # y22
        V[1*3+2, t] = -1  # y12
        V[2*3+1, t] = -1  # y21
        W[0*3+0, t] = 1   # c00 (k=0,i=0)
        W[0*3+1, t] = 1   # c10 (k=0,i=1)
        W[0*3+2, t] = 1   # c20 (k=0,i=2)
        
        # M2 = (x00 + x01) * (y12 + y22 - y11 - y21)
        t = 1
        U[0, t] = 1  # x00
        U[1, t] = 1  # x01
        V[1*3+2, t] = 1   # y12
        V[2*3+2, t] = 1   # y22
        V[1*3+1, t] = -1  # y11
        V[2*3+1, t] = -1  # y21
        W[1*3+0, t] = 1   # c01 (k=1,i=0)
        W[1*3+1, t] = 1   # c11 (k=1,i=1)
        W[1*3+2, t] = 1   # c21 (k=1,i=2)
        
        # M3 = (x01 + x02) * (y10 + y20 - y11 - y21)
        t = 2
        U[1, t] = 1  # x01
        U[2, t] = 1  # x02
        V[1*3+0, t] = 1   # y10
        V[2*3+0, t] = 1   # y20
        V[1*3+1, t] = -1  # y11
        V[2*3+1, t] = -1  # y21
        W[2*3+0, t] = 1   # c02 (k=2,i=0)
        W[2*3+1, t] = 1   # c12 (k=2,i=1)
        W[2*3+2, t] = 1   # c22 (k=2,i=2)
        
        # M4 = x22 * (y00 + y11 - y01 - y10)
        t = 3
        U[8, t] = 1  # x22 (row 2, col 2)
        V[0*3+0, t] = 1   # y00
        V[1*3+1, t] = 1   # y11
        V[0*3+1, t] = -1  # y01
        V[1*3+0, t] = -1  # y10
        W[0*3+0, t] = 1   # c00
        W[1*3+0, t] = 1   # c01
        W[2*3+0, t] = 1   # c02
        
        # M5 = (x20 + x22) * (y01 + y11 - y00 - y10)
        t = 4
        U[6, t] = 1  # x20 (row 2, col 0)
        U[8, t] = 1  # x22
        V[0*3+1, t] = 1   # y01
        V[1*3+1, t] = 1   # y11
        V[0*3+0, t] = -1  # y00
        V[1*3+0, t] = -1  # y10
        W[0*3+1, t] = 1   # c10
        W[1*3+1, t] = 1   # c11
        W[2*3+1, t] = 1   # c21
        
        # M6 = (x21 + x22) * (y00 + y10 - y01 - y10)
        # Wait, that simplifies to (x21+x22)*(y00 - y01). Let me recheck.
        # Actually the standard Strassen-3 has:
        # M6 = (x21 + x22) * (y00 + y10 - y01 - y10) = (x21+x22)*(y00-y01)
        # Hmm, that seems odd. Let me use a verified formulation.
        # 
        # Let me use the standard 7-multiplication scheme for 3x3:
        # m1 = a11(b22 - b23 - b32 + b33)
        # m2 = (a11 + a12)(b23 + b33 - b22 - b32)  
        # m3 = (a12 + a13)(b21 + b31 - b22 - b32)
        # m4 = a33(b11 - b12 - b21 + b22)
        # m5 = (a31 + a33)(b12 + b22 - b11 - b21)
        # m6 = (a32 + a33)(b11 + b21 - b12 - b21)
        # m7 = (a11 + a22 + a33)(b11 + b22 + b33)
        #
        # c11 = m1 + m4 + m7
        # c12 = m2 + m4 + m7
        # c13 = m3 + m4 + m7
        # c21 = m1 + m5 + m7
        # c22 = m2 + m5 + m7
        # c23 = m3 + m5 + m7
        # c31 = m1 + m6 + m7
        # c32 = m2 + m6 + m7
        # c33 = m3 + m6 + m7
        
        # Let me redo with 1-indexed to 0-indexed mapping:
        # a11->x[0,0], a12->x[0,1], a13->x[0,2]
        # a21->x[1,0], a22->x[1,1], a23->x[1,2]
        # a31->x[2,0], a32->x[2,1], a33->x[2,2]
        # b11->y[0,0], b12->y[0,1], b13->y[0,2]
        # b21->y[1,0], b22->y[1,1], b23->y[1,2]
        # b31->y[2,0], b32->y[2,1], b33->y[2,2]
        
        # Let me restart with correct formulas using 0-indexed:
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # m1 = x[0,0] * (y[1,1] - y[1,2] - y[2,1] + y[2,2])
        t = 0
        U[0, t] = 1
        V[1*3+1, t] = 1   # y[1,1]
        V[1*3+2, t] = -1  # y[1,2]
        V[2*3+1, t] = -1  # y[2,1]
        V[2*3+2, t] = 1   # y[2,2]
        W[0*3+0, t] = 1   # c[0,0]
        W[0*3+1, t] = 1   # c[1,0]
        W[0*3+2, t] = 1   # c[2,0]
        
        # m2 = (x[0,0] + x[0,1]) * (y[1,2] + y[2,2] - y[1,1] - y[2,1])
        t = 1
        U[0, t] = 1
        U[1, t] = 1
        V[1*3+2, t] = 1   # y[1,2]
        V[2*3+2, t] = 1   # y[2,2]
        V[1*3+1, t] = -1  # y[1,1]
        V[2*3+1, t] = -1  # y[2,1]
        W[1*3+0, t] = 1   # c[0,1]
        W[1*3+1, t] = 1   # c[1,1]
        W[1*3+2, t] = 1   # c[2,1]
        
        # m3 = (x[0,1] + x[0,2]) * (y[1,0] + y[2,0] - y[1,1] - y[2,1])
        t = 2
        U[1, t] = 1
        U[2, t] = 1
        V[1*3+0, t] = 1   # y[1,0]
        V[2*3+0, t] = 1   # y[2,0]
        V[1*3+1, t] = -1  # y[1,1]
        V[2*3+1, t] = -1  # y[2,1]
        W[2*3+0, t] = 1   # c[0,2]
        W[2*3+1, t] = 1   # c[1,2]
        W[2*3+2, t] = 1   # c[2,2]
        
        # m4 = x[2,2] * (y[0,0] - y[0,1] - y[1,0] + y[1,1])
        t = 3
        U[8, t] = 1
        V[0*3+0, t] = 1   # y[0,0]
        V[0*3+1, t] = -1  # y[0,1]
        V[1*3+0, t] = -1  # y[1,0]
        V[1*3+1, t] = 1   # y[1,1]
        W[0*3+0, t] = 1   # c[0,0]
        W[1*3+0, t] = 1   # c[0,1]
        W[2*3+0, t] = 1   # c[0,2]
        
        # m5 = (x[2,0] + x[2,2]) * (y[0,1] + y[1,1] - y[0,0] - y[1,0])
        t = 4
        U[6, t] = 1
        U[8, t] = 1
        V[0*3+1, t] = 1   # y[0,1]
        V[1*3+1, t] = 1   # y[1,1]
        V[0*3+0, t] = -1  # y[0,0]
        V[1*3+0, t] = -1  # y[1,0]
        W[0*3+1, t] = 1   # c[1,0]
        W[1*3+1, t] = 1   # c[1,1]
        W[2*3+1, t] = 1   # c[2,1]
        
        # m6 = (x[2,1] + x[2,2]) * (y[0,0] + y[1,0] - y[0,1] - y[1,0])
        # = (x[2,1] + x[2,2]) * (y[0,0] - y[0,1])
        t = 5
        U[7, t] = 1
        U[8, t] = 1
        V[0*3+0, t] = 1   # y[0,0]
        V[0*3+1, t] = -1  # y[0,1]
        W[0*3+2, t] = 1   # c[2,0]
        W[1*3+2, t] = 1   # c[2,1]
        W[2*3+2, t] = 1   # c[2,2]
        
        # m7 = (x[0,0] + x[1,1] + x[2,2]) * (y[0,0] + y[1,1] + y[2,2])
        t = 6
        U[0, t] = 1
        U[4, t] = 1
        U[8, t] = 1
        V[0*3+0, t] = 1   # y[0,0]
        V[1*3+1, t] = 1   # y[1,1]
        V[2*3+2, t] = 1   # y[2,2]
        W[0*3+0, t] = 1   # c[0,0]
        W[1*3+0, t] = 1   # c[0,1]
        W[2*3+0, t] = 1   # c[0,2]
        W[0*3+1, t] = 1   # c[1,0]
        W[1*3+1, t] = 1   # c[1,1]
        W[2*3+1, t] = 1   # c[2,1]
        W[0*3+2, t] = 1   # c[2,0]
        W[1*3+2, t] = 1   # c[2,1]
        W[2*3+2, t] = 1   # c[2,2]
        
        return U, V, W
    
    # Verify Strassen-3
    U_s3, V_s3, W_s3 = strassen_3x3()
    T_33 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_33[i*3+j, j*3+k, k*3+i] = 1.0
    
    T_recon_33 = np.zeros((9, 9, 9))
    for t in range(7):
        uv = np.outer(U_s3[:, t], V_s3[:, t])
        T_recon_33 += np.einsum('ab,c->abc', uv, W_s3[:, t])
    
    err_33 = np.max(np.abs(T_recon_33 - T_33))
    
    if err_33 > 1e-10:
        # Strassen-3 construction has errors, fall back to schoolbook
        U, V, W = U_sb, V_sb, W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now build the full 4x5*5x6 decomposition using Strassen-3 on 3x3x3 blocks
    #
    # Partition:
    # n=4: rows 0-2 (3 rows) + row 3 (1 row)
    # m=5: cols 0-2 (3 cols) + cols 3-4 (2 cols)
    # p=6: cols 0-2 (3 cols) + cols 3-5 (3 cols)
    #
    # The 3x3x3 core: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Rank: 7 (Strassen-3)
    #
    # Remaining terms (schoolbook):
    # A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]  : 3*3*3 = 27
    # A[0:3, 3:5] * B[0:3, 0:6] -> C[0:3, 0:6]  : 3*2*6 = 36
    # A[3, 0:3] * B[0:3, 0:6] -> C[3, 0:6]      : 1*3*6 = 18
    # A[0:3, 0:3] * B[3:5, 0:6] -> C[0:3, 0:6]  : 3*2*6 = 36
    # A[3, 3:5] * B[3:5, 0:6] -> C[3, 0:6]      : 1*2*6 = 12
    # A[3, 0:3] * B[3:5, 0:6] -> C[3, 0:6]      : 1*2*6 = 12
    #
    # Wait, I need to be more careful. The full decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let me partition:
    # I = {0,1,2} (3 rows), I' = {3} (1 row)
    # J = {0,1,2} (3 cols), J' = {3,4} (2 cols)
    # K = {0,1,2} (3 cols), K' = {3,4,5} (3 cols)
    #
    # C = A_I,J * B_J,K + A_I,J * B_J,K' + A_I,J' * B_J,K + A_I,J' * B_J,K'
    #   + A_I',J * B_J,K + A_I',J * B_J,K' + A_I',J' * B_J,K + A_I',J' * B_J,K'
    #
    # Block (I,J,K): 3x3*3x3 -> Strassen rank 7
    # Block (I,J,K'): 3x3*3x3 -> Strassen rank 7 (B_J,K' is 3x3)
    # Block (I,J',K): 3x2*3x3 -> schoolbook rank 3*2*3 = 18
    # Block (I,J',K'): 3x2*3x3 -> schoolbook rank 3*2*3 = 18
    # Block (I',J,K): 1x3*3x3 -> schoolbook rank 1*3*3 = 9
    # Block (I',J,K'): 1x3*3x3 -> schoolbook rank 1*3*3 = 9
    # Block (I',J',K): 1x2*3x3 -> schoolbook rank 1*2*3 = 6
    # Block (I',J',K'): 1x2*3x3 -> schoolbook rank 1*2*3 = 6
    #
    # Total: 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6 = 80
    
    r_total = 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6
    U_full = np.zeros((nm, r_total))
    V_full = np.zeros((mp, r_total))
    W_full = np.zeros((pn, r_total))
    
    col = 0
    
    # Block (I,J,K): A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Strassen-3
    for t in range(7):
        for a_idx in range(9):
            if U_s3[a_idx, t] != 0:
                i = a_idx // 3
                j = a_idx % 3
                U_full[i*5 + j, col] += U_s3[a_idx, t]
        for b_idx in range(9):
            if V_s3[b_idx, t] != 0:
                j = b_idx // 3
                k = b_idx % 3
                V_full[j*6 + k, col] += V_s3[b_idx, t]
        for c_idx in range(9):
            if W_s3[c_idx, t] != 0:
                k = c_idx // 3
                i = c_idx % 3
                W_full[k*4 + i, col] += W_s3[c_idx, t]
        col += 1
    
    # Block (I,J,K'): A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    # Strassen-3 (B block is 3x3: rows 0-2, cols 3-5)
    for t in range(7):
        for a_idx in range(9):
            if U_s3[a_idx, t] != 0:
                i = a_idx // 3
                j = a_idx % 3
                U_full[i*5 + j, col] += U_s3[a_idx, t]
        for b_idx in range(9):
            if V_s3[b_idx, t] != 0:
                j = b_idx // 3
                k = b_idx % 3
                V_full[j*6 + (3 + k), col] += V_s3[b_idx, t]
        for c_idx in range(9):
            if W_s3[c_idx, t] != 0:
                k = c_idx // 3
                i = c_idx % 3
                W_full[(3 + k)*4 + i, col] += W_s3[c_idx, t]
        col += 1
    
    # Block (I,J',K): A[0:3, 3:5] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Schoolbook: 3*2*3 = 18
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                U_full[i*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + i, col] = 1
                col += 1
    
    # Block (I,J',K'): A[0:3, 3:5] * B[0:3, 3:6] -> C[0:3, 3:6]
    # Schoolbook: 3*2*3 = 18
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 6):
                U_full[i*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + i, col] = 1
                col += 1
    
    # Block (I',J,K): A[3, 0:3] * B[0:3, 0:3] -> C[3, 0:3]
    # Schoolbook: 1*3*3 = 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_full[3*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + 3, col] = 1
                col += 1
    
    # Block (I',J,K'): A[3, 0:3] * B[0:3, 3:6] -> C[3, 3:6]
    # Schoolbook: 1*3*3 = 9
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                U_full[3*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + 3, col] = 1
                col += 1
    
    # Block (I',J',K): A[3, 3:5] * B[0:3, 0:3] -> C[3, 0:3]
    # Schoolbook: 1*2*3 = 6
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                U_full[3*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + 3, col] = 1
                col += 1
    
    # Block (I',J',K'): A[3, 3:5] * B[0:3, 3:6] -> C[3, 3:6]
    # Schoolbook: 1*2*3 = 6
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 6):
                U_full[3*5 + j, col] = 1
                V_full[j*6 + k, col] = 1
                W_full[k*4 + 3, col] = 1
                col += 1
    
    # Verify
    err_full = verify_decomposition(U_full, V_full, W_full)
    
    if err_full < 1e-10:
        best_rank = r_total
        best_decomp = (U_full.copy(), V_full.copy(), W_full.copy())
    else:
        # Fall back to schoolbook
        U, V, W = U_sb, V_sb, W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
