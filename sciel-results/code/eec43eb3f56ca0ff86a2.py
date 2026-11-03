import numpy as np


def construct():
    n, m, p = 4, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 16, 32, 32

    # 2x2 Strassen decomposition (rank 7)
    # U2[i, s]: coefficient for A[i,j] in Strassen product s
    # V2[j, s]: coefficient for B[j,k] in Strassen product s  
    # W2[i, s]: coefficient for C[i,k] in Strassen product s
    U2 = np.array([
        [ 1,  0,  0,  0,  1,  0,  0],  # row 0
        [ 0,  0,  0,  0,  1,  0,  1],  # row 1
        [ 0,  1,  0,  0,  0,  1,  0],  # row 2
        [ 1,  1,  0,  1,  0,  0, -1],  # row 3
    ], dtype=float)
    V2 = np.array([
        [ 1,  0,  1,  0,  0,  1,  0],  # row 0
        [ 0,  0, -1,  0,  0,  1,  0],  # row 1
        [ 0,  0,  0,  1,  0,  0,  1],  # row 2
        [ 1,  0,  0, -1,  1,  0,  1],  # row 3
    ], dtype=float)
    W2 = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],  # row 0
        [ 0,  0,  1,  0,  1,  0,  0],  # row 1
        [ 0,  1,  0,  1,  0,  0,  0],  # row 2
        [ 1, -1,  0,  0,  0,  1,  0],  # row 3
    ], dtype=float)

    # Verify Strassen: sum_s U2[i,s]*V2[j,s]*W2[k,s] should give C[i,k]=sum_j A[i,j]*B[j,k]
    # For each (i,k): sum_j A[i,j]*B[j,k] = sum_s (sum_j U2[i,s]*B[j,k]) * (sum_j' V2[j',s]*A[i,j']) * W2[k,s]
    # Actually: C[i,k] = sum_s U2[i,s] * V2[:,s] dot B... no
    # The Strassen formula: C[i,k] = sum_s (sum_j U2[j,s] * A[j,?]) ... 
    # Let me verify directly:
    # M[i,k] = sum_j A[i,j]*B[j,k]
    # Strassen says: M = W2^T @ (elementwise product of U2^T@A and B@V2)
    # Actually the standard form: C[i,k] = sum_s (U2[i,s]) * (sum_j A[i,j]*V2[j,s]) * (W2[k,s]) * ... no
    # 
    # Correct form: Define for each s:
    #   X_s = sum_i U2[i,s] * A[i, :]  (row vector)
    #   Y_s = sum_j V2[j,s] * B[:, j]  (column vector)  
    #   Then C = sum_s W2[:,s] * (X_s @ Y_s) ... that's not right either
    #
    # Standard Strassen: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11+A01)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M2+M5
    # C10 = M3+M6
    # C11 = M1-M2+M3+M6
    #
    # In terms of my matrices, the decomposition should satisfy:
    # sum_s U2[i,s]*V2[j,s]*W2[k,s] = delta for the identity case
    # For A=I, B=I: C=I, so sum_s U2[i,s]*V2[i,s]*W2[k,s] = delta[i,k]

    U_list = []
    V_list = []
    W_list = []

    def add_4x4x4_block(A_row0, A_col0, B_row0, B_col0, C_row0, C_col0):
        """Add Strassen terms for a 4x4x4 multiplication block."""
        for s in range(7):
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            
            for ii in range(4):
                for jj in range(4):
                    u[A_row0 + ii, ]  # this won't work, let me redo
            # Better: build u, v, w directly
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            
            for ii in range(4):
                for jj in range(4):
                    # A element at global position (A_row0+ii, A_col0+jj)
                    # In u vector: index = (A_row0+ii)*m + (A_col0+jj)
                    u[(A_row0 + ii) * m + (A_col0 + jj)] += U2[ii, s]
            
            for jj in range(4):
                for kk in range(4):
                    # B element at global position (B_row0+jj, B_col0+kk)
                    # In v vector: index = (B_row0+jj)*p + (B_col0+kk)
                    v[(B_row0 + jj) * p + (B_col0 + kk)] += V2[jj, s]
            
            for ii in range(4):
                for kk in range(4):
                    # C element at global position (C_row0+ii, C_col0+kk)
                    # In w vector: index = (C_row0+kk)*n + (C_row0+ii)  -- wait
                    # Output factor follows W[k*n+i, r], so W index = k*n + i
                    # C[i,k] corresponds to W[k*n + i]
                    w[(C_row0 + kk) * n + (C_row0 + ii)] += W2[ii, s]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)

    # Two 4x4x4 blocks:
    # Block 0: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    add_4x4x4_block(0, 0, 0, 0, 0, 0)
    # Block 1: A[0:4,0:4] * B[0:4,4:8] -> C[0:4,4:8]
    add_4x4x4_block(0, 0, 0, 4, 0, 4)

    rank = len(U_list)
    U = np.array(U_list).T  # shape (nm, rank)
    V = np.array(V_list).T  # shape (mp, rank)
    W = np.array(W_list).T  # shape (pn, rank)

    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                # Wait, the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # That's element-wise product then sum, not dot of V*W
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))

    # If verification fails, return schoolbook
    if max_err > 1e-10:
        U_sb = np.zeros((nm, nm * mp // 4 * 4))  # just use full schoolbook
        # Schoolbook: rank = n*m*p = 128
        U_sb = np.zeros((nm, 128))
        V_sb = np.zeros((mp, 128))
        W_sb = np.zeros((pn, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
