import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def make_strassen_222():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: (4, 7) for A (2x2), V: (4, 7) for B (2x2), W: (4, 7) for C (2x2)
        A indexed as [i*2+j], B as [j*2+k], C as [k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1 = (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[2,0]=1; W[1,0]=1; W[3,0]=1
        
        # P2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[0,1]=1; W[2,1]=1
        
        # P3 = A11(B21-B11)
        U[0,2]=1
        V[1,2]=1; V[0,2]=-1
        W[0,2]=1; W[1,2]=1
        
        # P4 = A22(B12-B22)
        U[3,3]=1
        V[2,3]=1; V[3,3]=-1
        W[2,3]=1; W[3,3]=1
        
        # P5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[2,4]=1; W[3,4]=1
        
        # P6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[0,5]=1; W[1,5]=1
        
        # P7 = (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[2,6]=1; W[3,6]=1
        
        return U, V, W

    def make_schoolbook_122():
        """Return U, V, W for 1x2x2 schoolbook decomposition.
        U: (2, 4) for A (1x2), V: (4, 4) for B (2x2), W: (2, 4) for C (2x1)
        A indexed as [0*2+j] = j, B as [j*2+k], C as [k*1+0] = k
        """
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        col = 0
        for j in range(2):
            for k in range(2):
                U[j, col] = 1
                V[j*2+k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W

    # Build the full decomposition
    # Split: rows (2+1), middle (2+2), columns (2+2+2+2)
    # 8 blocks of (2x2x2) and 8 blocks of (1x2x2)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Precompute the sub-decompositions
    strassen_U, strassen_V, strassen_W = make_strassen_222()
    school_U, school_V, school_W = make_schoolbook_122()
    
    # Block layout:
    # For row_block in [0, 1] (sizes 2, 1):
    #   For mid_block in [0, 1] (sizes 2, 2):
    #     For col_block in [0, 1, 2, 3] (sizes 2, 2, 2, 2):
    #       If row_block == 0: use Strassen (2x2x2)
    #       If row_block == 1: use schoolbook (1x2x2)
    
    for row_block in range(2):
        row_start = 0 if row_block == 0 else 2
        row_size = 2 if row_block == 0 else 1
        
        for mid_block in range(2):
            mid_start = mid_block * 2
            mid_size = 2
            
            for col_block in range(4):
                col_start = col_block * 2
                col_size = 2
                
                if row_block == 0:
                    # Strassen 2x2x2
                    U_block = np.zeros((n*m, 7))
                    V_block = np.zeros((m*p, 7))
                    W_block = np.zeros((p*n, 7))
                    
                    for li in range(2):
                        for lj in range(2):
                            U_block[(row_start + li) * m + (mid_start + lj), :] = strassen_U[li*2 + lj, :]
                    
                    for lj in range(2):
                        for lk in range(2):
                            V_block[(mid_start + lj) * p + (col_start + lk), :] = strassen_V[lj*2 + lk, :]
                    
                    for lk in range(2):
                        for li in range(2):
                            W_block[(col_start + lk) * n + (row_start + li), :] = strassen_W[lk*2 + li, :]
                    
                    U_parts.append(U_block)
                    V_parts.append(V_block)
                    W_parts.append(W_block)
                    
                else:
                    # Schoolbook 1x2x2
                    U_block = np.zeros((n*m, 4))
                    V_block = np.zeros((m*p, 4))
                    W_block = np.zeros((p*n, 4))
                    
                    for lj in range(2):
                        U_block[(row_start) * m + (mid_start + lj), :] = school_U[lj, :]
                    
                    for lj in range(2):
                        for lk in range(2):
                            V_block[(mid_start + lj) * p + (col_start + lk), :] = school_V[lj*2 + lk, :]
                    
                    for lk in range(2):
                        W_block[(col_start + lk) * n + (row_start), :] = school_W[lk, :]
                    
                    U_parts.append(U_block)
                    V_parts.append(V_block)
                    W_parts.append(W_block)
    
    # Concatenate all blocks
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    total_rank = U.shape[1]
    
    # Verify the decomposition
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    if not np.allclose(T_recon, T_true, atol=1e-8):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        total_rank = rank
    
    # Check time
    if time.time() - start_time > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
