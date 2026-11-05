import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # A is 2x2: A11=(i1,j1), A12=(i1,j2), A21=(i2,j1), A22=(i2,j2)
    # B is 2x2: B11=(j1,k1), B12=(j1,k2), B21=(j2,k1), B22=(j2,k2)
    # C is 2x2: C11=(i1,k1), C12=(i1,k2), C21=(i2,k1), C22=(i2,k2)
    #
    # M1 = (A11+A22)(B11+B22) -> C11+=1, C22+=1
    # M2 = (A21+A22)B11       -> C21+=1, C22-=1
    # M3 = A11(B12-B22)       -> C12+=1, C22+=1
    # M4 = A22(B21-B11)       -> C11+=1, C21+=1
    # M5 = (A11+A12)B22       -> C11-=1, C12+=1
    # M6 = (A21-A11)(B11+B12) -> C22+=1
    # M7 = (A12-A22)(B21+B22) -> C11+=1
    
    def add_strassen(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1
        c = col_offset + 0
        U[a11, c] += 1; U[a22, c] += 1
        V[b11, c] += 1; V[b22, c] += 1
        W[c11, c] += 1; W[c22, c] += 1
        
        # M2
        c = col_offset + 1
        U[a21, c] += 1; U[a22, c] += 1
        V[b11, c] += 1
        W[c21, c] += 1; W[c22, c] += -1
        
        # M3
        c = col_offset + 2
        U[a11, c] += 1
        V[b12, c] += 1; V[b22, c] += -1
        W[c12, c] += 1; W[c22, c] += 1
        
        # M4
        c = col_offset + 3
        U[a22, c] += 1
        V[b21, c] += 1; V[b11, c] += -1
        W[c11, c] += 1; W[c21, c] += 1
        
        # M5
        c = col_offset + 4
        U[a11, c] += 1; U[a12, c] += 1
        V[b22, c] += 1
        W[c11, c] += -1; W[c12, c] += 1
        
        # M6
        c = col_offset + 5
        U[a21, c] += 1; U[a11, c] += -1
        V[b11, c] += 1; V[b12, c] += 1
        W[c22, c] += 1
        
        # M7
        c = col_offset + 6
        U[a12, c] += 1; U[a22, c] += -1
        V[b21, c] += 1; V[b22, c] += 1
        W[c11, c] += 1
        
        return col_offset + 7
    
    def add_schoolbook_221(i1, i2, j1, j2, k, col_offset):
        # 2x2 A block times 2x1 B block = 2x1 C block
        # Schoolbook: 4 terms
        for di in range(2):
            for dj in range(2):
                c = col_offset + di * 2 + dj
                ii = i1 + di
                jj = j1 + dj
                U[ii * m + jj, c] += 1
                V[jj * p + k, c] += 1
                W[k * n + ii, c] += 1
        return col_offset + 4
    
    # Block j=0,1 with k=0,1 (Strassen)
    col = add_strassen(0, 1, 0, 1, 0, 1, col)
    # Block j=0,1 with k=2,3 (Strassen)
    col = add_strassen(0, 1, 0, 1, 2, 3, col)
    # Block j=0,1 with k=4 (Schoolbook 2x2x1)
    col = add_schoolbook_221(0, 1, 0, 1, 4, col)
    # Block j=2,3 with k=0,1 (Strassen)
    col = add_strassen(0, 1, 2, 3, 0, 1, col)
    # Block j=2,3 with k=2,3 (Strassen)
    col = add_strassen(0, 1, 2, 3, 2, 3, col)
    # Block j=2,3 with k=4 (Schoolbook 2x2x1)
    col = add_schoolbook_221(0, 1, 2, 3, 4, col)
    
    # Verify: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=1e-9):
        # Fallback to schoolbook rank 40
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s))
        V_s = np.zeros((m * p, rank_s))
        W_s = np.zeros((p * n, rank_s))
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, c2] = 1
                    V_s[j * p + k, c2] = 1
                    W_s[k * n + i, c2] = 1
                    c2 += 1
        return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
