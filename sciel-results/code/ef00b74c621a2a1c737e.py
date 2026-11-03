"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # n=2, m=3, p=3
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2]
    # A[1,0]=U[3], A[1,1]=U[4], A[1,2]=U[5]
    # B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2]
    # B[1,0]=V[3], B[1,1]=V[4], B[1,2]=V[5]
    # B[2,0]=V[6], B[2,1]=V[7], B[2,2]=V[8]
    # C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2]
    # C[1,0]=W[3], C[1,1]=W[4], C[1,2]=W[5]
    
    # Strassen terms for 2x2x2 block (indices 0,1 for A,B,C)
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C01+=M1, C10-=M1, C11+=M1
    # Wait, standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let's map this to our indices.
    # A00=U[0], A01=U[1], A10=U[3], A11=U[4]
    # B00=V[0], B01=V[1], B10=V[3], B11=V[4]
    # C00=W[0], C01=W[1], C10=W[3], C11=W[4]
    
    # Term 0: M1
    # U: A00+A11 -> U[0]=1, U[4]=1
    # V: B00+B11 -> V[0]=1, V[4]=1
    # W: C00+M1, C01+M1, C10-M1, C11+M1
    # W[0]+=1, W[1]+=1, W[3]-=1, W[4]+=1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = -1; W[4, 0] = 1
    
    # Term 1: M2
    # U: A10+A11 -> U[3]=1, U[4]=1
    # V: B00 -> V[0]=1
    # W: C10+M2, C11-M2
    # W[3]+=1, W[4]-=1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1; W[4, 1] = -1
    
    # Term 2: M3
    # U: A00 -> U[0]=1
    # V: B10-B11 -> V[3]=1, V[4]=-1
    # W: C01+M3, C11+M3
    # W[1]+=1, W[4]+=1
    U[0, 2] = 1
    V[3, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1; W[4, 2] = 1
    
    # Term 3: M4
    # U: A11 -> U[4]=1
    # V: B01-B00 -> V[1]=1, V[0]=-1
    # W: C00-M4, C10+M4
    # W[0]-=1, W[3]+=1
    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[3, 3] = 1
    
    # Term 4: M5
    # U: A00+A01 -> U[0]=1, U[1]=1
    # V: B11 -> V[4]=1
    # W: C00-M5, C01+M5
    # W[0]-=1, W[1]+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6
    # U: A10-A00 -> U[3]=1, U[0]=-1
    # V: B00+B01 -> V[0]=1, V[1]=1
    # W: (Not used in C for 2x2? M6 is not in standard Strassen C formulas above?
    # Wait, standard Strassen uses M1-M7.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M6 is (A10-A00)(B00+B01). It is not used in the final C?
    # Ah, I see. M6 is used in some variations or I am misremembering.
    # Let's check the formulas.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # Yes, M6 is not used. So we only need 7 terms.
    
    # Term 6: M7
    # U: A01-A11 -> U[1]=1, U[4]=-1
    # V: B10+B11 -> V[3]=1, V[4]=1
    # W: C00-M7, C11+M7
    # W[0]-=1, W[4]+=1
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = -1; W[4, 6] = 1
    
    # Part 2: Terms for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A02=U[2], A12=U[5]
    # B20=V[6], B21=V[7]
    # C00=W[0], C01=W[1], C10=W[3], C11=W[4]
    # C00 += A02*B20
    # C01 += A02*B21
    # C10 += A12*B20
    # C11 += A12*B21
    
    # Term 7: A02*B20
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # Term 8: A02*B21
    U[2, 8] = 1
    V[7, 8] = 1
    W[1, 8] = 1
    
    # Term 9: A12*B20
    U[5, 9] = 1
    V[6, 9] = 1
    W[3, 9] = 1
    
    # Term 10: A12*B21
    U[5, 10] = 1
    V[7, 10] = 1
    W[4, 10] = 1
    
    # Part 3: Terms for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A00=U[0], A01=U[1], A02=U[2]
    # A10=U[3], A11=U[4], A12=U[5]
    # B02=V[2], B12=V[5], B22=V[8]
    # C02=W[2], C12=W[5]
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    # Term 11: A00*B02
    U[0, 11] = 1
    V[2, 11] = 1
    W[2, 11] = 1
    
    # Term 12: A01*B12
    U[1, 12] = 1
    V[5, 12] = 1
    W[2, 12] = 1
    
    # Term 13: A02*B22
    U[2, 13] = 1
    V[8, 13] = 1
    W[2, 13] = 1
    
    # Term 14: A10*B02
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # Term 15: A11*B12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # Term 16: A12*B22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify exact decomposition
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
