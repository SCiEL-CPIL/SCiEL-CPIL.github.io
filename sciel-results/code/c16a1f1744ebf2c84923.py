import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 140
    
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        val = sum(U[i*m+j, r] * V[k*p+l, r] * W[k*n+i, r] for r in range(rank))
                        expected = 1 if j == k else 0
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    def build_strassen_22():
        rank = 22
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # c00 = m1 - m4 + m6 + m7
        # c01 = m3 + m4 + m5
        # c10 = m2 + m4 - m6 - m7
        # c11 = m1 - m2 + m3 + m5
        
        # m1 = (a00+a11)(b00+b11)
        U[0, 0] += 1; U[4, 0] += 1  # a00+a11
        V[0, 0] += 1; V[5, 0] += 1  # b00+b11
        W[0, 0] += 1; W[3, 0] += 1  # c00, c11
        
        # m2 = (a10+a11)b00
        U[3, 1] += 1; U[4, 1] += 1  # a10+a11
        V[0, 1] += 1  # b00
        W[1, 1] += 1; W[3, 1] -= 1  # c10, -c11
        
        # m3 = a00(b01-b10)
        U[0, 2] += 1  # a00
        V[1, 2] += 1; V[5, 2] -= 1  # b01-b10
        W[2, 2] += 1; W[3, 2] += 1  # c01, c11
        
        # m4 = a11(b10+b11)
        U[4, 3] += 1  # a11
        V[4, 3] += 1; V[5, 3] += 1  # b10+b11
        W[0, 3] -= 1; W[2, 3] += 1; W[1, 3] += 1  # -c00, c01, c10
        
        # m5 = (a00+a01)b10
        U[0, 4] += 1; U[1, 4] += 1  # a00+a01
        V[4, 4] += 1  # b10
        W[2, 4] += 1; W[3, 4] += 1  # c01, c11
        
        # m6 = (a10-a00)(b00+b01)
        U[3, 5] += 1; U[0, 5] -= 1  # a10-a00
        V[0, 5] += 1; V[1, 5] += 1  # b00+b01
        W[0, 5] += 1; W[1, 5] -= 1  # c00, -c10
        
        # m7 = (a01-a11)(b10+b11)
        U[1, 6] += 1; U[4, 6] -= 1  # a01-a11
        V[4, 6] += 1; V[5, 6] += 1  # b10+b11
        W[0, 6] += 1; W[1, 6] -= 1  # c00, -c10
        
        # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        # c02 = m1' - m4' + m6' + m7'
        # c03 = m3' + m4' + m5'
        # c12 = m2' + m4' - m6' - m7'
        # c13 = m1' - m2' + m3' + m5'
        
        # m1' = (a00+a11)(b02+b13)
        U[0, 7] += 1; U[4, 7] += 1
        V[2, 7] += 1; V[7, 7] += 1  # b02+b13
        W[4, 7] += 1; W[7, 7] += 1  # c02, c13
        
        # m2' = (a10+a11)b02
        U[3, 8] += 1; U[4, 8] += 1
        V[2, 8] += 1  # b02
        W[5, 8] += 1; W[7, 8] -= 1  # c12, -c13
        
        # m3' = a00(b03-b13)
        U[0, 9] += 1
        V[3, 9] += 1; V[7, 9] -= 1  # b03-b13
        W[6, 9] += 1; W[7, 9] += 1  # c03, c13
        
        # m4' = a11(b12+b02)
        U[4, 10] += 1
        V[6, 10] += 1; V[2, 10] += 1  # b12+b02
        W[4, 10] -= 1; W[6, 10] += 1; W[5, 10] += 1  # -c02, c03, c12
        
        # m5' = (a00+a01)b12
        U[0, 11] += 1; U[1, 11] += 1
        V[6, 11] += 1  # b12
        W[6, 11] += 1; W[7, 11] += 1  # c03, c13
        
        # m6' = (a10-a00)(b02+b03)
        U[3, 12] += 1; U[0, 12] -= 1
        V[2, 12] += 1; V[3, 12] += 1  # b02+b03
        W[4, 12] += 1; W[5, 12] -= 1  # c02, -c12
        
        # m7' = (a01-a11)(b12+b13)
        U[1, 13] += 1; U[4, 13] -= 1
        V[6, 13] += 1; V[7, 13] += 1  # b12+b13
        W[4, 13] += 1; W[5, 13] -= 1  # c02, -c12
        
        # Cross: A[0:2,2] * B[2,:] -> C[0:2,:]
        for i in range(2):
            for l in range(4):
                col = 14 + i * 4 + l
                U[i*3 + 2, col] += 1
                V[8 + l, col] += 1
                W[l*2 + i, col] += 1
        
        return U, V, W
    
    U, V, W = build_strassen_22()
    rank = 22
    
    if not verify(U, V, W, rank):
        # Fallback to naive
        rank = 24
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    r = i * m * p + j * p + l
                    U[i*m + j, r] = 1
                    V[j*p + l, r] = 1
                    W[j*n + i, r] = 1
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to reduce rank by checking for redundant columns
    for remove_col in range(rank - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        new_rank = rank - 1
        if new_rank < 1:
            break
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, remove_col] = 0
        V_test[:, remove_col] = 0
        W_test[:, remove_col] = 0
        if verify(U_test, V_test, W_test, new_rank):
            U = np.delete(U_test, remove_col, axis=1)
            V = np.delete(V_test, remove_col, axis=1)
            W = np.delete(W_test, remove_col, axis=1)
            rank = new_rank
            best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 1) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
