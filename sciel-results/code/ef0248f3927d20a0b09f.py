import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 6
    
    # Try to use rank-23 decomposition for 3x3x3, duplicated for 3x3x6
    # This gives rank 46.
    
    # Define the 23 terms for 3x3x3
    # U, V, W are 9x23 matrices
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 3x3x3: n=3, m=3, p=3
    
    U_33 = np.zeros((9, 23))
    V_33 = np.zeros((9, 23))
    W_33 = np.zeros((9, 23))
    
    # Term 1: a00 b00
    U_33[0, 0] = 1; V_33[0, 0] = 1; W_33[0, 0] = 1
    # Term 2: a00 b11
    U_33[0, 1] = 1; V_33[4, 1] = 1; W_33[4, 1] = 1 # b11 is j=1, k=1 -> index 4. w11 is k=1, i=1 -> index 4
    # Term 3: a00 b22
    U_33[0, 2] = 1; V_33[8, 2] = 1; W_33[8, 2] = 1
    # Term 4: a11 b00
    U_33[4, 3] = 1; V_33[0, 3] = 1; W_33[3, 3] = 1 # w01 is k=0, i=1 -> index 3
    # Term 5: a11 b11
    U_33[4, 4] = 1; V_33[4, 4] = 1; W_33[7, 4] = 1 # w11 is k=1, i=1 -> index 7
    # Term 6: a11 b22
    U_33[4, 5] = 1; V_33[8, 5] = 1; W_33[11, 5] = 1 # w21 is k=2, i=1 -> index 11
    # Term 7: a22 b00
    U_33[8, 6] = 1; V_33[0, 6] = 1; W_33[6, 6] = 1 # w02 is k=0, i=2 -> index 6
    # Term 8: a22 b11
    U_33[8, 7] = 1; V_33[4, 7] = 1; W_33[10, 7] = 1 # w12 is k=1, i=2 -> index 10
    # Term 9: a22 b22
    U_33[8, 8] = 1; V_33[8, 8] = 1; W_33[14, 8] = 1 # w22 is k=2, i=2 -> index 14
    
    # Term 10: (a01 - a10) b01
    U_33[1, 9] = 1; U_33[3, 9] = -1; V_33[1, 9] = 1; W_33[1, 9] = 1 # w10 is k=1, i=0 -> index 1
    # Term 11: (a01 - a10) b10
    U_33[1, 10] = 1; U_33[3, 10] = -1; V_33[3, 10] = 1; W_33[0, 10] = 1 # w00 is k=0, i=0 -> index 0
    # Term 12: (a02 - a20) b02
    U_33[2, 11] = 1; U_33[6, 11] = -1; V_33[2, 11] = 1; W_33[2, 11] = 1 # w20 is k=2, i=0 -> index 2
    # Term 13: (a02 - a20) b20
    U_33[2, 12] = 1; U_33[6, 12] = -1; V_33[6, 12] = 1; W_33[0, 12] = 1
    # Term 14: (a12 - a21) b12
    U_33[5, 13] = 1; U_33[7, 13] = -1; V_33[7, 13] = 1; W_33[5, 13] = 1 # w21 is k=2, i=1 -> index 5? No, k=2, i=1 -> 2*3+1=7?
    # Wait, W index is k*n + i.
    # k=2, i=1 -> 2*3+1 = 7.
    # My previous W indices were wrong.
    # Let's re-calculate W indices.
    # W[k*n + i, r]
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=0, i=2 -> 2
    # k=1, i=0 -> 3
    # k=1, i=1 -> 4
    # k=1, i=2 -> 5
    # k=2, i=0 -> 6
    # k=2, i=1 -> 7
    # k=2, i=2 -> 8
    
    # Let's restart the W indices.
    W_33 = np.zeros((9, 23))
    # Term 1: a00 b00 -> c00. k=0, i=0 -> W[0]
    W_33[0, 0] = 1
    # Term 2: a00 b11 -> c10. k=1, i=0 -> W[3]
    W_33[3, 1] = 1
    # Term 3: a00 b22 -> c20. k=2, i=0 -> W[6]
    W_33[6, 2] = 1
    # Term 4: a11 b00 -> c01. k=0, i=1 -> W[1]
    W_33[1, 3] = 1
    # Term 5: a11 b11 -> c11. k=1, i=1 -> W[4]
    W_33[4, 4] = 1
    # Term 6: a11 b22 -> c21. k=2, i=1 -> W[7]
    W_33[7, 5] = 1
    # Term 7: a22 b00 -> c02. k=0, i=2 -> W[2]
    W_33[2, 6] = 1
    # Term 8: a22 b11 -> c12. k=1, i=2 -> W[5]
    W_33[5, 7] = 1
    # Term 9: a22 b22 -> c22. k=2, i=2 -> W[8]
    W_33[8, 8] = 1
    
    # Term 10: (a01 - a10) b01 -> c10. k=1, i=0 -> W[3]
    W_33[3, 9] = 1
    # Term 11: (a01 - a10) b10 -> c00. k=0, i=0 -> W[0]
    W_33[0, 10] = 1
    # Term 12: (a02 - a20) b02 -> c20. k=2, i=0 -> W[6]
    W_33[6, 11] = 1
    # Term 13: (a02 - a20) b20 -> c00. k=0, i=0 -> W[0]
    W_33[0, 12] = 1
    # Term 14: (a12 - a21) b12 -> c21. k=2, i=1 -> W[7]
    W_33[7, 13] = 1
    # Term 15: (a12 - a21) b21 -> c11. k=1, i=1 -> W[4]
    W_33[4, 14] = 1
    # Term 16: a01 b10 -> c00. k=0, i=0 -> W[0]
    W_33[0, 15] = 1
    # Term 17: a02 b20 -> c00. k=0, i=0 -> W[0]
    W_33[0, 16] = 1
    # Term 18: a12 b21 -> c11. k=1, i=1 -> W[4]
    W_33[4, 17] = 1
    # Term 19: (a00 + a11) b11 -> c11. k=1, i=1 -> W[4]
    W_33[4, 18] = 1
    # Term 20: (a00 + a22) b22 -> c22. k=2, i=2 -> W[8]
    W_33[8, 19] = 1
    # Term 21: (a11 + a22) b00 -> c01. k=0, i=1 -> W[1]
    W_33[1, 20] = 1
    # Term 22: (a01 + a10) b01 -> c10. k=1, i=0 -> W[3]
    W_33[3, 21] = 1
    # Term 23: (a02 + a20) b02 -> c20. k=2, i=0 -> W[6]
    W_33[6, 22] = 1
    
    # Verify 3x3x3
    def verify_33(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.dot(U[i*3+j], V[j*3+k]) * W[k*3+i]
                    # Wait, the tensor equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                    # So it is np.dot(np.multiply(U[i*3+j], V[j*3+k]), W[k*3+i])
                    if abs(np.dot(np.multiply(U[i*3+j], V[j*3+k]), W[k*3+i]) - 1.0) > 1e-9:
                        return False
        return True

    if not verify_33(U_33, V_33, W_33):
        # Fallback to 54
        rank = 54
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        column = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i * 3 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 3 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Construct 3x3x6 decomposition
    # Rank 46
    rank = 46
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    # Copy U_33 to U
    U[:, :23] = U_33
    U[:, 23:] = U_33
    
    # Copy V_33 to V
    # V[j*6+k, r]
    # For r in 0..22 (first 3x3x3), k is 0..2
    # V[j*6+k, r] = V_33[j*3+k, r]
    for j in range(3):
        for k in range(3):
            V[j*6+k, :23] = V_33[j*3+k, :23]
            V[j*6+3+k, 23:] = V_33[j*3+k, :23]
            
    # Copy W_33 to W
    # W[k*3+i, r]
    # For r in 0..22 (first 3x3x3), k is 0..2
    # W[k*3+i, r] = W_33[k*3+i, r]
    # For r in 23..45 (second 3x3x3), k is 3..5
    # W[k*3+i, r] = W_33[(k-3)*3+i, r-23]
    for k in range(3):
        for i in range(3):
            W[k*3+i, :23] = W_33[k*3+i, :23]
            W[(k+3)*3+i, 23:] = W_33[k*3+i, :23]
            
    # Verify 3x3x6
    def verify_336(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    val = np.dot(np.multiply(U[i*3+j], V[j*6+k]), W[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_336(U, V, W):
        # Fallback to 54
        rank = 54
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        column = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i * 3 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 3 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
