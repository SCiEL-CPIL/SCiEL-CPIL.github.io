"""Matrix multiplication 6x6x6 using Kronecker product of schoolbook 3x3 (rank 27) and Strassen 2x2 (rank 7)."""
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen 2x2 decomposition (rank 7) - VERIFIED
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22)
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    # M2 = (A21+A22)B11
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1
    
    # M3 = A11(B21-B22)
    U2[0,2] = 1
    V2[2,2] = 1; V2[3,2] = -1
    W2[1,2] = 1
    
    # M4 = A22(B12-B11)
    U2[3,3] = 1
    V2[1,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[2,3] = 1
    
    # M5 = (A11+A12)B22
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[1,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1
    
    # Verify Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[2*i+j, :] * V2[2*j+k, :] * W2[2*k+i, :])
                assert val == 1, f"Strassen failed at i={i},j={j},k={k}: got {val}"
    
    # Schoolbook 3x3 decomposition (rank 27)
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
    
    # Verify 3x3 schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = np.sum(U3[i*3+j, :] * V3[j*3+k, :] * W3[k*3+i, :])
                assert val == 1, f"3x3 failed at i={i},j={j},k={k}: got {val}"
    
    # Combine via Kronecker product: 6 = 3 * 2
    # A[i,j] with i=i3*2+i2, j=j3*2+j2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This decomposes as 3x3 outer structure with 2x2 inner structure
    
    rank = 27 * 7
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for c3 in range(27):
        for c2 in range(7):
            c = c3 * 7 + c2
            for i3 in range(3):
                for j3 in range(3):
                    for k3 in range(3):
                        u3_val = U3[i3*3+j3, c3]
                        v3_val = V3[j3*3+k3, c3]
                        w3_val = W3[k3*3+i3, c3]
                        if u3_val == 0 or v3_val == 0 or w3_val == 0:
                            continue
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u2_val = U2[i2*2+j2, c2]
                                    v2_val = V2[j2*2+k2, c2]
                                    w2_val = W2[k2*2+i2, c2]
                                    if u2_val == 0 or v2_val == 0 or w2_val == 0:
                                        continue
                                    
                                    i = i3*2 + i2
                                    j = j3*2 + j2
                                    k = k3*2 + k2
                                    
                                    U[i*6+j, c] = u3_val * u2_val
                                    V[j*6+k, c] = v3_val * v2_val
                                    W[k*6+i, c] = w3_val * w2_val
    
    # Verify the full 6x6x6 decomposition
    for i in range(6):
        for j in range(6):
            for k in range(6):
                val = np.sum(U[i*6+j, :] * V[j*6+k, :] * W[k*6+i, :])
                assert val == 1, f"6x6 failed at i={i},j={j},k={k}: got {val}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
