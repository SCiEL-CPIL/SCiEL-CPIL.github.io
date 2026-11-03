"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Kronecker product: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Factor 1: 2x2x2 matrix multiplication, rank 7 (Strassen)
    # Factor 2: 2x2x3 matrix multiplication, rank 11 (Strassen for first 2 cols + trivial for 3rd col)
    
    # Strassen decomposition for 2x2x2:
    # U1 rows: 4 (i*m+j), V1 rows: 4 (j*p+k), W1 rows: 4 (k*n+i)
    # rank 7
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a11+a22)(b11+b22)
    U1[0,0] = 1; U1[3,0] = 1
    V1[0,0] = 1; V1[3,0] = 1
    W1[0,0] = 1; W1[3,0] = 1
    
    # M2 = (a21+a22)b11
    U1[2,1] = 1; U1[3,1] = 1
    V1[0,1] = 1
    W1[1,1] = 1; W1[3,1] = -1
    
    # M3 = a11(b12-b22)
    U1[0,2] = 1
    V1[1,2] = 1; V1[3,2] = -1
    W1[2,2] = 1; W1[3,2] = 1
    
    # M4 = a22(b21-b11)
    U1[3,3] = 1
    V1[2,3] = 1; V1[0,3] = -1
    W1[0,3] = 1; W1[1,3] = 1
    
    # M5 = (a11+a12)b22
    U1[0,4] = 1; U1[1,4] = 1
    V1[3,4] = 1
    W1[0,4] = -1; W1[2,4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U1[2,5] = 1; U1[0,5] = -1
    V1[0,5] = 1; V1[1,5] = 1
    W1[3,5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U1[1,6] = 1; U1[3,6] = -1
    V1[2,6] = 1; V1[3,6] = 1
    W1[0,6] = 1
    
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # U2 rows: 4 (i*m+j), V2 rows: 6 (j*p+k), W2 rows: 6 (k*n+i)
    # rank 11
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Strassen part (columns 0-1), terms 0-6
    # A is 2x2, B is 2x2 (first 2 cols), C is 2x2 (first 2 cols)
    # U2[i*2+j, c] = U1[i*2+j, c]
    for c in range(7):
        U2[:, c] = U1[:, c]
        
        # V: j*p+k where p=2 for the 2x2 part, so V1[j*2+k, c]
        # V2[j*3+k, c] for k=0,1 maps to V1[j*2+k, c]
        V2[0, c] = V1[0, c]  # j=0,k=0
        V2[1, c] = V1[1, c]  # j=0,k=1
        V2[2, c] = 0         # j=1,k=0 (but wait, V1 has 4 rows for 2x2, V2 has 6 rows for 2x3)
        # Actually V1 rows are j*2+k, so V1[0]=j=0,k=0, V1[1]=j=0,k=1, V1[2]=j=1,k=0, V1[3]=j=1,k=1
        # V2 rows are j*3+k, so V2[0]=j=0,k=0, V2[1]=j=0,k=1, V2[2]=j=0,k=2, V2[3]=j=1,k=0, V2[4]=j=1,k=1, V2[5]=j=1,k=2
        # For the Strassen part (cols 0,1), we only use k=0,1
        V2[0, c] = V1[0, c]  # j=0,k=0
        V2[1, c] = V1[1, c]  # j=0,k=1
        V2[3, c] = V1[2, c]  # j=1,k=0
        V2[4, c] = V1[3, c]  # j=1,k=1
        V2[2, c] = 0         # j=0,k=2
        V2[5, c] = 0         # j=1,k=2
        
        # W: k*n+i where n=2 for the 2x2 part
        # W1 rows are k*2+i, W2 rows are k*2+i for k=0,1
        W2[0, c] = W1[0, c]  # k=0,i=0
        W2[1, c] = W1[1, c]  # k=0,i=1
        W2[2, c] = W1[2, c]  # k=1,i=0
        W2[3, c] = W1[3, c]  # k=1,i=1
        W2[4, c] = 0         # k=2,i=0
        W2[5, c] = 0         # k=2,i=1
    
    # Trivial part (column 2), terms 7-10
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U2[0, 7] = 1  # i=0,j=0
    V2[2, 7] = 1  # j=0,k=2
    W2[4, 7] = 1  # k=2,i=0
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U2[1, 8] = 1  # i=0,j=1
    V2[5, 8] = 1  # j=1,k=2
    W2[4, 8] = 1  # k=2,i=0
    
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U2[2, 9] = 1  # i=1,j=0
    V2[2, 9] = 1  # j=0,k=2
    W2[5, 9] = 1  # k=2,i=1
    
    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U2[3, 10] = 1  # i=1,j=1
    V2[5, 10] = 1  # j=1,k=2
    W2[5, 10] = 1  # k=2,i=1
    
    # Now Kronecker product:
    # Total rank = 7 * 11 = 77
    # U full: (4*4, 77) = (16, 77)
    # V full: (4*6, 77) = (24, 77)
    # W full: (6*4, 77) = (24, 77)
    
    rank1, rank2 = 7, 11
    rank = rank1 * rank2
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    # U_full[i1*8+i2*4+j1*2+j2, c1*11+c2] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    row = i1*8 + i2*4 + j1*2 + j2
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            c = c1 * rank2 + c2
                            U[row, c] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
    
    # V_full[j1*12+j2*6+k1*3+k2, c1*11+c2] = V1[j1*2+k1, c1] * V2[j2*3+k2, c2]
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(3):
                    row = j1*12 + j2*6 + k1*3 + k2
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            c = c1 * rank2 + c2
                            V[row, c] = V1[j1*2+k1, c1] * V2[j2*3+k2, c2]
    
    # W_full[k1*12+k2*4+i1*2+i2, c1*11+c2] = W1[k1*2+i1, c1] * W2[k2*2+i2, c2]
    for k1 in range(2):
        for k2 in range(3):
            for i1 in range(2):
                for i2 in range(2):
                    row = k1*12 + k2*4 + i1*2 + i2
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            c = c1 * rank2 + c2
                            W[row, c] = W1[k1*2+i1, c1] * W2[k2*2+i2, c2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

