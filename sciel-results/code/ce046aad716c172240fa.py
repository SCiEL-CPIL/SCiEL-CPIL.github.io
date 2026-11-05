import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Strassen 2x2x2 factors
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U_22 = np.zeros((4, 7), dtype=int)
    V_22 = np.zeros((4, 7), dtype=int)
    W_22 = np.zeros((4, 7), dtype=int)
    
    # Strassen terms
    # 0: (A11+A22)(B11+B22)
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    W_22[0, 0] = 1; W_22[3, 0] = 1
    
    # 1: (A21+A22)B11
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    W_22[1, 1] = 1; W_22[3, 1] = 1 # Wait, W index is k*2+i.
    # Term 1 contributes to C21 and C22?
    # C = A B.
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2
    # M3 = A11(B21-B22) -> C12 -= M3? No, C12 += M3?
    # Let's check Strassen formula carefully.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # Let's re-verify the W matrix entries for 2x2.
    # W[k, i, r]
    # r=0 (M1): C11, C22. W[0,0]=1, W[1,1]=1. (k=0,i=0 and k=1,i=1)
    # r=1 (M2): C21. W[0,1]=1. (k=0,i=1)
    # r=2 (M3): C12. W[1,0]=1. (k=1,i=0)
    # r=3 (M4): C11, C21. W[0,0]=-1, W[0,1]=1.
    # r=4 (M5): C11, C12. W[0,0]=1, W[1,0]=1.
    # r=5 (M6): C22. W[1,1]=1.
    # r=6 (M7): C11, C22. W[0,0]=-1, W[1,1]=1.
    
    # Let's re-fill W_22 based on this.
    W_22 = np.zeros((4, 7), dtype=int)
    # r=0
    W_22[0, 0] = 1 # k=0, i=0
    W_22[3, 0] = 1 # k=1, i=1
    # r=1
    W_22[2, 1] = 1 # k=0, i=1
    # r=2
    W_22[1, 2] = 1 # k=1, i=0
    # r=3
    W_22[0, 3] = -1 # k=0, i=0
    W_22[2, 3] = 1  # k=0, i=1
    # r=4
    W_22[0, 4] = 1  # k=0, i=0
    W_22[1, 4] = 1  # k=1, i=0
    # r=5
    W_22[3, 5] = 1  # k=1, i=1
    # r=6
    W_22[0, 6] = -1 # k=0, i=0
    W_22[3, 6] = 1  # k=1, i=1

    # U and V for Strassen
    U_22 = np.zeros((4, 7), dtype=int)
    V_22 = np.zeros((4, 7), dtype=int)
    # r=0: (A11+A22)(B11+B22)
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    # r=1: (A21+A22)B11
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    # r=2: A11(B21-B22)
    U_22[0, 2] = 1
    V_22[2, 2] = 1; V_22[3, 2] = -1
    # r=3: A22(B12-B11)
    U_22[3, 3] = 1
    V_22[1, 3] = 1; V_22[0, 3] = -1
    # r=4: (A11+A12)B22
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    # r=5: (A21-A11)(B11+B12)
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    # r=6: (A12-A22)(B21+B22)
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[2, 6] = 1; V_22[3, 6] = 1

    # Kronecker product for 4x4x4
    U_44 = np.kron(U_22, U_22)
    V_44 = np.kron(V_22, V_22)
    W_44 = np.kron(W_22, W_22)
    
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    U_41 = np.zeros((16, 16), dtype=int)
    V_41 = np.zeros((16, 16), dtype=int)
    W_41 = np.zeros((4, 16), dtype=int) # p=1, n=4 -> 1*4=4 rows
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_41[i * 4 + j, r] = 1
            V_41[j * 1 + 0, r] = 1 # j*1+k, k=0
            W_41[0 * 4 + i, r] = 1 # k*n+i, k=0

    # Combine
    # U: (16, 49+16)
    # V: (16, 49+16)
    # W: (5*4, 49+16) = (20, 65)
    
    U_final = np.hstack([U_44, U_41])
    V_final = np.hstack([V_44, V_41])
    
    # W_44 has shape (16, 49). Indices k in 0..3, i in 0..3.
    # W_41 has shape (4, 16). Indices k' in 0..0, i in 0..3.
    # We need to place W_41 into rows 16..19 of W_final.
    W_final = np.zeros((20, 65), dtype=int)
    W_final[0:16, 0:49] = W_44
    W_final[16:20, 49:65] = W_41
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
