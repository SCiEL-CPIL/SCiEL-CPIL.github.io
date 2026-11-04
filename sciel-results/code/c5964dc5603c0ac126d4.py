import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 18
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: A_2x2 * B_2x4
    # This corresponds to indices i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # We use two copies of Strassen 2x2x2
    # Copy 1: k in {0,1}
    # Copy 2: k in {2,3}
    
    # Strassen 2x2x2 rank 7 factors
    # U_str (6x7), V_str (4x7), W_str (4x7)
    # Indices for 2x2x2:
    # U: i*2+j (i=0,1; j=0,1) -> 0,1,2,3
    # V: j*2+k (j=0,1; k=0,1) -> 0,1,2,3
    # W: k*2+i (k=0,1; i=0,1) -> 0,1,2,3
    
    # Standard Strassen decomposition for 2x2x2
    # Let's define the 7 terms
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # U rows: i*2+j
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    U_str = np.zeros((4, 7))
    V_str = np.zeros((4, 7))
    W_str = np.zeros((4, 7))
    
    # M1
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[1, 0] = 1; W_str[2, 0] = 1; W_str[3, 0] = 1 # Wait, W is k*2+i
    # C11 = M1... C12=M3+M5... C21=M2+M4... C22=M1-M2+M3+M6
    # W indices:
    # 0: k=0, i=0 (C11)
    # 1: k=0, i=1 (C21)
    # 2: k=1, i=0 (C12)
    # 3: k=1, i=1 (C22)
    
    # M1 contributes to C11, C22
    W_str[0, 0] = 1
    W_str[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21, C22
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[1, 1] = 1 # C21
    W_str[3, 1] = -1 # C22 (M1-M2)
    
    # M3: A11(B12-B22) -> C12, C22
    U_str[0, 2] = 1
    V_str[1, 2] = 1; V_str[3, 2] = -1
    W_str[2, 2] = 1 # C12
    W_str[3, 2] = 1 # C22
    
    # M4: A22(B21-B11) -> C11, C21
    U_str[3, 3] = 1
    V_str[2, 3] = 1; V_str[0, 3] = -1
    W_str[0, 3] = 1 # C11
    W_str[1, 3] = 1 # C21
    
    # M5: (A11+A12)B22 -> C11, C12
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[0, 4] = -1 # C11 (M1-M5)
    W_str[2, 4] = 1 # C12
    
    # M6: (A21-A11)(B11+B12) -> C22
    U_str[2, 5] = 1; U_str[0, 5] = -1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1 # C22
    
    # M7: (A12-A22)(B21+B22) -> C11
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[2, 6] = 1; V_str[3, 6] = 1
    W_str[0, 6] = 1 # C11
    
    # Now map to 2x2x4
    # Copy 1: k=0,1. V indices j*4+k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 4. j=1, k=1 -> 5.
    # W indices k*2+i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    
    U[:, 0:7] = U_str
    # V mapping
    V[0, 0:7] = V_str[0, :]
    V[1, 0:7] = V_str[1, :]
    V[4, 0:7] = V_str[2, :]
    V[5, 0:7] = V_str[3, :]
    # W mapping
    W[0:4, 0:7] = W_str
    
    # Copy 2: k=2,3.
    # V indices:
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 6. j=1, k=3 -> 7.
    # W indices:
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    # k=3, i=0 -> 6. k=3, i=1 -> 7.
    
    U[:, 7:14] = U_str
    V[2, 7:14] = V_str[0, :]
    V[3, 7:14] = V_str[1, :]
    V[6, 7:14] = V_str[2, :]
    V[7, 7:14] = V_str[3, :]
    W[4:8, 7:14] = W_str
    
    # Part 2: A_2x1 * B_1x4
    # j=2.
    # U indices: i*3+2. i=0 -> 2. i=1 -> 5.
    # V indices: 2*4+k = 8+k. k=0,1,2,3 -> 8,9,10,11.
    # W indices: k*2+i.
    
    # Rank 4 decomposition for this part.
    # T = (E_{0,2} + E_{1,2}) \otimes I_4?
    # No, T = E_{0,2} \otimes I_4 + E_{1,2} \otimes I_4?
    # C_{0,k} = A_{0,2} B_{2,k}
    # C_{1,k} = A_{1,2} B_{2,k}
    
    # We can use 4 terms, one for each k.
    # Term k:
    # U: E_{0,2} and E_{1,2}?
    # If we use U = [1, 1]^T, V = F_{2,k}, W = G_{k,0} + G_{k,1}?
    # No, W needs to select C_{0,k} and C_{1,k}.
    # W_{k,0} = 1, W_{k,1} = 1?
    # Then C_{0,k} gets A_{0,2} B_{2,k} and C_{1,k} gets A_{1,2} B_{2,k}.
    
    # So for each k in 0..3:
    # U[2, col] = 1
    # U[5, col] = 1
    # V[8+k, col] = 1
    # W[k*2, col] = 1
    # W[k*2+1, col] = 1
    
    for k in range(4):
        col = 14 + k
        U[2, col] = 1
        U[5, col] = 1
        V[8+k, col] = 1
        W[k*2, col] = 1
        W[k*2+1, col] = 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
