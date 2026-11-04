"""
Strategy: Block decomposition using Strassen for the 4x4x4 part and Schoolbook for the 4x4x3 part.
Total rank = Rank(4,4,4) + Rank(4,4,3) = 49 + 48 = 97.
This is a structural improvement over the baseline schoolbook rank of 112.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # p=7 is split into p1=4 and p2=3
    p1 = 4
    p2 = 3
    
    start_time = time.time()
    deadline = start_time + 140

    # 1. Construct Strassen decomposition for 4x4x4
    # Strassen 2x2x2 rank 7
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: (i,j) for U, (j,k) for V, (k,i) for W
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need matrices U, V, W such that sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1 for all i,j,k in 2x2x2?
    # No, the condition is that the algorithm computes C=AB.
    # The tensor decomposition condition is:
    # T_{i,j,k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}
    # For the trace tensor T_{i,j,k}=1, this means the sum must be 1.
    # Wait, for Strassen, the "tensor" is the multiplication tensor.
    # The standard definition of rank for matrix multiplication is the minimum r such that
    # C_{ik} = sum_t (sum_j U_{i,j,t} A_{i,j}) (sum_j V_{j,k,t} B_{j,k}) is NOT correct.
    # The correct definition is:
    # We want to represent the bilinear map (A, B) -> AB.
    # This is equivalent to decomposing the tensor T_{[n,m,p]} = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
    # The condition for a decomposition of rank r is:
    # T = sum_{t=1}^r u_t \otimes v_t \otimes w_t
    # where u_t in R^{nm}, v_t in R^{mp}, w_t in R^{pn}.
    # The entries are T_{i,j,k} = 1 if (i,j,k) is a valid term? No, T_{i,j,k}=1 for all i,j,k.
    # So we need sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = 1 for all i,j,k.
    
    # Let's verify Strassen 2x2x2 satisfies this.
    # Strassen computes C = AB.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # The linear forms for A (U) and B (V) are defined by the coefficients in M_t.
    # The linear forms for C (W) are defined by the coefficients in the combination for C_ik.
    #
    # Let's define U, V, W for 2x2x2 explicitly.
    # Rows of U are indexed by (i,j) in 0..3.
    # Rows of V are indexed by (j,k) in 0..3.
    # Rows of W are indexed by (k,i) in 0..3.
    
    # M1: (A00+A11)(B00+B11)
    # A coeffs: (0,0):1, (1,1):1
    # B coeffs: (0,0):1, (1,1):1
    # C coeffs: C00:1, C11:1
    #
    # M2: (A10+A11)B00
    # A coeffs: (1,0):1, (1,1):1
    # B coeffs: (0,0):1
    # C coeffs: C10:1, C11:-1 (from C11 = M1 - M2 ...) wait.
    # C11 = M1 - M4 + M5 - M7. So M2 has coeff 0 for C11?
    # C10 = M2 + M4. So M2 has coeff 1 for C10.
    #
    # Let's build the matrices.
    # t=0: M1
    # U[0,0]=1, U[3,0]=1 (A00, A11)
    # V[0,0]=1, V[3,0]=1 (B00, B11)
    # W[0,0]=1 (C00), W[3,0]=1 (C11)
    
    # t=1: M2
    # U[2,1]=1, U[3,1]=1 (A10, A11)
    # V[0,1]=1 (B00)
    # W[2,1]=1 (C10)
    
    # t=2: M3
    # U[0,2]=1 (A00)
    # V[1,2]=1, V[3,2]=-1 (B01, B11)
    # W[1,2]=1 (C01)
    
    # t=3: M4
    # U[3,3]=1 (A11)
    # V[2,3]=1, V[0,3]=-1 (B10, B00)
    # W[0,3]=-1 (C00), W[2,3]=1 (C10)
    
    # t=4: M5
    # U[0,4]=1, U[1,4]=1 (A00, A01)
    # V[3,4]=1 (B11)
    # W[1,4]=1 (C01), W[3,4]=1 (C11)
    
    # t=5: M6
    # U[2,5]=1, U[0,5]=-1 (A10, A00)
    # V[0,5]=1, V[1,5]=1 (B00, B01)
    # W[3,5]=1 (C11)
    
    # t=6: M7
    # U[1,6]=1, U[3,6]=-1 (A01, A11)
    # V[2,6]=1, V[3,6]=1 (B10, B11)
    # W[0,6]=-1 (C00)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1
    
    # M3
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1
    
    # M4
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=-1; W2[2,3]=1
    
    # M5
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[1,4]=1; W2[3,4]=1
    
    # M6
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=-1

    # Now, construct 4x4x4 using 2x2 blocks.
    # A is 4x4, B is 4x4, C is 4x4.
    # A = [[A00, A01], [A10, A11]] where Ai are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bi are 2x2.
    # C = [[C00, C01], [C10, C11]] where Ci are 2x2.
    
    # We apply Strassen 7 times, once for each 2x2 block multiplication?
    # No, Strassen 4x4x4 uses 49 multiplications.
    # It applies Strassen 2x2x2 to the blocks.
    # There are 4 block multiplications: C00=A00B00+A01B10, etc.
    # But Strassen for 4x4x4 is just Strassen 2x2x2 applied to the 2x2 blocks of 2x2 matrices.
    # So we have 7 "super-multiplications".
    # Each super-multiplication is a 2x2x2 Strassen step on 2x2 matrices.
    # So we have 7 * 7 = 49 scalar multiplications.
    
    # Let's build the 4x4x4 matrices.
    # U4: 16x49
    # V4: 16x49
    # W4: 16x49
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # The 7 Strassen steps for the 4x4x4 problem operate on the 2x2 blocks.
    # Let's denote the block indices as (I, J) for A, (J, K) for B, (K, I) for C.
    # I, J, K in {0, 1}.
    
    # The 7 Strassen combinations for the block level:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Each S_t is a 2x2 matrix product computed via 2x2x2 Strassen.
    # So for each S_t, we have 7 scalar multiplications.
    # Total 49.
    
    # We need to map the block-level Strassen coefficients to the 4x4x4 indices.
    # U4 row index: i*4 + j (0..15)
    # V4 row index: j*4 + k (0..15)
    # W4 row index: k*4 + i (0..15)
    
    # For each block-level step s in 0..6:
    #   We have coefficients for A-blocks, B-blocks, C-blocks.
    #   For each scalar step t in 0..6 within that block step:
    #     We have coefficients for A-entries, B-entries, C-entries within the blocks.
    
    # Let's define the block-level Strassen matrices (7x3 for each of A, B, C? No, 7x4 for each block matrix).
    # U_block: 4x7 (coeffs for A00, A01, A10, A11)
    # V_block: 4x7 (coeffs for B00, B01, B10, B11)
    # W_block: 4x7 (coeffs for C00, C01, C10, C11)
    
    # U_block[s, I*2+J] = coeff of A_{I,J} in S_s
    # V_block[s, J*2+K] = coeff of B_{J,K} in S_s
    # W_block[s, K*2+I] = coeff of C_{K,I} in S_s
    
    U_block = np.zeros((4, 7))
    V_block = np.zeros((4, 7))
    W_block = np.zeros((4, 7))
    
    # S1
    U_block[0, 0]=1; U_block[0, 3]=1 # A00, A11
    V_block[0, 0]=1; V_block[0, 3]=1 # B00, B11
    W_block[0, 0]=1; W_block[0, 3]=1 # C00, C11
    
    # S2
    U_block[1, 2]=1; U_block[1, 3]=1 # A10, A11
    V_block[1, 0]=1 # B00
    W_block[1, 2]=1 # C10
    
    # S3
    U_block[2, 0]=1 # A00
    V_block[2, 1]=1; V_block[2, 3]=-1 # B01, B11
    W_block[2, 1]=1 # C01
    
    # S4
    U_block[3, 3]=1 # A11
    V_block[3, 2]=1; V_block[3, 0]=-1 # B10, B00
    W_block[3, 0]=-1; W_block[3, 2]=1 # C00, C10
    
    # S5
    U_block[4, 0]=1; U_block[4, 1]=1 # A00, A01
    V_block[4, 3]=1 # B11
    W_block[4, 1]=1; W_block[4, 3]=1 # C01, C11
    
    # S6
    U_block[5, 2]=1; U_block[5, 0]=-1 # A10, A00
    V_block[5, 0]=1; V_block[5, 1]=1 # B00, B01
    W_block[5, 3]=1 # C11
    
    # S7
    U_block[6, 1]=1; U_block[6, 3]=-1 # A01, A11
    V_block[6, 2]=1; V_block[6, 3]=1 # B10, B11
    W_block[6, 0]=-1 # C00
    
    # Now fill U4, V4, W4
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U4
            for I in range(2):
                for J in range(2):
                    coeff_block = U_block[s, I*2+J]
                    if coeff_block != 0:
                        # Inside block A_{I,J}, we use U2[t, i,j]
                        for i in range(2):
                            for j in range(2):
                                coeff_scalar = U2[i*2+j, t]
                                if coeff_scalar != 0:
                                    global_i = I*2 + i
                                    global_j = J*2 + j
                                    U4[global_i*4 + global_j, col] += coeff_block * coeff_scalar
                                    
            # V4
            for J in range(2):
                for K in range(2):
                    coeff_block = V_block[s, J*2+K]
                    if coeff_block != 0:
                        for j in range(2):
                            for k in range(2):
                                coeff_scalar = V2[j*2+k, t]
                                if coeff_scalar != 0:
                                    global_j = J*2 + j
                                    global_k = K*2 + k
                                    V4[global_j*4 + global_k, col] += coeff_block * coeff_scalar
                                    
            # W4
            for K in range(2):
                for I in range(2):
                    coeff_block = W_block[s, K*2+I]
                    if coeff_block != 0:
                        for k in range(2):
                            for i in range(2):
                                coeff_scalar = W2[k*2+i, t]
                                if coeff_scalar != 0:
                                    global_k = K*2 + k
                                    global_i = I*2 + i
                                    W4[global_k*4 + global_i, col] += coeff_block * coeff_scalar

    # 2. Construct Schoolbook decomposition for 4x4x3
    # Rank 48
    U3 = np.zeros((16, 48))
    V3 = np.zeros((12, 48))
    W3 = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U3[i*4+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*4+i, col] = 1
                col += 1

    # 3. Combine
    # Total rank = 49 + 48 = 97
    U_final = np.zeros((16, 97))
    V_final = np.zeros((28, 97))
    W_final = np.zeros((28, 97))
    
    # Part 1: 4x4x4 (k=0..3)
    # U_final[:, 0:49] = U4
    # V_final rows for k=0..3 are 0..15. Rows for k=4..6 are 16..27.
    # V4 has shape (16, 49). It corresponds to j=0..3, k=0..3.
    # Global V index: j*7 + k.
    # For j=0..3, k=0..3, indices are 0..3, 7..10, 14..17, 21..24.
    # We need to place V4 into V_final.
    for j in range(4):
        for k in range(4):
            local_idx = j*4 + k
            global_idx = j*7 + k
            V_final[global_idx, 0:49] = V4[local_idx, :]
            
    # W4 has shape (16, 49). It corresponds to k=0..3, i=0..3.
    # Global W index: k*4 + i.
    # For k=0..3, i=0..3, indices are 0..15.
    W_final[0:16, 0:49] = W4
    
    U_final[:, 0:49] = U4

    # Part 2: 4x4x3 (k=4..6)
    # U_final[:, 49:97] = U3
    # V3 has shape (12, 48). It corresponds to j=0..3, k_local=0..2.
    # Global k = 4 + k_local.
    # Global V index: j*7 + (4 + k_local).
    for j in range(4):
        for k_local in range(3):
            local_idx = j*3 + k_local
            global_k = 4 + k_local
            global_idx = j*7 + global_k
            V_final[global_idx, 49:97] = V3[local_idx, :]
            
    # W3 has shape (12, 48). It corresponds to k_local=0..2, i=0..3.
    # Global k = 4 + k_local.
    # Global W index: (4 + k_local)*4 + i.
    for k_local in range(3):
        for i in range(4):
            local_idx = k_local*4 + i
            global_k = 4 + k_local
            global_idx = global_k*4 + i
            W_final[global_idx, 49:97] = W3[local_idx, :]
            
    U_final[:, 49:97] = U3

    # Convert to lists of lists for the interface
    # Ensure entries are integers or half-integers (they are integers here)
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
