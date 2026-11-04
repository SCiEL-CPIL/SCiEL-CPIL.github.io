import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrices
    # U_strassen: 2x7, V_strassen: 2x7, W_strassen: 2x7
    # Indices: 0..6
    U_s = [
        [1, 0, 1, 0, 1, -1, 1], # A11
        [0, 1, 0, 1, 1, 0, -1]  # A22? No, A21, A22
    ]
    # Correct Strassen U:
    # M1: A11+A22 -> [1, 1]
    # M2: A21+A22 -> [1, 1]
    # M3: A11 -> [1, 0]
    # M4: A22 -> [0, 1]
    # M5: A11+A12 -> [1, 1]
    # M6: A21-A11 -> [-1, 1]
    # M7: A12-A22 -> [1, -1]
    
    U_s = [
        [1, 1, 1, 0, 1, -1, 1], # Row 0 (A11, A12)
        [1, 1, 0, 1, 1, 1, -1]  # Row 1 (A21, A22)
    ]
    
    V_s = [
        [1, 1, 1, 0, 0, 1, 0], # Row 0 (B11, B12)
        [1, 0, -1, 1, 1, 1, 1] # Row 1 (B21, B22)
    ]
    
    W_s = [
        [1, 0, 0, 1, 0, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, -1, 1, 0, 0, 1, 0] # C22
    ]
    
    for block in range(4):
        # Block is 3x2
        # B indices:
        # Row 0: 2*block, 2*block+1
        # Row 1: 2*block+2, 2*block+3 (in m*p index: 1*8 + 2*block, etc)
        # Row 2: 2*block+4, 2*block+5
        
        # Term 0: Rank 1 part (A_L * B_1)
        col = block * 8
        
        # U: A_L is A[0,0], A[1,0]
        U[0][col] = 1 # A[0,0]
        U[3][col] = 1 # A[1,0]
        
        # V: B_1 is B[0, 2*block], B[0, 2*block+1]
        V[2*block][col] = 1
        V[2*block+1][col] = 1
        
        # W: C[0, 2*block], C[0, 2*block+1], C[1, 2*block], C[1, 2*block+1]
        # W index k*n + i
        # C[0, 2*block] -> k=0, i=0 -> 0
        # C[0, 2*block+1] -> k=1, i=0 -> 1
        # C[1, 2*block] -> k=0, i=1 -> 2
        # C[1, 2*block+1] -> k=1, i=1 -> 3
        W[0][col] = 1
        W[1][col] = 1
        W[2][col] = 1
        W[3][col] = 1
        
        # Terms 1-6: Strassen part (A_R * B_2)
        for t in range(7):
            col = block * 8 + t + 1
            
            # U: A_R is A[0,1], A[0,2], A[1,1], A[1,2]
            # U_s[0] corresponds to A[0,1] (index 1)
            # U_s[1] corresponds to A[1,1] (index 4)
            # Wait, A is 2x3.
            # A[0,1] is index 1.
            # A[0,2] is index 2.
            # A[1,1] is index 4.
            # A[1,2] is index 5.
            
            # U_s row 0 is for A[0,1] and A[0,2]?
            # My U_s was defined for 2x2 A.
            # A_R = [[A01, A02], [A11, A12]]
            # U_s[0][t] multiplies A01? No.
            # U_s[0] is the coefficient for A_R[0,0] which is A01.
            # U_s[0] is also the coefficient for A_R[0,1]? No.
            # In Strassen, U is 2x7.
            # U[0, t] is coeff for A11.
            # U[1, t] is coeff for A21? No.
            # Standard Strassen:
            # U is n x r.
            # For 2x2, U is 2x7.
            # U[0, t] is coeff for A[0,0].
            # U[1, t] is coeff for A[1,0].
            # What about A[0,1] and A[1,1]?
            # They are not in U?
            # No, the tensor decomposition is T_{ijk} = sum U_i V_j W_k.
            # Here i is row of A, j is col of A/row of B, k is col of B.
            # For 2x2x2:
            # U is 2x7. V is 2x7. W is 2x7.
            # U[i, t] is coeff for A[i, 0]? No.
            # A is 2x2.
            # The decomposition is for the tensor T_{i j k}.
            # U[i, t] V[j, t] W[k, t].
            # So U[i, t] is coeff for A[i, j]? No, j is summed.
            # U[i, t] is just a number.
            # The product is sum_t U[i, t] V[j, t] W[k, t] A[i, j] B[j, k]?
            # No, A[i, j] is input.
            # The term is U[i, t] A[i, j] V[j, t] B[j, k] W[k, t]?
            # This doesn't match the shape.
            
            # Let's stick to the seed code structure.
            # Seed: U[i*m+j][col] = 1.
            # This means U is indexed by (i, j).
            # So U[i*m+j, t] is the coefficient for A[i, j].
            
            # For Strassen 2x2:
            # We need to map A_R[0,0]=A[0,1], A_R[0,1]=A[0,2], etc.
            
            # U_s[0][t] is coeff for A_R[0,0] = A[0,1] -> Index 1
            # U_s[1][t] is coeff for A_R[1,0] = A[1,1] -> Index 4
            # What about A_R[0,1] = A[0,2] and A_R[1,1] = A[1,2]?
            # They are not in U_s?
            # In the tensor T_{i j k}, i is row of A.
            # j is col of A.
            # So U is n x m x r?
            # Yes, U[i*m+j, t].
            
            # So for Strassen 2x2:
            # U[0*2+0, t] = U_s[0][t] (A11)
            # U[0*2+1, t] = U_s[0][t]? No.
            # U_s is 2x7.
            # U_s[0][t] is coeff for A[0,0].
            # U_s[1][t] is coeff for A[1,0].
            # Where are A[0,1] and A[1,1]?
            # They are not in U?
            # Ah, V is m x p x r?
            # V[j*p+k, t].
            # V[0*2+0, t] = V_s[0][t] (B11)
            # V[0*2+1, t] = V_s[0][t]? No.
            
            # The Strassen decomposition is:
            # C = sum_t U_t V_t W_t.
            # U_t is vector 2. V_t is vector 2. W_t is matrix 2x2.
            # U_t[i] is coeff for A[i, 0]? No.
            # U_t[i] is coeff for A[i, j]?
            # It's a tensor.
            
            # Let's use the explicit 10-term or 7-term formulas for the flattened indices.
            
            # For A_R (2x2) and B_2 (2x2):
            # A_R[0,0] = A[0,1] (idx 1)
            # A_R[0,1] = A[0,2] (idx 2)
            # A_R[1,0] = A[1,1] (idx 4)
            # A_R[1,1] = A[1,2] (idx 5)
            
            # B_2[0,0] = B[1, 2*block] (idx 8 + 2*block)
            # B_2[0,1] = B[1, 2*block+1] (idx 8 + 2*block+1)
            # B_2[1,0] = B[2, 2*block] (idx 16 + 2*block)
            # B_2[1,1] = B[2, 2*block+1] (idx 16 + 2*block+1)
            
            # C[0, 2*block] (idx 0)
            # C[0, 2*block+1] (idx 1)
            # C[1, 2*block] (idx 2)
            # C[1, 2*block+1] (idx 3)
            
            # Strassen terms:
            # M1: (A11+A22)(B11+B22)
            # U: A11(1), A22(5) -> U[1]+=1, U[5]+=1
            # V: B11(8+2b), B22(16+2b+1) -> V[8+2b]+=1, V[16+2b+1]+=1
            # W: C11(0), C12(1), C21(2), C22(3) -> W[0]+=1, W[1]+=0, W[2]+=0, W[3]+=1?
            # C11 = M1 + ...
            # C12 = ...
            # C21 = ...
            # C22 = ...
            
            # We need to add contributions to W for each C element.
            
            # M1:
            U[1][col] += 1
            U[5][col] += 1
            V[8 + 2*block][col] += 1
            V[16 + 2*block + 1][col] += 1
            W[0][col] += 1
            W[3][col] += 1
            
            # M2: (A21+A22)B11
            U[4][col] += 1
            U[5][col] += 1
            V[8 + 2*block][col] += 1
            W[2][col] += 1
            W[3][col] -= 1
            
            # M3: A11(B12-B22)
            U[1][col] += 1
            V[8 + 2*block + 1][col] += 1
            V[16 + 2*block + 1][col] -= 1
            W[1][col] += 1
            W[3][col] += 1
            
            # M4: A22(B21-B11)
            U[5][col] += 1
            V[16 + 2*block][col] += 1
            V[8 + 2*block][col] -= 1
            W[0][col] += 1
            W[2][col] += 1
            
            # M5: (A11+A12)B22
            U[1][col] += 1
            U[2][col] += 1
            V[16 + 2*block + 1][col] += 1
            W[0][col] -= 1
            W[1][col] += 1
            
            # M6: (A21-A11)(B11+B12)
            U[4][col] += 1
            U[1][col] -= 1
            V[8 + 2*block][col] += 1
            V[8 + 2*block + 1][col] += 1
            W[3][col] += 1
            
            # M7: (A12-A22)(B21+B22)
            U[2][col] += 1
            U[5][col] -= 1
            V[16 + 2*block][col] += 1
            V[16 + 2*block + 1][col] += 1
            W[0][col] += 1
            W[3][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
