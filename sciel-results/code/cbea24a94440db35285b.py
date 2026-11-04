"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a 4+1 block decomposition.
    # Split 5x5 into 4x4, 4x1, 1x4, 1x1 blocks.
    # Rank of 4x4x4 multiplication is 49 (Strassen x Strassen).
    # Rank of cross terms is small.
    
    # 1. Construct 4x4x4 decomposition of rank 49
    # This is the Kronecker product of two 2x2x2 Strassen decompositions.
    
    # Strassen 2x2x2 decomposition
    # U, V, W for 2x2x2
    # Indices: i, j for U (2*2=4), j, k for V (2*2=4), k, i for W (2*2=4)
    # Rank 7
    U_2 = [
        [1, 0, 0, 0], # A11
        [0, 0, 1, 0], # A12
        [0, 1, 0, 0], # A21
        [0, 0, 0, 1]  # A22
    ]
    # Wait, standard Strassen coefficients.
    # Let's use a known list of (u, v, w) triples for 2x2x2.
    # u is vector of 4, v is vector of 4, w is vector of 4.
    # w = u * v (element-wise product of linear forms)
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for 2x2 matrix: 00, 01, 10, 11
    # U vectors (for A):
    u1 = [1, 0, 0, 1] # A11 + A22
    u2 = [0, 0, 1, 1] # A21 + A22
    u3 = [1, 0, 0, 0] # A11
    u4 = [0, 0, 0, 1] # A22
    u5 = [1, 1, 0, 0] # A11 + A12
    u6 = [-1, 0, 1, 0] # A21 - A11
    u7 = [0, 1, 0, -1] # A12 - A22
    
    # V vectors (for B):
    v1 = [1, 1, 1, 1] # B11 + B22 (Wait, B11 is 00, B12 is 01, B21 is 10, B22 is 11)
    # B11=1, B12=0, B21=0, B22=1 -> [1,0,0,1]
    # Let's be precise.
    # B matrix: [[B11, B12], [B21, B22]]
    # Index 0: B11, 1: B12, 2: B21, 3: B22
    
    v1 = [1, 0, 0, 1] # B11 + B22
    v2 = [1, 0, 0, 0] # B11
    v3 = [0, 1, 0, -1] # B12 - B22
    v4 = [0, 0, 1, -1] # B21 - B11 (Wait, B21 is index 2, B11 is index 0) -> [ -1, 0, 1, 0 ]?
    # M4 = A22 (B21 - B11). A22 is index 3.
    # B21 - B11 = [ -1, 0, 1, 0 ]
    v4 = [-1, 0, 1, 0]
    
    v5 = [0, 0, 0, 1] # B22
    v6 = [1, 1, 0, 0] # B11 + B12
    v7 = [0, 0, 1, 1] # B21 + B22
    
    # W vectors (for C):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 contributes to C11 (1), C22 (1)
    # M2 contributes to C21 (1), C22 (-1)
    # M3 contributes to C12 (1), C22 (1)
    # M4 contributes to C11 (1), C21 (1)
    # M5 contributes to C11 (-1), C12 (1)
    # M6 contributes to C22 (1)
    # M7 contributes to C11 (1)
    
    w1 = [1, 0, 0, 1] # C11, C22
    w2 = [0, 0, 1, -1] # C21, C22
    w3 = [0, 1, 0, 1] # C12, C22
    w4 = [1, 0, 1, 0] # C11, C21
    w5 = [-1, 1, 0, 0] # C11, C12
    w6 = [0, 0, 0, 1] # C22
    w7 = [1, 0, 0, 0] # C11
    
    # Check sums:
    # C11: w1(1) + w4(1) + w5(-1) + w7(1) = 2. Should be 1?
    # Strassen: C11 = M1 + M4 - M5 + M7.
    # M1: C11=1. M4: C11=1. M5: C11=-1. M7: C11=1. Sum = 2.
    # Wait, Strassen formula:
    # C11 = M1 - M5 + M4 + M7
    # M1 = (A11+A22)(B11+B22)
    # M5 = (A11+A12)B22
    # M4 = A22(B21-B11)
    # M7 = (A12-A22)(B21+B22)
    # Let's check A11B11 term:
    # M1: A11B11. M5: -A11B22. M4: 0. M7: 0.
    # Result: A11B11. Correct.
    # Let's check A11B12 term:
    # M1: A11B12. M5: 0. M4: 0. M7: A12B21? No.
    # M7 = A12B21 + A12B22 - A22B21 - A22B22.
    # No A11B12 in M7.
    # So A11B12 comes from M1 only?
    # M1 = A11B11 + A11B22 + A22B11 + A22B22.
    # Wait, M1 has A11B11 and A11B22.
    # C11 should have A11B11 + A12B21.
    # My M1 definition: (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is correct for M1.
    
    # The W vectors represent the coefficients of the rank-1 tensors.
    # The tensor is sum_k U_k \otimes V_k \otimes W_k.
    # The resulting matrix C_{ij} = sum_k U_k[i] V_k[j] W_k[i+j? No.
    # The indices are (i, j, k) for A_i, B_j, C_k.
    # C_k = sum_r U_r[i] V_r[j] W_r[k] ?
    # Standard form: C_{ij} = sum_r (sum_k A_{ik} U_{r, ik}) (sum_l B_{kj} V_{r, kj}) W_{r, ij} ?
    # No, the decomposition is:
    # C = sum_r U_r * V_r * W_r
    # where U_r, V_r, W_r are matrices.
    # In our flattened form:
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # C[i, j] = sum_k sum_r U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
    
    # So for 2x2x2:
    # U is 4x7, V is 4x7, W is 4x7.
    # U[i*2+k, r] is the coefficient of A_{ik} in the r-th term.
    # V[k*2+j, r] is the coefficient of B_{kj} in the r-th term.
    # W[j*2+i, r] is the coefficient of C_{ji} in the r-th term?
    # Task says: W[k*n+i, r].
    # So W index is k*n + i.
    # C[i, j] = sum_k sum_r U[i*m + k, r] V[k*p + j, r] W[j*n + i, r]
    
    # Let's verify the W vectors with this convention.
    # C11 (i=0, j=0): W[0*2+0, r] = W[0, r].
    # C12 (i=0, j=1): W[1*2+0, r] = W[2, r].
    # C21 (i=1, j=0): W[0*2+1, r] = W[1, r].
    # C22 (i=1, j=1): W[1*2+1, r] = W[3, r].
    
    # My previous W vectors were indexed [C11, C12, C21, C22] which is [0, 2, 1, 3].
    # Let's reorder W vectors to match [W0, W1, W2, W3] = [C11, C21, C12, C22].
    # W0 (C11): 1, 0, 0, 1, -1, 0, 1
    # W1 (C21): 0, 1, 0, 1, 0, 0, 0
    # W2 (C12): 0, 0, 1, 0, 1, 0, 0
    # W3 (C22): 1, -1, 1, 0, 0, 1, 0
    
    # Let's re-verify C11 sum:
    # M1: 1. M4: 1. M5: -1. M7: 1. Sum = 2.
    # But C11 = A11B11 + A12B21.
    # M1 has A11B11.
    # M4 has 0.
    # M5 has 0.
    # M7 has A12B21.
    # So sum should be 1.
    # Where is the error?
    # M1 = (A11+A22)(B11+B22).
    # M4 = A22(B21-B11).
    # M5 = (A11+A12)B22.
    # M7 = (A12-A22)(B21+B22).
    
    # A11B11 terms:
    # M1: 1.
    # M4: 0.
    # M5: 0.
    # M7: 0.
    # Sum: 1. Correct.
    
    # A12B21 terms:
    # M1: 0.
    # M4: 0.
    # M5: 0.
    # M7: 1.
    # Sum: 1. Correct.
    
    # A11B22 terms:
    # M1: 1.
    # M4: 0.
    # M5: -1.
    # M7: 0.
    # Sum: 0. Correct.
    
    # A22B11 terms:
    # M1: 1.
    # M4: -1.
    # M5: 0.
    # M7: 0.
    # Sum: 0. Correct.
    
    # A22B22 terms:
    # M1: 1.
    # M4: 0.
    # M5: 0.
    # M7: -1.
    # Sum: 0. Correct.
    
    # So the W coefficients are correct.
    
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ])
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ])
    
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ])
    
    # Now Kronecker product for 4x4x4
    # U_4[i*4+k, r1*7+r2] = U_2[i*2+i2, r1] * U_2[i2*2+k2, r2] ?
    # No, the Kronecker product of decompositions.
    # If T1 = sum U1 \otimes V1 \otimes W1 and T2 = sum U2 \otimes V2 \otimes W2
    # Then T1 \otimes T2 = sum_{r1, r2} (U1[r1] \otimes U2[r2]) \otimes (V1[r1] \otimes V2[r2]) \otimes (W1[r1] \otimes W2[r2])
    
    U_4 = np.kron(U_2, U_2)
    V_4 = np.kron(V_2, V_2)
    W_4 = np.kron(W_2, W_2)
    
    # Now handle 5x5x5 with 4+1 split.
    # U_5 is (25, rank), V_5 is (25, rank), W_5 is (25, rank).
    # Rank = 49 + 25 = 74.
    
    rank_44 = 49
    rank_cross = 25
    total_rank = rank_44 + rank_cross
    
    U_5 = np.zeros((25, total_rank))
    V_5 = np.zeros((25, total_rank))
    W_5 = np.zeros((25, total_rank))
    
    # 1. 4x4x4 part (top-left 4x4 of A, B, C)
    # A indices 0-3, B indices 0-3, C indices 0-3.
    # U_5[i*5+k, r] = U_4[i*4+k, r] for i,k in 0-3.
    # V_5[k*5+j, r] = V_4[k*4+j, r] for k,j in 0-3.
    # W_5[j*5+i, r] = W_4[j*4+i, r] for j,i in 0-3.
    
    for i in range(4):
        for k in range(4):
            for r in range(49):
                U_5[i*5 + k, r] = U_4[i*4 + k, r]
                
    for k in range(4):
        for j in range(4):
            for r in range(49):
                V_5[k*5 + j, r] = V_4[k*4 + j, r]
                
    for j in range(4):
        for i in range(4):
            for r in range(49):
                W_5[j*5 + i, r] = W_4[j*4 + i, r]
                
    # 2. Cross terms
    # B_41 G_14: A is 4x1 (col 4), B is 1x4 (row 0? No, B is 1x4 means row 4 of A? No.)
    # Split:
    # A = [[A44, A41], [A14, A11]]
    # B = [[B44, B41], [B14, B11]]
    # C = [[C44, C41], [C14, C11]]
    
    # Terms:
    # C44 = A44 B44 + A41 B14
    # C41 = A44 B41 + A41 B11
    # C14 = A14 B44 + A11 B14
    # C11 = A14 B41 + A11 B11
    
    # We handled A44 B44.
    # Now handle:
    # A41 B14: A41 is 4x1, B14 is 1x4.
    # A41 indices: i in 0-3, k=4.
    # B14 indices: k=4, j in 0-3.
    # C44 contribution: i in 0-3, j in 0-3.
    # Rank 4.
    
    # A41 B11: A41 is 4x1, B11 is 1x1.
    # A41 indices: i in 0-3, k=4.
    # B11 indices: k=4, j=4.
    # C41 contribution: i in 0-3, j=4.
    # Rank 4.
    
    # A14 B44: A14 is 1x4, B44 is 4x4.
    # A14 indices: i=4, k in 0-3.
    # B44 indices: k in 0-3, j in 0-3.
    # C14 contribution: i=4, j in 0-3.
    # Rank 4.
    
    # A14 B41: A14 is 1x4, B41 is 4x1.
    # A14 indices: i=4, k in 0-3.
    # B41 indices: k in 0-3, j=4.
    # C11 contribution: i=4, j=4.
    # Rank 4.
    
    # A11 B14: A11 is 1x1, B14 is 1x4.
    # A11 indices: i=4, k=4.
    # B14 indices: k=4, j in 0-3.
    # C14 contribution: i=4, j in 0-3.
    # Rank 4.
    
    # A11 B11: A11 is 1x1, B11 is 1x1.
    # A11 indices: i=4, k=4.
    # B11 indices: k=4, j=4.
    # C11 contribution: i=4, j=4.
    # Rank 1.
    
    # Total cross rank: 4+4+4+4+4+1 = 21?
    # Wait, I said 25 earlier.
    # Let's recount.
    # A41 B14: 4
    # A41 B11: 4
    # A14 B44: 4
    # A14 B41: 4
    # A11 B14: 4
    # A11 B11: 1
    # Sum: 21.
    # Total rank: 49 + 21 = 70.
    
    # Let's implement these 21 terms.
    
    r_offset = 49
    
    # A41 B14 -> C44
    # A41: i in 0-3, k=4.
    # B14: k=4, j in 0-3.
    # C44: i in 0-3, j in 0-3.
    for t in range(4):
        r = r_offset + t
        # U: i=t, k=4
        U_5[t*5 + 4, r] = 1
        # V: k=4, j=t
        V_5[4*5 + t, r] = 1
        # W: j=t, i=t
        W_5[t*5 + t, r] = 1
        
    # A41 B11 -> C41
    # A41: i in 0-3, k=4.
    # B11: k=4, j=4.
    # C41: i in 0-3, j=4.
    for t in range(4):
        r = r_offset + 4 + t
        U_5[t*5 + 4, r] = 1
        V_5[4*5 + 4, r] = 1
        W_5[4*5 + t, r] = 1
        
    # A14 B44 -> C14
    # A14: i=4, k in 0-3.
    # B44: k in 0-3, j in 0-3.
    # C14: i=4, j in 0-3.
    for t in range(4):
        r = r_offset + 8 + t
        U_5[4*5 + t, r] = 1
        V_5[t*5 + 0, r] = 1 # Wait, B44 is k in 0-3, j in 0-3.
        # We need to sum over k.
        # This term is A14 B44.
        # (A14 B44)_j = sum_k A_4k B_kj.
        # This is a dot product.
        # It can be decomposed into 4 terms?
        # Yes, sum_k (A_4k) (B_kj).
        # Term k: U[4*5+k] = 1, V[k*5+j] = 1, W[4*5+j] = 1.
        # So for each k, we have a term that contributes to all j.
        # So we need 4 terms, one for each k.
        # Each term k contributes to C14 (i=4, j=0..3).
        # So W[4*5+j] should be 1 for all j?
        # No, W is a vector.
        # If we use 4 terms, each term k has W vector with 1s at j=0..3?
        # No, W[4*5+j] is the coefficient for C_4j.
        # If we use term k, it contributes to all C_4j.
        # So W[4*5+j] = 1 for all j.
        # But we can't have a single W vector that is 1 for all j if we want to separate them?
        # Actually, we can.
        # Term k: U[4*5+k]=1, V[k*5+j]=1 (for all j), W[4*5+j]=1 (for all j).
        # This works.
        
        # Let's correct the loop.
        # We need 4 terms for A14 B44.
        # Term k (k=0..3):
        # U[4*5+k] = 1
        # V[k*5+j] = 1 for j=0..3
        # W[4*5+j] = 1 for j=0..3
        
    # Let's rewrite the cross terms properly.
    
    # Reset W_5 for cross terms to 0 to avoid conflicts if I made a mistake above.
    # Actually, I'll just overwrite the specific indices.
    
    # A14 B44
    for k in range(4):
        r = r_offset + 8 + k
        U_5[4*5 + k, r] = 1
        for j in range(4):
            V_5[k*5 + j, r] = 1
            W_5[4*5 + j, r] = 1
            
    # A14 B41 -> C11
    # A14: i=4, k in 0-3.
    # B41: k in 0-3, j=4.
    # C11: i=4, j=4.
    for k in range(4):
        r = r_offset + 12 + k
        U_5[4*5 + k, r] = 1
        V_5[k*5 + 4, r] = 1
        W_5[4*5 + 4, r] = 1
        
    # A11 B14 -> C14
    # A11: i=4, k=4.
    # B14: k=4, j in 0-3.
    # C14: i=4, j in 0-3.
    for j in range(4):
        r = r_offset + 16 + j
        U_5[4*5 + 4, r] = 1
        V_5[4*5 + j, r] = 1
        W_5[4*5 + j, r] = 1
        
    # A11 B11 -> C11
    # A11: i=4, k=4.
    # B11: k=4, j=4.
    # C11: i=4, j=4.
    r = r_offset + 20
    U_5[4*5 + 4, r] = 1
    V_5[4*5 + 4, r] = 1
    W_5[4*5 + 4, r] = 1
    
    # Total rank used: 49 + 21 = 70.
    
    return {"U": U_5.tolist(), "V": V_5.tolist(), "W": W_5.tolist()}

# EVOLVE-BLOCK-END
