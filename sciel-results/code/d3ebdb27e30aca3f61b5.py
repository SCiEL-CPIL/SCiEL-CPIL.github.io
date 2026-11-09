import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _construct_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n = 4
    m = 4
    p = 4
    U_sub, V_sub, W_sub = _strassen_2x2()
    
    U = np.zeros((n*m, 49))
    V = np.zeros((m*p, 49))
    W = np.zeros((p*n, 49))
    
    # Strassen blocks for 4x4:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    
    # We need to map the 4x4 indices to the 2x2 block indices.
    # A_{I,J} where I, J in 0..1.
    # A_{I,J} is a 2x2 matrix.
    # Entry A_{i,j} where i=I*2+u, j=J*2+v.
    
    # For each Strassen term r (0..6), we have a sub-decomposition.
    # The sub-decomposition uses U_sub, V_sub, W_sub.
    # U_sub indices correspond to the 2x2 block entries.
    # We need to scale/shift U_sub to the 4x4 indices.
    
    # The Strassen linear forms are on the 2x2 blocks.
    # Let's define the mapping from 4x4 indices to 2x2 block indices.
    # This is essentially a Kronecker product structure.
    
    # For each r in 0..6:
    #   For each s in 0..6 (sub-rank):
    #      rank_idx = r * 7 + s
    #      U[4x4_idx, rank_idx] = U_sub[2x2_idx, s] * coeff_A
    #      This is tricky because U_sub already contains the coefficients for the 2x2 multiplication.
    #      Actually, U_sub, V_sub, W_sub are for the 2x2x2 tensor.
    #      The 4x4x4 tensor is composed of 7 such 2x2x2 tensors, but the "inputs" to these 2x2x2 tensors are linear combinations of the 4x4 entries.
    #      Wait, no. Strassen computes 7 products of 2x2 matrices.
    #      Each product is a 2x2x2 multiplication.
    #      So we just need to place the 2x2x2 decomposition into the 4x4x4 space.
    #      The "inputs" to the 2x2x2 multiplication are the blocks themselves.
    #      So U_sub maps block-entries to the rank.
    #      We need to map 4x4-entries to block-entries.
    #      This is just an identity mapping on the block level, but expanded.
    
    # Let's verify.
    # M_r = sum_{u,v} A_block_{u,v} B_block_{u,v}
    # A_block_{u,v} is a 2x2 matrix.
    # So M_r = sum_{u,v, a,b} A_{I_u, J_v, a, b} ...
    # This is getting complex.
    # Simpler: The rank-49 decomposition is the Kronecker product of the rank-7 decomposition with itself?
    # No, it's 7 copies of rank-7.
    
    # Let's just use the property that M_{4,4,4} is the tensor product of M_{2,2,2} with M_{2,2,2}?
    # Rank(A x B) = Rank(A) * Rank(B).
    # Rank(M_{2,2,2}) = 7.
    # Rank(M_{4,4,4}) <= 7 * 7 = 49.
    # The decomposition is the tensor product.
    
    U_kron = np.kron(U_sub, U_sub)
    V_kron = np.kron(V_sub, V_sub)
    W_kron = np.kron(W_sub, W_sub)
    
    # Check shapes:
    # U_sub (4, 7). U_kron (16, 49). Correct.
    # V_sub (4, 7). V_kron (16, 49). Correct.
    # W_sub (4, 7). W_kron (16, 49). Correct.
    
    return U_kron, V_kron, W_kron

def construct():
    n, m, p = DIMENSIONS
    # p=7. We split into 4 and 3.
    # Use two 4x4x4 decompositions.
    U1, V1, W1 = _construct_4x4x4()
    U2, V2, W2 = _construct_4x4x4()
    
    # We need to adjust V and W for the second part to correspond to columns 4..6 of B.
    # B is 4x7. B1 is 4x4 (cols 0-3). B2 is 4x3 (cols 4-6).
    # We treat B2 as 4x4 with last col 0.
    # So V2 should have non-zeros only for j, k where k < 4?
    # V index is j*p + k.
    # For V1, k is 0..3.
    # For V2, k is 4..6? No, we pad to 0..3 and shift?
    # If we use the same V2, it will multiply B_{j, 0..3} again.
    # We want it to multiply B_{j, 4..6}.
    # So we need to shift the k-index in V2 by 4?
    # V2 has shape (16, 49). Rows are j*4 + k (since p=4 in the sub-problem).
    # But in the final problem, p=7.
    # So V2 rows should be j*7 + (k+4).
    
    U = np.zeros((n*m, 98))
    V = np.zeros((m*p, 98))
    W = np.zeros((p*n, 98))
    
    # First part: cols 0-3
    U[:, 0:49] = U1
    V[:, 0:49] = 0
    for r in range(49):
        for row in range(16):
            j = row // 4
            k = row % 4
            V[j*7 + k, r] = V1[row, r]
    W[:, 0:49] = W1
    
    # Second part: cols 4-6 (padded to 4)
    U[:, 49:98] = U2
    V[:, 49:98] = 0
    for r in range(49):
        for row in range(16):
            j = row // 4
            k = row % 4
            # Map k (0..3) to 4..7 (but we only have 4..6, so 4..6 are used, 7 is ignored/zero)
            # Actually, we just use k+4. If k+4 >= 7, it's out of bounds.
            # But k is 0..3, so k+4 is 4..7.
            # p=7, so valid k are 0..6.
            # k=3 -> 7 is invalid.
            # So we lose the last column of the padded B2?
            # B2 is 4x3. We padded to 4x4.
            # The 4th column of B2 is 0.
            # So even if we map to index 7, it doesn't exist.
            # We should only map k=0,1,2 to 4,5,6.
            # k=3 corresponds to the padded column, which is 0.
            # So we can ignore k=3.
            if k < 3:
                V[j*7 + (k+4), r + 49] = V2[row, r]
            # else: V2[row, r] * B_{j, 3} (padded) = 0. So we can skip or set to 0.
            
    W[:, 49:98] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
