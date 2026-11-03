import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen for 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U indices for 2x2: (i,j) -> i*2+j
    # 0: a, 1: b, 2: c, 3: d
    U_22 = np.zeros((4, 7))
    # 1: (a+d) -> a:1, d:1
    U_22[0, 0] = 1; U_22[3, 0] = 1
    # 2: (c+d) -> c:1, d:1
    U_22[2, 1] = 1; U_22[3, 1] = 1
    # 3: a -> a:1
    U_22[0, 2] = 1
    # 4: d -> d:1
    U_22[3, 3] = 1
    # 5: (a+b) -> a:1, b:1
    U_22[0, 4] = 1; U_22[1, 4] = 1
    # 6: (c-a) -> c:1, a:-1
    U_22[2, 5] = 1; U_22[0, 5] = -1
    # 7: (b-d) -> b:1, d:-1
    U_22[1, 6] = 1; U_22[3, 6] = -1
    
    # V indices for 2x2: (j,k) -> j*2+k
    # 0: e, 1: f, 2: g, 3: h
    V_22 = np.zeros((4, 7))
    # 1: (e+h) -> e:1, h:1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    # 2: e -> e:1
    V_22[0, 1] = 1
    # 3: (f-h) -> f:1, h:-1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    # 4: (g-e) -> g:1, e:-1
    V_22[2, 3] = 1; V_22[0, 3] = -1
    # 5: h -> h:1
    V_22[3, 4] = 1
    # 6: (e+f) -> e:1, f:1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    # 7: (g+h) -> g:1, h:1
    V_22[2, 6] = 1; V_22[3, 6] = 1
    
    # W indices for 2x2: (k,i) -> k*2+i
    # 0: C00, 1: C01, 2: C10, 3: C11
    W_22 = np.zeros((4, 7))
    # C00 = m1 + m4 - m5 + m7
    W_22[0, 0] = 1; W_22[0, 3] = 1; W_22[0, 4] = -1; W_22[0, 6] = 1
    # C01 = m3 + m5
    W_22[1, 2] = 1; W_22[1, 4] = 1
    # C10 = m2 + m4
    W_22[2, 1] = 1; W_22[2, 3] = 1
    # C11 = m1 - m2 + m3 + m6
    W_22[3, 0] = 1; W_22[3, 1] = -1; W_22[3, 2] = 1; W_22[3, 5] = 1
    
    # 2. Trivial for 1x2x2
    # A = [a, b], B = [[e, f], [g, h]]
    # C0 = ae + bg, C1 = af + bh
    # Rank 4
    U_122 = np.zeros((2, 4))
    U_122[0, 0] = 1 # a
    U_122[0, 2] = 1 # a
    U_122[1, 1] = 1 # b
    U_122[1, 3] = 1 # b
    
    V_122 = np.zeros((4, 4))
    V_122[0, 0] = 1 # e
    V_122[2, 1] = 1 # g
    V_122[1, 2] = 1 # f
    V_122[3, 3] = 1 # h
    
    W_122 = np.zeros((2, 4))
    # C0 = m0 + m1
    W_122[0, 0] = 1; W_122[0, 1] = 1
    # C1 = m2 + m3
    W_122[1, 2] = 1; W_122[1, 3] = 1
    
    # 3. Construct 2x4x4 using Kronecker of 2x2x2 and 1x2x2
    # U_244 = kron(U_22, U_122) ?
    # Dimensions:
    # U_22: 4x7. U_122: 2x4.
    # Kron U: (4*2) x (7*4) = 8x28. Correct.
    # But indices must match.
    # U_22 index (i,j) where i in 0..1, j in 0..1.
    # U_122 index (i',j') where i' in 0..0, j' in 0..1.
    # Combined index (i, j, i', j')?
    # For M_{2,4,4}, A is 2x4.
    # A index (i, j) where i in 0..1, j in 0..3.
    # We split j into j1 (0..1) and j2 (0..1)?
    # The Kronecker product M_A \otimes M_B computes:
    # A_combined[i, j] = A_A[i, j_A] * A_B[i_B, j_B]? No.
    # Standard Kronecker:
    # A is n_A n_B x m_A m_B.
    # A[i, j] where i = i_A n_B + i_B, j = j_A m_B + j_B.
    # Here n_A=2, n_B=1. m_A=2, m_B=2.
    # i = i_A * 1 + i_B = i_A + i_B? No, i_B is 0.
    # So i = i_A.
    # j = j_A * 2 + j_B.
    # So j_A is the high bit of j (0 or 1), j_B is low bit.
    # This matches splitting the 4 columns into two groups of 2.
    
    U_244 = np.kron(U_22, U_122)
    V_244 = np.kron(V_22, V_122)
    W_244 = np.kron(W_22, W_122)
    
    # Check W indices for 2x4x4.
    # W_22 index (k, i) -> k*2 + i.
    # W_122 index (k', i') -> k'*1 + i' = k'.
    # Kron W index (k, i, k', i') -> (k*1 + k')*2 + (i*1 + i')?
    # No, Kron of vectors u (size N) and v (size M) gives size NM.
    # Index is i_u * M + i_v.
    # W_22 size 4. W_122 size 2.
    # Kron size 8.
    # Index = idx_22 * 2 + idx_122.
    # idx_22 = k*2 + i.
    # idx_122 = k'.
    # So index = (k*2 + i)*2 + k' = 4k + 2i + k'.
    # For M_{2,4,4}, n=2, p=4.
    # W index should be k*n + i = 4k + i.
    # Our index is 4k + 2i + k'.
    # This does NOT match 4k + i.
    # The Kronecker product ordering might be swapped or need permutation.
    # Let's check the tensor definition.
    # T[i,j,k] for M_{n,m,p}.
    # Kron M_A \otimes M_B:
    # i = i_A n_B + i_B
    # j = j_A m_B + j_B
    # k = k_A p_B + k_B
    # W index is k*n + i = (k_A p_B + k_B) (n_A n_B) + (i_A n_B + i_B)
    # = k_A p_B n_A n_B + k_B n_A n_B + i_A n_B + i_B.
    # In our case:
    # n_A=2, n_B=1. p_A=2, p_B=2.
    # n = 2*1 = 2. p = 2*2 = 4.
    # W index = (k_A 2 + k_B) 2 + (i_A 1 + i_B)
    # = 4 k_A + 2 k_B + i_A + i_B.
    # Since n_B=1, i_B=0.
    # W index = 4 k_A + 2 k_B + i_A.
    # This matches the Kron index derived above: 4k_A + 2i_A + k_B (if we map i->i_A, k->k_A, k'->k_B).
    # Wait, in W_22, index is k_A * 2 + i_A.
    # In W_122, index is k_B.
    # Kron index = (k_A * 2 + i_A) * 2 + k_B = 4 k_A + 2 i_A + k_B.
    # Target index for M_{2,4,4} with n=2, p=4 is k * 2 + i.
    # Here k = k_A * 2 + k_B. i = i_A.
    # Target = (k_A * 2 + k_B) * 2 + i_A = 4 k_A + 2 k_B + i_A.
    # My Kron index: 4 k_A + 2 i_A + k_B.
    # They are different if i_A and k_B are swapped.
    # i_A is 0 or 1. k_B is 0 or 1.
    # We need to permute the columns/rows of the Kron product?
    # Actually, the W matrix rows correspond to the output indices.
    # We can just construct the final W by mapping the rows.
    
    # 4. Construct 1x4x4
    # U_144 = kron(U_111, U_122)?
    # M_{1,1,1} rank 1. U=[1], V=[1], W=[1].
    # M_{1,4,4} = M_{1,1,1} \otimes M_{1,2,2} \otimes M_{1,2,2}?
    # Simpler: Just use the 16 rank-1 terms.
    U_144 = np.zeros((4, 16))
    V_144 = np.zeros((16, 16))
    W_144 = np.zeros((4, 16))
    for k in range(4):
        for i in range(4): # i is index of A (0..3)
            idx = k * 4 + i
            U_144[i, idx] = 1
            V_144[k * 4 + i, idx] = 1 # V index j*p + k? No.
            # V index for M_{1,4,4}: j (row of B) is 0..3, k (col of B) is 0..3.
            # V index = j * 4 + k.
            # Here j=i (since A_i multiplies B_{ik}).
            # So V index = i * 4 + k.
            V_144[i * 4 + k, idx] = 1
            W_144[k, idx] = 1 # W index k (since n=1)
            
    # 5. Combine
    rank_244 = U_244.shape[1]
    rank_144 = U_144.shape[1]
    total_rank = rank_244 + rank_144
    
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # Fill U
    # Rows 0-7 from U_244
    U[0:8, 0:rank_244] = U_244
    # Rows 8-11 from U_144
    # U_144 has 4 rows (i=0..3).
    # In M_{3,4,4}, the 3rd row of A (i=2) corresponds to these.
    # Global row index for A: i_global * 4 + j.
    # For i_global=2, rows are 8, 9, 10, 11.
    # U_144 row i_sub (0..3) maps to global row 8 + i_sub.
    U[8:12, rank_244:total_rank] = U_144
    
    # Fill V
    # V is same for both?
    # V_244 is 16x28.
    # V_144 is 16x16.
    # Just concatenate columns.
    V[:, 0:rank_244] = V_244
    V[:, rank_244:total_rank] = V_144
    
    # Fill W
    # W_244 is 8x28.
    # W_144 is 4x16.
    # Map W_244 rows to global W rows.
    # W_244 row idx = 4 k_A + 2 i_A + k_B.
    # Global row for i=0,1: 4 k + i.
    # k = 2 k_A + k_B.
    # Global row = 4 (2 k_A + k_B) + i_A = 8 k_A + 4 k_B + i_A.
    # This mapping is not 1-to-1 with simple slicing.
    # We must iterate.
    for idx_sub in range(8):
        # Decode idx_sub
        # idx_sub = (k_A * 2 + i_A) * 2 + k_B
        # k_B = idx_sub % 2
        # rem = idx_sub // 2
        # i_A = rem % 2
        # k_A = rem // 2
        k_B = idx_sub % 2
        rem = idx_sub // 2
        i_A = rem % 2
        k_A = rem // 2
        
        k_global = 2 * k_A + k_B
        i_global = i_A
        row_global = k_global * 3 + i_global
        
        W[row_global, 0:rank_244] += W_244[idx_sub, :]
        
    # Map W_144 rows to global W rows.
    for idx_sub in range(4):
        # idx_sub is k (0..3)
        k_global = idx_sub
        i_global = 2
        row_global = k_global * 3 + i_global
        
        W[row_global, rank_244:total_rank] += W_144[idx_sub, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
