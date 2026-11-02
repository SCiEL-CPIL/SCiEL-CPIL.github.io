"""Matrix multiplication 2x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    U_rows = n * m  # 8
    V_rows = m * p  # 20
    W_rows = p * n  # 10
    
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries, v_entries, w_entries are dicts: {index: value}"""
        u_vec = [0.0] * U_rows
        v_vec = [0.0] * V_rows
        w_vec = [0.0] * W_rows
        for idx, val in u_entries.items():
            u_vec[idx] = val
        for idx, val in v_entries.items():
            v_vec[idx] = val
        for idx, val in w_entries.items():
            w_vec[idx] = val
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Index helpers
    def u_idx(i, j):
        """U index for A[i,j], i in 0..1, j in 0..3"""
        return i * m + j
    
    def v_idx(j, k):
        """V index for B[j,k], j in 0..3, k in 0..4"""
        return j * p + k
    
    def w_idx(k, i):
        """W index for C[k,i], k in 0..4, i in 0..1 (trace convention: W[k*n+i])"""
        return k * n + i
    
    # Strategy: Split B into left columns (0,1) and right columns (2,3,4)
    # Split A into top row (i=0) and bottom row (i=1)
    # 
    # Block 1: A_top (1x4) * B_top (4x2) -> C_top (2x2)
    #   Use rank-4 decomposition for 1x4x2 (schoolbook: 1*4*2=8, but we can do better)
    #   Actually 1x4x2: rank is at most 4 (each column of C is a dot product)
    #   C[0,0] = sum_j A[0,j]*B[j,0], C[0,1] = sum_j A[0,j]*B[j,1]
    #   C[1,0] = sum_j A[1,j]*B[j,0], C[1,1] = sum_j A[1,j]*B[j,1]
    #   
    # Let me use a different decomposition. 
    # 
    # Full schoolbook is rank 2*4*5 = 40.
    # 
    # Let's try: Split B into two 4x2 blocks: B_left (cols 0,1) and B_right (cols 2,3,4)
    # B_left: 4x2, B_right: 4x3
    # 
    # For B_left (4x2): A (2x4) * B_left (4x2) -> C_left (2x2)
    #   This is 2x4x2. Rank of 2x4x2 tensor: schoolbook is 16.
    #   We can use: for each of the 4 middle indices, multiply A_col_j by B_col_j_pair
    #   Actually, let's think of it as: C[i,k] = sum_j A[i,j]*B[j,k] for k in {0,1}
    #   We can factor: for each j, we have A[i,j] for i in {0,1} and B[j,k] for k in {0,1}
    #   So for each j, the contribution is A_col_j (2x1) outer B_row_j (1x2) = 2x2 matrix
    #   That's 4 terms of rank... each term is 2*1*2 = 4 entries, but it's a rank-1 tensor in the full space?
    #   No. Each term is: u = A_col_j (2 entries in U), v = B_row_j (2 entries in V), w = identity on C (4 entries in W)
    #   Wait, that's not right either.
    #
    # Let me just use the standard approach and try to merge.
    #
    # Actually, let me use a known good decomposition:
    # 2x4x5 can be decomposed by splitting the 4 into 2+2:
    # C = A * B = A[:,:2]*B[:,:2] + A[:,2:]*B[:,2:]  -- no, that's not how matrix mult works.
    #
    # Correct split: C = sum_{j=0}^{3} A[:,j] * B[j,:]
    # = A[:,0]*B[0,:] + A[:,1]*B[1,:] + A[:,2]*B[2,:] + A[:,3]*B[3,:]
    # Each term A[:,j] (2x1) * B[j,:] (1x5) = 2x5 matrix, rank 1 in outer product sense.
    # As a tensor: u = A[:,j] (2 entries), v = B[j,:] (5 entries), w = all ones in C (10 entries)
    # Wait no. The tensor is T_{i,j,k} = delta_{i,j} * delta_{j,k}... no.
    # T[i,j,k] = 1 if this term contributes to C[i,k] through B[j,k] and A[i,j].
    # The term for fixed j: contributes to all (i,k) pairs: A[i,j]*B[j,k]
    # So u has 1 at position (i,j) for each i, v has 1 at position (j,k) for each k, w has 1 at (k,i) for each (k,i).
    # That's rank 1 per j? No, it's a sum over i and k.
    #
    # Let me just be concrete. Schoolbook:
    # For each (i,j,k): u_{i,j}=1, v_{j,k}=1, w_{k,i}=1. Total 2*4*5=40 terms.
    #
    # To reduce rank, we need to combine terms. 
    #
    # Key insight: For fixed j, the sub-tensor over (i,k) is A[:,j] * B[j,:] which is an outer product.
    # So for fixed j, we need 2*5 = 10 terms in schoolbook, but the outer product structure means
    # we can represent it with rank... well, as a 2x5 matrix, it has rank at most 2.
    # But in the tensor decomposition, each "term" is a rank-1 tensor in 3D.
    # The sub-tensor for fixed j is: T[i,j,k] = A[i,j]*B[j,k] for the contribution.
    # This is (A[:,j]) outer (B[j,:]) in the (i,k) plane, with j fixed.
    # As a 3D tensor slice at j, it's a 2x5 matrix. Its "tensor rank" in the full 3D sense...
    # Actually, the full tensor T[i,j,k] = delta_{i,j}*delta_{j,k} is not what we want.
    #
    # I'm overcomplicating this. Let me just implement the schoolbook (rank 40) and verify it works,
    # then try to find a better one through systematic search or known results.
    #
    # Known: rank(2x4x5) <= 34 (from literature on small matrix multiplication).
    # Let me try to construct a rank-34 or better decomposition.
    
    # Let me try a different approach: use the fact that 2x2x2 has rank 7 (Strassen).
    # Split the 4 middle index into two groups of 2: j in {0,1} and j in {2,3}
    # Split the 5 last index... let's keep it as 5 for now.
    #
    # C = A[:,:2]*B[:,:2] + A[:,2:]*B[:,2:]  -- this is WRONG for matrix multiplication.
    # Matrix multiplication: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # The first sum is A[:,:2] * B[:,:2] (2x2 * 2x5 = 2x5)
    # The second sum is A[:,2:] * B[:,2:] (2x2 * 2x5 = 2x5)
    # 
    # So: C = A_L * B_L + A_R * B_R
    # where A_L = A[:,:2] (2x2), B_L = B[:,:2] (4x5)... 
    # Wait, B is 4x5. B[:,:2] doesn't make sense for the middle index.
    # B[j,k] where j is the middle index (0..3) and k is the last index (0..4).
    # So B_L = B[0:2, :] (2x5) and B_R = B[2:4, :] (2x5).
    # A_L = A[:, 0:2] (2x2) and A_R = A[:, 2:4] (2x2).
    #
    # C = A_L @ B_L + A_R @ B_R
    # Each product is 2x2 * 2x5 = 2x5.
    # 
    # For 2x2 * 2x5: we can use Strassen on the 2x2 part and then multiply by 5.
    # Actually, let's think of it as: for each of the 5 columns of B, we do 2x2 * 1x5... no.
    #
    # 2x2 * 2x5: rank? Schoolbook is 2*2*5 = 20.
    # We can do: for each column k of B (k=0..4), we compute A @ B[:,k] which is 2x1.
    # That's 5 separate 2x2*1 multiplications. Each 2x2*1 has rank... 
    # C[i] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1.
    # This is a 2x2x1 tensor: T[i,j,k] with k fixed. Rank is 4 (schoolbook) but can we do better?
    # 2x2x1: T[i,j,0] = delta_{i,j}. This is just the identity, rank 2 (two terms: i=j=0 and i=j=1).
    # Wait: C[0] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # Terms: (A[0,0], B[0,k], C[0]), (A[0,1], B[1,k], C[0]), (A[1,0], B[0,k], C[1]), (A[1,1], B[1,k], C[1])
    # That's 4 terms per column k, so 5*4 = 20 for the 2x2*2x5 product. Same as schoolbook.
    #
    # But we can use Strassen-like tricks. For 2x2*2x2, Strassen gives rank 7.
    # For 2x2*2x5, we can think of it as 2x2*2x2 + 2x2*2x1 + 2x2*2x1 + 2x2*2x1
    # = 7 + 4 + 4 + 4 = 19? Not better.
    #
    # Alternatively: 2x2*2x5 can be done with rank... let me just use 20 per half.
    # Total: 20 + 20 = 40. Still schoolbook.
    #
    # Let me try yet another approach. Use the polynomial/rational approach hinted at.
    # Or just implement schoolbook and verify, then search for improvements.
    
    # For now, let me implement a clean schoolbook decomposition (rank 40) and verify correctness.
    rank = n * m * p
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(k, i)][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
