import numpy as np

def construct():
    n, m, p = 2, 3, 5
    
    # Block decomposition:
    #   A (2x3) * B (3x5) -> C (2x5)
    #   Split j (middle dim) into {0,1} and {2}
    #   Split k (columns of B) into {0,1}, {2,3}, {4}
    #
    # Block (i=0,1; j=0,1; k=0,1): 2x2x2 -> Strassen rank 7
    # Block (i=0,1; j=0,1; k=2,3): 2x2x2 -> Strassen rank 7
    # Block (i=0,1; j=0,1; k=4):   2x2x1 -> schoolbook rank 4? No, can do rank 2
    #   Actually for 2x2x1: C[i,k=4] = A[i,0]*B[0,4] + A[i,1]*B[1,4]
    #   This is rank 2 (two terms).
    # Block (i=0,1; j=2; k=0,1):   2x1x2 -> schoolbook rank 4? 
    #   C[i,k] = A[i,2]*B[2,k] for k=0,1. Rank 4.
    # Block (i=0,1; j=2; k=2,3):   2x1x2 -> schoolbook rank 4.
    # Block (i=0,1; j=2; k=4):     2x1x1 -> rank 2.
    #
    # Total: 7 + 7 + 2 + 4 + 4 + 2 = 26
    #
    # Wait, let me reconsider. The tensor is T[i,j,k] = 1 for all i,j,k.
    # We need to decompose this into sum of rank-1 terms.
    #
    # Alternative approach: use the fact that 2x3x5 can be decomposed
    # by treating it as combinations of smaller known-rank tensors.
    #
    # Let me try a different split:
    #   Split i into singletons: i=0 and i=1
    #   For each i, we have 1x3x5 which is rank 15 (schoolbook).
    #   That gives 30 total. Not helpful.
    #
    # Let me stick with the block approach but be more careful.
    #
    # For 2x2x1 (i in {0,1}, j in {0,1}, k={4}):
    #   C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4]
    #   Term 1: U has 1 at (0,0) and (1,0) [i=0,j=0 and i=1,j=0], V has 1 at (0,4), W has 1 at (4,0) and (4,1)
    #   Wait no. Rank-1 term: U[i,j] * V[j,k] * W[k,i]
    #   For k=4: we need C[i,4] = sum_j A[i,j]*B[j,4]
    #   Rank 2: 
    #     r=0: U[0,0]=1, V[0,4]=1, W[4,0]=1
    #     r=1: U[1,0]=1, V[0,4]=1, W[4,1]=1
    #     But also need j=1:
    #     r=2: U[0,1]=1, V[1,4]=1, W[4,0]=1
    #     r=3: U[1,1]=1, V[1,4]=1, W[4,1]=1
    #   So it's rank 4 for 2x2x1. Hmm.
    #
    # Actually 2x2x1 tensor: T[i,j,4] = 1 for i,j in {0,1}.
    # This is the outer product of a 2x2 all-ones matrix with a 1-vector.
    # Rank = rank of the 2x2 matrix = 1? No.
    # T[i,j,k] where k is fixed at 4. As a matrix in (i,j) it's all ones, rank 1.
    # But it's a 3-tensor. The border rank...
    # Actually for fixed k, T[:, :, 4] is a 2x2 matrix of ones, rank 1.
    # But the tensor has only one slice. So it should be rank 1?
    # No: rank-1 term is u_i * v_j * w_k. If w_4 = 1 and w_other = 0, then
    # u_i * v_j * 1 must equal 1 for all i,j. So u_i = 1, v_j = 1 for all i,j.
    # That's one rank-1 term! So 2x2x1 has rank 1.
    #
    # Wait, but the tensor only has entries at k=4. So T[i,j,4]=1, T[i,j,k!=4]=0.
    # A rank-1 tensor u_i v_j w_k has all entries nonzero if u,v,w are nonzero.
    # To have T[i,j,k]!=0 only when k=4, we need w_k=0 for k!=4. So w = (0,0,0,0,1).
    # Then u_i v_j * 1 = 1 for all i,j in {0,1}. So u_0=u_1=1, v_0=v_1=1.
    # Yes! Rank 1.
    #
    # Similarly 2x1x2: T[i,2,k]=1 for i in {0,1}, k in {0,1}.
    # As a tensor with j fixed at 2: T[i,2,k]. Rank-1: u_i * v_2 * w_k.
    # v_2=1, u_i=1 for all i, w_k=1 for k in {0,1}, w_k=0 otherwise.
    # So rank 1!
    #
    # And 2x1x1: T[i,2,4]=1. Rank 1.
    #
    # So revised total:
    #   2x2x2 (j=0,1; k=0,1): Strassen rank 7
    #   2x2x2 (j=0,1; k=2,3): Strassen rank 7
    #   2x2x1 (j=0,1; k=4): rank 1
    #   2x1x2 (j=2; k=0,1): rank 1
    #   2x1x2 (j=2; k=2,3): rank 1
    #   2x1x1 (j=2; k=4): rank 1
    # Total: 7+7+1+1+1+1 = 18
    #
    # Hmm, but wait. These rank-1 terms overlap in the index space.
    # The full tensor T[i,j,k]=1 for ALL i,j,k. We're decomposing it as sum of blocks.
    # Each block is a sub-tensor with support on a subset of (i,j,k).
    # The sum of all blocks must give T[i,j,k]=1 everywhere.
    #
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): contributes 1 to T[i,j,k] for those indices.
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): contributes 1 to T[i,j,k] for those indices.
    # Block (i in {0,1}, j in {0,1}, k=4): contributes 1 to T[i,j,4].
    # Block (i in {0,1}, j=2, k in {0,1}): contributes 1 to T[i,2,k].
    # Block (i in {0,1}, j=2, k in {2,3}): contributes 1 to T[i,2,k].
    # Block (i in {0,1}, j=2, k=4): contributes 1 to T[i,2,4].
    #
    # Together these cover all (i,j,k) with value 1. Good.
    #
    # But can we do better than Strassen for the 2x2x2 blocks?
    # The 2x2x2 matrix multiplication tensor has rank 7 (Strassen is optimal for 2x2x2? 
    # Actually it's known that the 2x2x2 tensor has rank 7, and Strassen achieves this.)
    #
    # So total rank 18. Score = 30/18 = 1.667.
    #
    # Can we do even better? What about mixing the blocks?
    # For instance, the 2x2x1 block (j=0,1; k=4) and the 2x1x1 block (j=2; k=4)
    # could potentially be merged. The combined block is j in {0,1,2}, k=4, i in {0,1}.
    # That's a 2x3x1 tensor: T[i,j,4]=1 for all i,j.
    # Rank of 2x3x1: T[i,j,4]=1, T[i,j,k!=4]=0.
    # Rank-1: u_i * v_j * w_k with w_4=1, w_other=0. u_i=1, v_j=1. Rank 1.
    # So we can merge those two into one rank-1 term.
    #
    # Similarly, the two 2x1x2 blocks (j=2; k=0,1) and (j=2; k=2,3) can be merged
    # into j=2, k in {0,1,2,3}, i in {0,1}. That's 2x1x4 tensor.
    # T[i,2,k]=1 for k in {0,1,2,3}. Rank 1 (u_i=1, v_2=1, w_k=1 for k in {0,1,2,3}).
    #
    # Revised total:
    #   2x2x2 (j=0,1; k=0,1): rank 7
    #   2x2x2 (j=0,1; k=2,3): rank 7
    #   2x3x1 (j=0,1,2; k=4): rank 1
    #   2x1x4 (j=2; k=0,1,2,3): rank 1
    # Total: 7+7+1+1 = 16
    # Score = 30/16 = 1.875
    #
    # Can we do better? What if we merge more?
    # The 2x2x2 blocks... can we combine them with the other blocks?
    # 
    # What about 2x2x4 (j=0,1; k=0,1,2,3)? That's two 2x2x2 blocks.
    # Rank of 2x2x4: it's the tensor for 2x2 matrix times 2x4 matrix.
    # C[i,k] = sum_j A[i,j] B[j,k] for k in {0,1,2,3}.
    # This is equivalent to doing 2x2x2 multiplication twice, but sharing the A factors.
    # Using Strassen-like ideas: the 2x2x4 tensor has rank... 
    # It can be viewed as 4 copies of the 2x2x1 output, but the A part is shared.
    # Actually, it's the same as computing 2x2 times 2x4.
    # One way: use Strassen for the first 2 columns (rank 7) and Strassen for the next 2 (rank 7), total 14.
    # But can we do better? The 2x2x4 tensor... 
    # Actually, I think the rank is 14 (two independent 2x2x2's).
    # 
    # Alternatively, think of it as: we have 4 output columns. For Strassen, we compute 7 products of 2x2 submatrices of A and 2x2 submatrices of B. If B is 2x4, we'd need to form 7 2x4-like products... this gets complicated.
    #
    # Let me just go with rank 16 and verify it.
    
    rank = 16
    U = np.zeros((n * m, rank))  # (6, 16)
    V = np.zeros((m * p, rank))  # (15, 16)
    W = np.zeros((p * n, rank))  # (10, 16)
    
    col = 0
    
    def add_strassen_2x2x2(i_base, j_base, k_base):
        """Add Strassen terms for 2x2x2 block at given base indices."""
        nonlocal col
        # i_base, i_base+1 for i; j_base, j_base+1 for j; k_base, k_base+1 for k
        i0, i1 = i_base, i_base + 1
        j0, j1 = j_base, j_base + 1
        k0, k1 = k_base, k_base + 1
        
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] = 1; U[i1*m+j1, col] = 1
        V[j0*p+k0, col] = 1; V[j1*p+k1, col] = 1
        W[k0*n+i0, col] = 1; W[k1*n+i1, col] = 1
        col += 1
        
        # M2 = (A[i1,j0]+A[i1,j1]) B[j0,k0]
        U[i1*m+j0, col] = 1; U[i1*m+j1, col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+i1, col] = 1; W[k1*n+i1, col] = -1
        col += 1
        
        # M3 = A[i0,j0] (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] = 1
        V[j0*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i0, col] = 1; W[k1*n+i1, col] = 1
        col += 1
        
        # M4 = A[i1,j1] (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] = 1
        V[j1*p+k0, col] = 1; V[j0*p+k0, col] = -1
        W[k0*n+i0, col] = 1; W[k0*n+i1, col] = 1
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j1]) B[j1,k1]
        U[i0*m+j0, col] = 1; U[i0*m+j1, col] = 1
        V[j1*p+k1, col] = 1
        W[k0*n+i0, col] = -1; W[k1*n+i0, col] = 1
        col += 1
        
        # M6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] = 1; U[i0*m+j0, col] = -1
        V[j0*p+k0, col] = 1; V[j0*p+k1, col] = 1
        W[k1*n+i1, col] = 1
        col += 1
        
        # M7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k0, col] = 1; V[j1*p+k1, col] = 1
        W[k0*n+i0, col] = 1
        col += 1
    
    def add_rank1_block(i_list, j_list, k_list):
        """Add a rank-1 term for a block where T[i,j,k]=1 for i in i_list, j in j_list, k in k_list."""
        nonlocal col
        for i in i_list:
            for j in j_list:
                U[i*m+j, col] = 1
        for j in j_list:
            for k in k_list:
                V[j*p+k, col] = 1
        for k in k_list:
            for i in i_list:
                W[k*n+i, col] = 1
        col += 1
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen rank 7
    add_strassen_2x2x2(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen rank 7
    add_strassen_2x2x2(0, 0, 2)
    
    # Block 3: i in {0,1}, j in {0,1,2}, k={4} -> rank 1
    add_rank1_block([0, 1], [0, 1, 2], [4])
    
    # Block 4: i in {0,1}, j={2}, k in {0,1,2,3} -> rank 1
    add_rank1_block([0, 1], [2], [0, 1, 2, 3])
    
    # Verify
    Ur = U.T.reshape(rank, n, m)
    Vr = V.T.reshape(rank, m, p)
    Wr = W.T.reshape(rank, p, n)
    tensor = np.einsum('rij,rjk,rki->ijk', Ur, Vr, Wr)
    
    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback to rank 30 schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
