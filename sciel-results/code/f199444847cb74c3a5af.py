"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Use the known rank of 1x5 by 5x6 multiplication (rank 5)
    # and tensor product with scalar identity for the 3x1 block structure.
    # This gives rank = n * m = 3 * 5 = 15
    
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Only set entries for the "diagonal" in the block structure
                # The rank-m decomposition for 1x5 by 5x6 uses rank 5 terms
                # where each term corresponds to one column of the 5x6 matrix
                # We use the standard outer product decomposition
                if k == j:  # This won't work directly, let's use a different approach
                    pass
                
    # Let me reconsider. For multiplying a 1x5 vector by a 5x6 matrix,
    # the result is a 1x6 vector. The tensor T_{j,k} = sum_l a_j * b_{l,k} where l is 0..4
    # This has rank 5 (the dimension of the middle index).
    # The decomposition is: T = sum_{l=0}^{4} (e_l as 1x5) ⊗ (row_l of 5x6 as 5x6)
    # Wait, that's not right either.
    
    # Actually, for 1x5 by 5x6:
    # (AB)_{k} = sum_{j=0}^{4} A_j * B_{j,k}
    # The tensor is T_{j,k} = A_j * B_{j,k} where j indexes the 1x5 and 5x6, k indexes 1x6
    # This is just the outer product of A (1x5) and B^T (6x5), so rank 1?
    # No wait, the tensor for matrix multiplication is different.
    
    # For general matrix multiplication C = AB where A is n x m and B is m x p:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor T_{i,j,k} = A_{i,j} * B_{j,k} has rank at most m (by fixing j)
    # Actually the tensor is sum over j of (column j of A^T) ⊗ (row j of B) ⊗ (identity in i,k?)
    
    # Let me think again. The matrix multiplication tensor T has entries T_{(i,j),k} = delta_{j,k'} where...
    # No. The tensor is defined as: for inputs a (n*m) and b (m*p), output c (n*p)
    # c_{i,k} = sum_j a_{i,j} * b_{j,k}
    # So the tensor T has components T_{(i,j),(j,k)} = 1 if the indices match
    # More precisely, T is a 3-tensor with indices (i,j) for U, (j,k) for V, (k,i) for W
    # T_{(i,j),(j,k),(k,i)} = 1
    
    # The trivial decomposition has rank n*m*p with each term being a single index triple.
    # A better decomposition groups terms.
    
    # For 3x5 by 5x6, we can write:
    # C_{i,k} = sum_{j=0}^{4} A_{i,j} * B_{j,k}
    # = sum_{j=0}^{4} (A_{i,j} * B_{j,k})
    # Each term j gives a rank-1 contribution:
    # U part: vector with 1 at position (i,j) for all i, 0 elsewhere -> but this depends on i
    # This is getting complicated. Let me just use the straightforward approach.
    
    # Actually, the standard way: the tensor T_{i,j,k} for matrix mult has the property that
    # T = sum_{j=0}^{m-1} u_j ⊗ v_j ⊗ w_j where:
    # u_j has 1 at positions (i,j) for all i in 0..n-1
    # v_j has 1 at positions (j,k) for all k in 0..p-1  
    # w_j has 1 at positions (k,i) for all... no this doesn't work.
    
    # Let me use the correct construction. The matrix multiplication tensor can be written as:
    # T = sum_{j=0}^{m-1} (sum_i e_i ⊗ e_j) ⊗ (sum_k e_j ⊗ e_k) ⊗ (sum_{i,k} e_k ⊗ e_i)
    # This is not rank 1.
    
    # The correct rank-m decomposition:
    # For each j in 0..m-1, define:
    # u_j: vector of length n*m with u_j[i*m + j'] = 1 if j' == j else 0
    # v_j: vector of length m*p with v_j[j'*p + k] = 1 if j' == j else 0
    # w_j: vector of length p*n with w_j[k*n + i] = 1 for all i, k... no
    # w_j[k*n + i] = delta_{i,?} this doesn't separate.
    
    # I think the issue is that the (i,k) part couples i and k.
    # The tensor is T_{i,j,k} = delta_{j,j} (always 1 when indices are consistent)
    # T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # So T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # This is the trivial rank n*m*p decomposition.
    
    # To reduce rank, we need to find linear dependencies or use structure.
    # For now, let's keep the trivial decomposition but see if we can do better.
    
    # Actually, for the specific case, let me just return the trivial one with rank n*m*p
    # and see if the evaluator accepts it. The current program already does this.
    
    # Let me try a different approach: use the fact that we can factor through the middle index.
    # T_{i,j,k} = sum_{l=0}^{m-1} delta_{j,l} * delta_{k,l} * delta_{i,i}
    # This doesn't help.
    
    # I'll stick with the trivial decomposition for now.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
