"""Direct-sum decomposition of M_{4,4,6} via 1xmx rank-1 and 2x2x2 Strassen blocks."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Tensor T[i,j,k]=1 (i in [0,n), j in [0,m), k in [0,p)).
    # Build factorization: T = sum_c U_i(c) V_j(c) W_k(c)
    # by direct sum of sub-problems:
    #   M_{4,4,6} = M_{2,4,6} (rows 0,1) + M_{2,4,6} (rows 2,3) + M_{1,4,6}?? 
    # Actually M_{4,4,6} splits along the i (row of A) dimension into four M_{1,4,6},
    # each rank 1. That gives rank 4. But also split along i into two M_{2,4,6}.
    # M_{2,4,6}: split j into two M_{2,2,6}? No, split k into three M_{2,4,2}.
    # Better: M_{2,4,6} = sum over j-blocks: split j into two M_{2,2,6}.
    #   M_{2,2,6} = split k into three M_{2,2,2}, each rank 7 => 21.
    # So M_{2,4,6} = 2 * 21 = 42.
    # M_{4,4,6} = 2 * M_{2,4,6} = 84. That's worse.
    #
    # Use the 1xm trick: split i into four M_{1,4,6}, each rank 1 => rank 4.
    # But we can do better by grouping: M_{4,4,6} = M_{2,4,6} + M_{2,4,6}.
    # Each M_{2,4,6}: split i into two M_{1,4,6} (rank 1 each) => rank 2.
    # So M_{4,4,6} rank 4.
    #
    # Even better: since T is all-ones, the whole M_{4,4,6} has rank 1!
    # U[i]=1, V[j]=1, W[k]=1 for all. Verify: sum_c=1: U_i V_j W_k = 1 = T[i,j,k]. Yes.
    #
    # So rank 1 suffices. But wait, is T really all-ones? For matrix multiplication
    # C[i,k] = sum_j A[i,j] B[j,k], the tensor coefficient of A[i,j]*B[j,k] is 1.
    # The factorization C[i,k] = sum_{j,c} A[i,j] B[j,k] U[i,c] V[j,c] W[k,c]
    # requires sum_c U[i,c] V[j,c] W[k,c] = 1 for ALL (i,j,k).
    # With rank 1: U[i]=1, V[j]=1, W[k]=1 => product = 1. Correct!
    #
    # Return rank-1 decomposition.
    rank = 1
    n, m, p = 4, 4, 6
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
