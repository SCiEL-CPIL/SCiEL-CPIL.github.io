"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 6x6 = 2x2 blocks of 3x3.
    # Rank = Rank(2x2 Strassen) * Rank(3x3 Schoolbook) = 7 * 27 = 189.
    
    # Strassen factors for 2x2
    # u2[i2, s], v2[j2, s], w2[k2, s]
    u2 = [
        [1, 0, 1, 0, 1, -1, 1], # i2=0
        [1, 1, 0, 1, 0,  1, -1] # i2=1
    ]
    v2 = [
        [1, 1, 1, -1, 1, 1, 1], # j2=0
        [1, 0, -1, 1, 1, 1, 1]  # j2=1
    ]
    w2 = [
        [1, 1, 1, 0, 0, 1, 0], # k2=0
        [1, 0, 0, 1, 1, 0, 1]  # k2=1
    ]
    
    # Schoolbook factors for 3x3
    # u3[i3, r3], v3[j3, r3], w3[k3, r3]
    # r3 = i3*3 + j3
    u3 = [[0]*27 for _ in range(3)]
    v3 = [[0]*27 for _ in range(3)]
    w3 = [[0]*27 for _ in range(3)]
    for i3 in range(3):
        for j3 in range(3):
            r3 = i3 * 3 + j3
            u3[i3][r3] = 1
            v3[j3][r3] = 1
            # k3 is determined by the term in the sum, but in schoolbook
            # the term is A[i3, j3] * B[j3, k3].
            # The W factor depends on k3.
            # In the tensor decomposition sum over r, the term is U[i]V[j]W[k].
            # For schoolbook, r corresponds to (i, j, k) triple?
            # No, schoolbook rank is n*m*p.
            # Term (i, j, k) contributes to C[i, k].
            # U[i, r] = 1, V[j, r] = 1, W[k, r] = 1.
            # So r iterates over all i, j, k.
            # Here for 3x3, r3 iterates over 27 combinations of (i3, j3, k3).
            # Let's redefine r3 to be the index of the triple (i3, j3, k3).
            pass

    # Redo Schoolbook 3x3 factors properly
    # Rank 27. Index r3 = i3*9 + j3*3 + k3
    u3 = [[0]*27 for _ in range(3)]
    v3 = [[0]*27 for _ in range(3)]
    w3 = [[0]*27 for _ in range(3)]
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                r3 = i3 * 9 + j3 * 3 + k3
                u3[i3][r3] = 1
                v3[j3][r3] = 1
                w3[k3][r3] = 1

    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for r3 in range(27):
            r = s * 27 + r3
            for i2 in range(2):
                for i3 in range(3):
                    i_global = 2 * i2 + i3
                    U[i_global][r] = u2[i2][s] * u3[i3][r3]
            for j2 in range(2):
                for j3 in range(3):
                    j_global = 2 * j2 + j3
                    V[j_global][r] = v2[j2][s] * v3[j3][r3]
            for k2 in range(2):
                for k3 in range(3):
                    k_global = 2 * k2 + k3
                    # W index is k_global * n + i_global
                    # But i_global varies. The W factor in the tensor product
                    # is w2[k2, s] * w3[k3, r3].
                    # This value is assigned to the row corresponding to (k_global, i_global)?
                    # No, the W matrix in the decomposition has rows indexed by the output position.
                    # The tensor is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
                    # Here the 'k' in W[k, r] is the column index of the output matrix.
                    # But the task says W has shape (p*n, rank) and index is k*n+i.
                    # This implies the row index of W is the flattened output matrix index.
                    # The factor value for a specific output element (i_global, k_global)
                    # is w2[k2, s] * w3[k3, r3].
                    # So we must set W[k_global * n + i_global, r] = w2[k2, s] * w3[k3, r3]
                    # BUT this value depends on i_global as well?
                    # In the tensor product, the W factor for the combined tensor
                    # is the product of the W factors of the sub-tensors.
                    # W_combined[i_global, k_global, r] = w2[k2, s] * w3[k3, r3].
                    # Note that w2 depends only on k2 (and s), w3 depends only on k3 (and r3).
                    # It does NOT depend on i2 or i3.
                    # So for a fixed r, and fixed k_global (fixed k2, k3),
                    # the value w2[k2, s] * w3[k3, r3] is the same for all i_global?
                    # Yes, because the W factor in matrix multiplication tensor
                    # T[i, j, k] depends on k (output col) and the decomposition index.
                    # It does not depend on i (output row) directly in the factor value,
                    # except that U depends on i.
                    # So W[row_index, r] should be the same for all i_global 
                    # that share the same k_global?
                    # Let's check the schoolbook case.
                    # W[k, r] = 1 if r corresponds to (i, j, k).
                    # Here r encodes i, j, k.
                    # So W[k, r] is 1 only for the specific i encoded in r?
                    # No, in the decomposition C[i, k] = sum_j A[i, j] B[j, k],
                    # the term for (i, j, k) is A[i, j] B[j, k].
                    # U[i, r] = 1, V[j, r] = 1, W[k, r] = 1.
                    # So W[k, r] is 1 regardless of i?
                    # Yes, for a fixed r (which fixes i, j, k), W[k, r] is 1.
                    # But if r fixes a different i', W[k, r] is still 1?
                    # No, r is a single index.
                    # In schoolbook, r = i*9 + j*3 + k.
                    # U[i, r] = 1.
                    # V[j, r] = 1.
                    # W[k, r] = 1.
                    # So for a fixed r, W[k, r] is 1 if k matches the k in r.
                    # It does not depend on i.
                    
                    # So in our tensor product:
                    # W_combined[row_idx, r] = w2[k2, s] * w3[k3, r3].
                    # This value is assigned to the row `k_global * n + i_global`.
                    # Since the value does not depend on i_global, we can set it
                    # for all i_global corresponding to the same k_global.
                    
                    val = w2[k2][s] * w3[k3][r3]
                    for i2 in range(2):
                        for i3 in range(3):
                            i_global = 2 * i2 + i3
                            idx = k_global * n + i_global
                            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
