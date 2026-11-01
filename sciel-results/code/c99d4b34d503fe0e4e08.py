"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-24 decomposition for 2x2 by 2x6 matrix multiplication
    using the naive algorithm.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W):
    """
    Verifies the decomposition: U * V^T * W^T should equal the multiplication tensor.
    T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    Here indices are flattened:
    i -> (row_A, col_A)
    j -> (row_B, col_B)
    k -> (row_C, col_C) -> actually (col_B, row_C) based on W[k*n+i]
    
    The tensor T is defined such that C_{i, k} = sum_j A_{i, j} B_{j, k}.
    In the decomposition:
    sum_c U[row_A, col_A, c] * V[row_B, col_B, c] * W[col_B, row_C, c] = delta(row_C == row_A) * delta(col_B == col_B) * delta(col_A == row_B)?
    
    Actually, the standard check is:
    For all i (row A), j (col A / row B), k (col B):
    The coefficient of A_{i,j} B_{j,k} in the product should be 1 if it contributes to C_{i,k}.
    
    The decomposition computes:
    C_{i, k} = sum_c (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k}) * W_{k,i,c} ?
    No, the structure is:
    U is (n*m) x rank. Rows are (i,j).
    V is (m*p) x rank. Rows are (j,k).
    W is (p*n) x rank. Rows are (k,i).
    
    The computed value for C_{i, k} is:
    sum_c W_{k,i,c} * (sum_j U_{i,j,c} A_{i,j}) * (sum_l V_{l,k,c} B_{l,k})
    
    We want this to equal sum_j A_{i,j} B_{j,k}.
    
    So for a specific (i, j, k), the term A_{i,j} B_{j,k} should appear with coefficient 1.
    This requires that for each c, the combination of U, V, W factors out the A and B terms correctly.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Build the tensor T from U, V, W
    # T[i, j, k] is the coefficient of A_{i,j} B_{j,k} in the result.
    # The result C_{i, k} = sum_c W_{k,i,c} * (sum_j U_{i,j,c} A_{i,j}) * (sum_l V_{l,k,c} B_{l,k})
    # Expanding:
    # C_{i, k} = sum_j sum_l (sum_c W_{k,i,c} U_{i,j,c} V_{l,k,c}) A_{i,j} B_{l,k}
    # We want C_{i, k} = sum_j A_{i,j} B_{j,k}.
    # So we need:
    # sum_c W_{k,i,c} U_{i,j,c} V_{l,k,c} = 1 if j == l, else 0.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    expected = 1 if j == l else 0
                    calculated = 0
                    for c in range(rank):
                        # U index: i*m + j
                        # V index: l*p + k
                        # W index: k*n + i
                        u_val = U[i*m + j][c]
                        v_val = V[l*p + k][c]
                        w_val = W[k*n + i][c]
                        calculated += u_val * v_val * w_val
                    if calculated != expected:
                        return False
    return True

# EVOLVE-BLOCK-END
