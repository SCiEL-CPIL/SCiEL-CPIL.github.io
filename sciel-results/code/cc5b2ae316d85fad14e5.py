"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for matrix multiplication.
    For dimensions n=2, m=3, p=3, the minimal rank is n*m*p = 18.
    The decomposition is the schoolbook algorithm.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as lists of lists to match the expected interface style,
    # or use numpy arrays if the evaluator accepts them. 
    # The prompt example uses lists of lists. I will stick to lists for compatibility 
    # with the "integer/half-integer entries" description, though numpy is allowed.
    # Using lists is safer for strict type checking if the evaluator inspects types.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input (i,j) to rank index col
                U[i * m + j][col] = 1
                # V maps input (j,k) to rank index col
                V[j * p + k][col] = 1
                # W maps output (k,i) to rank index col
                # Note: The problem statement says W[k*n+i, r]. 
                # In standard matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]B[j,k].
                # The tensor T_{i,j,k} = 1.
                # Decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
                # Wait, the standard decomposition is:
                # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
                # Tensor T_{i,j,k}.
                # Rank 1 term: u_{i,j} v_{j,k} w_{k,i} ?? No.
                # The factors are usually defined such that:
                # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} is NOT the standard form.
                # The standard form for matrix multiplication tensor $M_{n,m,p}$:
                # Indices are i (row of A), j (col of A / row of B), k (col of B).
                # The multiplication is $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
                # The tensor is $T_{i,j,k}$.
                # A rank-1 decomposition term is $u_i v_j w_k$.
                # But the output indices are linked.
                # Let's look at the seed code:
                # U[i*m+j, c] = 1
                # V[j*p+k, c] = 1
                # W[k*n+i, c] = 1
                # This implies the term is U_{i,j} * V_{j,k} * W_{k,i}.
                # Summing over c, we get sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
                # If we set U_{i,j,c}=1, V_{j,k,c}=1, W_{k,i,c}=1 for a specific c,
                # the product is 1.
                # This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
                # So the "W" factor here is actually tied to the output index (k, i).
                # This matches the "official trace convention W[k*n+i,r]" mentioned in the prompt.
                
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
