"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix 
    by a 4x6 matrix using a structured approach.
    
    We use the fact that matrix multiplication can be decomposed using the 
    outer product structure. For an n x m times m x p multiplication, we can
    potentially reduce the rank below n*m*p by exploiting linearity.
    
    For small dimensions, we can try to find a lower rank decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Start with the trivial rank = n*m*p decomposition
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Populate the decomposition factors
    # Index mapping:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i (Trace convention)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][idx] = 1.0
                V[v_idx][idx] = 1.0
                W[w_idx][idx] = 1.0
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
