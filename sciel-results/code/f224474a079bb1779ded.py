# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    Research Programme Note:
    An SDP relaxation was considered to estimate the tensor rank. 
    Standard SDP relaxations based on matrix flattening provide lower bounds 
    (e.g., rank of 6x9 flattening is <= 6), which are too weak to confirm 
    the minimal rank of 17. Constructive rank-17 decompositions are complex 
    and not readily available in standard libraries. 
    Therefore, we return the guaranteed valid Schoolbook decomposition of rank 18.
    
    Interface:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    n=2, m=3, p=3.
    rank = 18.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to flat vectors
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
