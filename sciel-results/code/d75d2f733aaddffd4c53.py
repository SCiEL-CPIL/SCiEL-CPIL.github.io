# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # The tensor M_{2,4,5} decomposes as M_{1,4,5} \oplus M_{1,4,5}.
    # R(M_{1,4,5}) = 4*5 = 20.
    # Total rank = 40.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct block for i=0
    # T_0 = sum_{j,k} e_j \otimes f_{jk} \otimes e_k (with W index k*n+0)
    col_offset = 0
    for j in range(m):
        for k in range(p):
            U[0*m + j][col_offset] = 1
            V[j*p + k][col_offset] = 1
            W[k*n + 0][col_offset] = 1
            col_offset += 1
            
    # Construct block for i=1
    # T_1 = sum_{j,k} e_j \otimes f_{jk} \otimes e_k (with W index k*n+1)
    for j in range(m):
        for k in range(p):
            U[1*m + j][col_offset] = 1
            V[j*p + k][col_offset] = 1
            W[k*n + 1][col_offset] = 1
            col_offset += 1
            
    # Verify schoolbook
    assert _validate(U, V, W, n, m, p), "Schoolbook decomposition failed validation!"
    
    # Return the best decomposition found
    # Ensure entries are proper integers or half-integers
    def _clean_entry(v):
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        doubled = v * 2
        if abs(doubled - round(doubled)) < 1e-9:
            return round(doubled) / 2.0
        return v

    result_U = [[_clean_entry(v) for v in row] for row in U]
    result_V = [[_clean_entry(v) for v in row] for row in V]
    result_W = [[_clean_entry(v) for v in row] for row in W]
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
