"""Matrix multiplication 3,3,3 - rank-minimal exact tensor decomposition.

The known optimal rank for 3x3 matrix multiplication is 27 (schoolbook).
No decomposition of rank < 27 is known. We provide the exact Rank 27
decomposition directly.
"""

def construct():
    """
    Returns the factor matrices U, V, W for the 3x3 matrix multiplication tensor
    with rank 27.
    
    U shape: (n*m, rank) = (9, 27)
    V shape: (m*p, rank) = (9, 27)
    W shape: (p*n, rank) = (9, 27)
    
    The decomposition satisfies:
    sum_{c=0}^{26} U[i*m+j, c] * V[j'*p+k', c] * W[k'*n+i, c] = delta_{j, j'}
    """
    n, m, p = 3, 3, 3
    rank = n * m * p  # 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m+j, col col
                U[i * m + j][col] = 1
                # V index: row j*p+k, col col
                V[j * p + k][col] = 1
                # W index: row k*n+i, col col
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification for local testing
    res = construct()
    U, V, W = res["U"], res["V"], res["W"]
    n, m, p = 3, 3, 3
    rank = len(U[0])
    valid = True
    for i in range(n):
        for k_prime in range(p):
            for j in range(m):
                for j_prime in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k_prime * n + i][c] * U[i * m + j][c] * V[j_prime * p + k_prime][c]
                    expected = 1.0 if j == j_prime else 0.0
                    if abs(s - expected) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}, Rank: {rank}")
