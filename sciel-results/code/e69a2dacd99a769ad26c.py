def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    
    # Check for leakage (zeros)
    # We check a few specific inconsistent index triples
    # a = i*m + j, b = j*p + k, c = k*n + i
    # Inconsistent if j_a != j_b or k_b != k_c or i_c != i_a
    
    def check_zero(a, b, c):
        actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        return abs(actual) <= tol

    if n >= 2 and m >= 2 and p >= 2:
        # Original check: a=0 (0,0), b=p (1,0), c=0 (0,0) -> j mismatch
        if not check_zero(0, p, 0):
            return False
        
        # New check: a=0 (0,0), b=1 (0,1), c=0 (0,0) -> k mismatch
        # a=0 -> i=0, j=0
        # b=1 -> j=0, k=1
        # c=0 -> k=0, i=0
        # k=1 vs k=0
        if not check_zero(0, 1, 0):
            return False
            
        # New check: a=1 (0,1), b=0 (0,0), c=0 (0,0) -> j mismatch
        # a=1 -> i=0, j=1
        # b=0 -> j=0, k=0
        # c=0 -> k=0, i=0
        # j=1 vs j=0
        if not check_zero(1, 0, 0):
            return False

    return True
