<<<<<<< SEARCH
def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                lhs = 0.0
                for r in range(rank):
                    lhs += U[a][r] * V[b][r] * W[c][r]
                if abs(lhs - expected) > 1e-9:
                    return False
    return True
=======
def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact using integer arithmetic."""
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0
                lhs = 0
                for r in range(rank):
                    lhs += U[a][r] * V[b][r] * W[c][r]
                if lhs != expected:
                    return False
    return True
>>>>>>> REPLACE
