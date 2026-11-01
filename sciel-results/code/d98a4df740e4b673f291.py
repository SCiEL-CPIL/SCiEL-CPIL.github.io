import time
import numpy as np

DIMENSIONS = [2, 3, 9]
_DEADLINE = 140.0  # wall-clock seconds allowed by evaluator, with safety margin

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """
    Construct the schoolbook (trivial) decomposition of rank n*m*p.
    Each term isolates one (i,j,k) tuple.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}


def _validate_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[(i,j),(j',k),(k',i')] = 1 iff i=i' and j=j' and k=k', else 0.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that the (i,j,k) entry sums to 1
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
                # Check that no other (i',j',k') with same (j,k) but different i' gets spurious contribution
                # Actually, we need to check all entries, not just the diagonal ones
    # More thorough check: verify all entries
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            # T[(i,j),(j2,k),(k2,i2)] should be 1 iff i==i2 and j==j2 and k==k2
                            expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                            s = 0
                            for r in range(rank):
                                s += U[i * m + j][r] * V[j2 * p + k][r] * W[k2 * n + i2][r]
                            if s != expected:
                                return False
    return True


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Build the schoolbook decomposition
    result = _build_schoolbook(n, m, p)
    
    # Validate
    if _validate_decomposition(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback (should not happen)
    return {"U": [[0]], "V": [[0]], "W": [[0]]}
# EVOLVE-BLOCK-END
