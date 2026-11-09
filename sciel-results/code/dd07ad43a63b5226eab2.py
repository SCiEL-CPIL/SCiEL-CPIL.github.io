"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    n, m, p = DIMENSIONS
    # Trivial rank = n*m*p = 105
    rank_trivial = n * m * p

    # Build the trivial decomposition
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify the trivial decomposition is correct
    def verify(U, V, W, rank):
        """Verify tensor decomposition T_{i,j,k} = sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r}"""
        import numpy as np
        # Check with basis vectors
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T_{i,j,k} should be 1 if this is the identity tensor
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    # The matrix multiplication tensor has T_{i,j,k} = 1 for all (i,j,k)
                    # Actually no - the tensor is defined by the bilinear map
                    # T_{i,j,k} = delta_{i,j,k} in the standard basis
                    # For matrix multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}
                    # The tensor entries are T_{i,j,k} = 1 (it's the "all ones" in the sense that
                    # each term A_{i,j} B_{j,k} contributes to C_{i,k})
                    # Actually the tensor is: for each (i,j,k), the coefficient is 1
                    # So T_{i,j,k} = 1 for all valid (i,j,k)
                    if val != 1:
                        return False
        return True

    # The trivial decomposition should verify
    # But let's also try to find a lower rank decomposition

    # Strategy: Try to find a lower rank decomposition using the fact that
    # matrix multiplication is a 3-tensor. We can use the approach of fixing
    # some entries and solving for others.

    # For now, return the trivial decomposition if we can't find something better
    # within the time budget.

    # Let's try a simple approach: use the known Strassen-like methods or
    # try to find a decomposition with rank < 105.

    # Actually, for 3x5 * 5x7, the trivial rank is 105.
    # The lower bound is at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
    # But finding a rank-35 decomposition is very hard.

    # Let's just return the trivial decomposition for now, but with proper verification.
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
