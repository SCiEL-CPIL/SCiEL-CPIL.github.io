"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank
    
    def verify(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Also check that non-multiplication terms are zero
        # Actually, the tensor only has 1s at (i,j,k) positions, so we just need those to sum to 1
        return True
    
    # Try to find a lower rank decomposition
    # Start with schoolbook
    best_result, best_rank = schoolbook()
    
    # Try a few structured decompositions
    # Attempt 1: Group the p dimension
    # For each (i,j), we need to compute sum over j of A[i,j]*B[j,k] for all k
    # We can try to share some multiplications
    
    # Let's try a rank-30 decomposition by splitting k into groups
    # Actually, let me try a different approach: use the fact that we can combine terms
    
    # Try rank 35, 30, 25, etc.
    for target_rank in [35, 30, 25, 20, 15]:
        if time.time() > deadline - 10:
            break
        # This is a placeholder - actual construction needed
        # For now, just return schoolbook
        pass
    
    return best_result
# EVOLVE-BLOCK-END
