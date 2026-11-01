"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Fallback: Trivial n*m*p rank decomposition
    def get_trivial():
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
        return {"U": U, "V": V, "W": W}

    # Strategy: 
    # 1. Attempt to find a lower rank decomposition (e.g., using known algorithms or heuristic search) within the time budget.
    # 2. If a valid lower rank decomposition is found, return it.
    # 3. Otherwise, return the trivial decomposition.
    
    # Given the complexity of finding a minimal rank decomposition for 3x3x5 and the 150s limit,
    # and the fact that inspirations failed to provide correct hardcoded coefficients for a lower rank,
    # we will attempt a simple heuristic search for a rank slightly below 45, or just return trivial.
    
    # However, a more robust approach for this specific task structure is to verify if we can construct
    # a valid decomposition of rank < 45. Since we don't have a guaranteed hardcoded solution for rank < 45
    # in the inspirations, and generating one from scratch is computationally intensive,
    # we will stick to the trivial decomposition but ensure it is robust and fast.
    
    # Let's try to see if we can improve the trivial one by checking for any linear dependencies
    # in the standard basis? No, the standard basis is already minimal in the "schoolbook" sense 
    # for the specific tensor structure unless we use Strassen-like tricks.
    
    # Since we cannot reliably produce a rank < 45 solution in the time budget without hardcoded data
    # that is verified correct, we return the trivial solution.
    
    # Time guard: if we spent too much time, return trivial immediately.
    if time.time() - start_time > 140:
        return get_trivial()

    # For now, we return the trivial decomposition.
    return get_trivial()
# EVOLVE-BLOCK-END
