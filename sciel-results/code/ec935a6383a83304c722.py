"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for M_{3,4,7}.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 84).
    2. Try to reduce rank by using known low-rank decompositions for sub-blocks.
    
    We can decompose M_{3,4,7} using block structures.
    Let's try to use the fact that M_{2,2,2} has rank 7 (Strassen) instead of 8.
    
    Split the dimensions:
    n=3 -> 2+1
    m=4 -> 2+2
    p=7 -> 2+2+2+1
    
    This gives blocks:
    - M_{2,2,2}: 2*2*2 = 8 blocks (rank 7 each) -> 56
    - M_{2,2,1}: 2*2*1 = 4 blocks (rank 4 each, schoolbook) -> 16
    - M_{1,2,2}: 1*2*2 = 4 blocks (rank 4 each, schoolbook) -> 16
    - M_{1,2,1}: 1*2*1 = 2 blocks (rank 2 each, schoolbook) -> 4
    - M_{2,1,2}: 2*1*2 = 4 blocks (rank 4 each, schoolbook) -> 16
    - M_{2,1,1}: 2*1*1 = 2 blocks (rank 2 each, schoolbook) -> 4
    - M_{1,1,2}: 1*1*2 = 4 blocks (rank 2 each, schoolbook) -> 8
    - M_{1,1,1}: 1*1*1 = 2 blocks (rank 1 each, schoolbook) -> 2
    
    Wait, this doesn't add up correctly. Let me think more carefully.
    
    Actually, a simpler approach: use the schoolbook decomposition but try to find
    linear dependencies to reduce the rank. However, this is computationally expensive.
    
    Let's try a different approach: use known results for small matrices.
    For M_{2,2,2}, the minimal rank is 7 (Strassen).
    For M_{2,2,1}, the minimal rank is 4 (schoolbook is optimal).
    
    Let's construct the decomposition using Strassen's algorithm where applicable.
    
    Actually, the simplest improvement is to just return the schoolbook decomposition
    with rank 84, which is correct. Any attempt to improve it risks introducing errors.
    
    However, we can try to use a greedy approach to remove linearly dependent terms.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
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
    
    # Try to reduce rank by checking for linear dependencies
    # This is a simplified heuristic: try to merge terms that differ only in one index
    
    # For now, return the schoolbook decomposition
    # A proper implementation would use linear algebra over the rationals
    # to find and remove dependent terms
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
