"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    The schoolbook decomposition has rank n*m*p = 3*3*6 = 54.
    We attempt to find a lower rank decomposition using known results and heuristics.
    
    For 3x3 * 3x6 matrix multiplication, the tensor is M_{3,3,6}.
    Known results:
    - The rank of M_{2,2,2} is 7 (Strassen).
    - The rank of M_{3,3,3} is at most 23 (Landsberg).
    - For M_{3,3,6}, we can think of it as 6 copies of M_{3,3,1} or 3 copies of M_{3,3,2}.
    
    A simple approach: Use Strassen for 2x2 blocks and handle the remaining 1x1 or 1x2 parts.
    
    However, finding an explicit rank-minimal decomposition for M_{3,3,6} is non-trivial.
    Let's try to construct a decomposition with rank less than 54.
    
    One approach: Decompose the 3x6 matrix as [3x2 | 3x2 | 3x2] and use Strassen-like methods.
    
    Actually, let's try a different approach. We can use the fact that M_{3,3,6} can be embedded
    in M_{4,4,8} or similar, but that's not helpful directly.
    
    Let's try to use a greedy or heuristic approach to find a lower rank decomposition.
    Since we need an exact decomposition with integer/half-integer entries, we can try to
    find a decomposition with rank 23 or less, but this is very hard.
    
    For now, let's return the schoolbook decomposition of rank 54, but with a more
    efficient construction. We can also try to reduce the rank by combining terms.
    
    Actually, let's try to use the following idea:
    - The 3x3 * 3x6 multiplication can be seen as 6 independent 3x3 * 3x1 multiplications.
    - Each 3x3 * 3x1 multiplication has rank at most 9 (schoolbook), but can be lower.
    
    However, the tensor is not a direct sum of 6 copies of M_{3,3,1}, because the input
    dimensions are shared.
    
    Let's try a different strategy: use a known rank-23 decomposition for M_{3,3,3} and
    extend it to M_{3,3,6} by doubling the output dimension.
    
    If we have a rank-23 decomposition for M_{3,3,3}, we can use it twice to get a rank-46
    decomposition for M_{3,3,6}. This is better than 54.
    
    So, let's try to find a rank-23 decomposition for M_{3,3,3} and use it twice.
    
    The rank-23 decomposition for M_{3,3,3} is known but complex. Let's use a simpler
    approach: use Strassen for 2x2 subblocks and handle the 1x1 parts separately.
    
    For 3x3 * 3x3, we can partition into:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    B = [[B11, B12], [B21, B22]] similarly.
    
    Then C = AB can be computed using:
    - Strassen for the 2x2 * 2x2 part (rank 7)
    - Additional terms for the cross terms and 1x1 part
    
    This gives a rank of 7 + 4 + 4 + 1 = 16? No, that's not right.
    
    Actually, the correct way is to use the "Strassen-like" decomposition for 3x3.
    The best known rank for M_{3,3,3} is 23, but constructing it explicitly is hard.
    
    For simplicity, let's return the schoolbook decomposition of rank 54, but with a
    deadline guard to ensure we finish within 120s.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, leaving 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Check if we're still within deadline
    if time.time() < deadline:
        # Try to find a lower rank decomposition
        # For now, we'll just return the schoolbook decomposition
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
