"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    Strategy: Apply Strassen's divide-and-conquer by padding to 4x8 x 8x8,
    apply Strassen recursively, then project back to 3x5 x 5x7.
    
    We pad A (3x5) to 4x8 by adding zero rows/cols.
    We pad B (5x7) to 8x8 by adding zero rows/cols.
    The padded product C_pad (4x8) contains our desired C (3x7) in the top-left.
    
    Strassen on 4x4 x 4x4 (with inner dim 8 handled via further decomposition):
    Actually, let's use a cleaner approach: decompose using Kronecker products.
    
    3x5x7: We can write this as a sub-tensor of 4x8x8.
    4x8x8: split 4=2+2, 8=4+4, 8=4+4
    Apply Strassen: 7 multiplications of 2x4x4 tensors.
    Each 2x4x4: split 2=1+1, 4=2+2, 4=2+2
    Apply Strassen: 7 multiplications of 1x2x2 tensors.
    Each 1x2x2: schoolbook rank = 1*2*2 = 4
    
    Total rank: 7 * 7 * 4 = 196. That's worse than 105!
    
    Let me try a different decomposition.
    
    Actually, for 4x8x8:
    - Split into 2x2 blocks: each block is 2x4 for A, 4x4 for B, 2x4 for C
    - Strassen: 7 products of 2x4x4
    - Each 2x4x4 has schoolbook rank 2*4*4 = 32
    - Total: 7*32 = 224. Still worse.
    
    The issue is that Strassen helps when the dimensions are powers of 2 and square.
    For our rectangular case, it may not help.
    
    Let me just use the schoolbook decomposition which gives rank 105.
    """
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
