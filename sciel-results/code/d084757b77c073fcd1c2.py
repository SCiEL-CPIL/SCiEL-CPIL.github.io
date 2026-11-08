"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Try to construct a rank-reduced decomposition using block structure
    # Use Strassen for 2x2 blocks, Winograd for 3x3, schoolbook for rest
    
    # Approach: Partition the 5x5 matrices into blocks
    # A = [[A11, A12], [A21, A22]] where sizes are 2x2, 2x3, 3x2, 3x3
    # B = [[B11, B12], [B21, B22]] where sizes are 2x2, 2x3, 3x2, 3x3
    
    # C = A*B
    # C11 = A11*B11 + A12*B21  (2x2 result)
    # C12 = A11*B12 + A12*B22  (2x3 result)
    # C21 = A21*B11 + A22*B21  (3x2 result)
    # C22 = A21*B12 + A22*B22  (3x3 result)
    
    # For each product term, use appropriate rank:
    # 2x2 * 2x2 -> use Strassen rank 7
    # 2x2 * 3x2 -> schoolbook 12 (or we can think of it as 2*(2x2->2x1) )
    # 3x2 * 2x2 -> schoolbook 12
    # 3x2 * 3x2 -> schoolbook 18
    # 2x3 * 2x3 -> schoolbook 18
    # 3x3 * 3x3 -> use Winograd rank 23
    # 2x3 * 3x2 -> schoolbook 18
    # 3x2 * 2x3 -> schoolbook 18
    
    # Let me compute total rank with this approach:
    # C11: A11(2x2)*B11(2x2) -> Strassen 7 + A12(2x3)*B21(3x2) -> schoolbook 18 = 25
    # C12: A11(2x2)*B12(2x3) -> schoolbook 12 + A12(2x3)*B22(3x3) -> schoolbook 18 = 30
    # C21: A21(3x2)*B11(2x2) -> schoolbook 12 + A22(3x3)*B21(3x2) -> schoolbook 18 = 30
    # C22: A21(3x2)*B12(2x3) -> schoolbook 18 + A22(3x3)*B22(3x3) -> Winograd 23 = 41
    # Total: 25 + 30 + 30 + 41 = 126
    
    # That's worse than schoolbook! Let me reconsider.
    
    # Better approach: Use Strassen on the full problem by embedding
    # Or use a known rank-120 construction
    
    # Actually, let me just use the schoolbook rank-125 decomposition as baseline
    # and try to improve via iterative refinement or known constructions
    
    # For now, implement schoolbook which is guaranteed correct
    rank = n * m * p  # 125
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
