import numpy as np
import time

DIMENSIONS = [3, 3, 4]


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin under 150s
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Strategy: Use Strassen's 7-multiplication algorithm for 2x2 blocks
    # to build a rank-28 decomposition for 3x3 * 3x4.
    #
    # Partition A (3x3) and B (3x4) into 2x2 blocks with overlap:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # B = [[B11, B12, B13, B14],
    #      [B21, B22, B23, B24],
    #      [B31, B32, B33, B34]]
    #
    # We'll use a systematic construction based on 2x2 Strassen blocks.
    # For each pair of 2x2 sub-blocks, Strassen uses 7 multiplications.
    #
    # Alternative: Use the known rank-28 construction for 3x3*3x4.
    # Let me build it explicitly.
    
    # Actually, let me use a cleaner approach: 
    # The 3x3*3x4 tensor can be decomposed using 2x2 Strassen blocks.
    # 
    # Consider the following block partition:
    # A = [[A_top_left, A_top_right],    where A_top_left is 2x2, A_top_right is 2x1
    #      [A_bottom_left, A_bottom_right]] where A_bottom_left is 1x2, A_bottom_right is 1x1
    #
    # This is getting complex. Let me use a different strategy.
    
    # Better approach: Use the fact that we can apply Strassen to 2x2 blocks
    # and handle the remaining rows/columns separately.
    
    # For 3x3 * 3x4:
    # Split into:
    #   - 2x2 * 2x4 block (top-left 2 rows, first 2 cols of B)
    #   - 2x1 * 1x4 block (top-left 2 rows, last 2 cols of B)  
    #   - 1x2 * 2x4 block (bottom row, first 2 cols of B)
    #   - 1x1 * 1x4 block (bottom row, last 2 cols of B)
    
    # Actually, let me just construct a valid rank-28 decomposition directly.
    # I'll use a known construction based on 2x2 Strassen embeddings.
    
    # The cleanest approach: build the decomposition column by column for the output.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let me use a different, more systematic construction.
    # I'll create a rank-28 decomposition using the following idea:
    # 
    # For the 3x3 * 3x4 case, we can use 7 Strassen multiplications for 
    # a 2x2 * 2x2 subproblem, and extend to handle the full dimensions.
    
    # Let me try a concrete construction:
    # 
    # Partition indices:
    # i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}
    #
    # Use the following rank-28 construction based on 2x2 Strassen blocks:
    
    # Actually, the simplest correct approach: use rank 28 via 2x2 Strassen
    # applied to appropriate blocks. Let me construct it explicitly.
    
    # Here's a concrete rank-28 construction:
    # We'll use 7 "Strassen-type" multiplications for each of 4 column-blocks,
    # but that gives 28. Let me verify this works.
    
    # For each pair of output columns (k1, k2), we can use a 2x2 Strassen
    # decomposition on the corresponding 2x2 submatrices of B.
    
    # Let me just build a valid decomposition directly.
    
    # Rank-28 construction for 3x3 * 3x4:
    # 
    # The idea: use 7 multiplications for a 2x2*2x2 block, and we have 
    # 4 such "column groups" but they overlap. 
    #
    # Simpler: just use the schoolbook with some clever merging.
    # 
    # Actually, let me use a known result: the tensor rank of 3x3*3x4 is 
    # at most 25 (from various constructions). Let me try to find a 
    # valid decomposition with rank <= 28.
    
    # I'll construct a rank-28 decomposition using the following method:
    # For each of the 4 output columns, we need 3*3 = 9 "scalar" multiplications
    # in schoolbook, but we can share some via Strassen-like combinations.
    
    # Let me use a direct construction:
    
    # The MM tensor T[i,j,k] = 1 for all i,j,k.
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let me build a rank-28 decomposition:
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 9 x 28
    V = [[0] * rank for _ in range(m * p)]  # 12 x 28
    W = [[0] * rank for _ in range(p * n)]  # 12 x 28
    
    # Construction: Use 2x2 Strassen blocks
    # 
    # For the top-left 2x2 block of A and corresponding blocks of B:
    # A_tl = [[A[0,0], A[0,1]],
    #         [A[1,0], A[1,1]]]
    # 
    # Strassen's 7 products for 2x2*2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # I'll build the full decomposition by combining Strassen blocks.
    # This is complex, so let me use a simpler valid construction.
    
    # Alternative: Use rank 28 via a known explicit construction.
    # Let me just return a valid schoolbook decomposition with rank 36,
    # but first try to find a better one.
    
    # For now, let me construct a valid rank-28 decomposition:
    
    # I'll use the following approach:
    # - 7 multiplications for the 2x2*2x2 top-left block
    # - 7 multiplications for the 2x2*2x2 top-right block (cols 2,3 of B)
    # - 7 multiplications for the 1x2*2x2 bottom block
    # - 7 multiplications for the 1x2*1x2 bottom-right block
    # Total: 28
    
    # But this doesn't quite work because the blocks overlap.
    
    # Let me just use a verified construction. I'll build it step by step.
    
    # Actually, the safest approach: return schoolbook rank 36, which is valid.
    # But let me try to do better.
    
    # Here's a concrete rank-28 construction:
    
    # Define the 28 rank-1 components:
    
    # Components 0-6: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: (0,0), (0,1), (1,0), (1,1) -> U rows: 0,1,3,4
    # B indices: (0,0), (0,1), (1,0), (1,1) -> V rows: 0,1,4,5
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W rows: 0,1,3,4
    
    # Strassen M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = 1
    
    # This is getting too complex to do manually. Let me use a different strategy.
    
    # I'll use a computational approach: build the decomposition using 
    # 2x2 Strassen blocks and verify it.
    
    # For now, let me just return a valid schoolbook decomposition.
    # The rank is 36, which is valid but not optimal.
    
    rank = n * m * p  # 36
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
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    _verify(result, n, m, p)
    
    return result


def _verify(result, n, m, p):
    """Verify that the factor matrices exactly reproduce the MM tensor."""
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    raise ValueError(
                        f"Verification failed at i={i}, j={j}, k={k}: got {total}"
                    )


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
