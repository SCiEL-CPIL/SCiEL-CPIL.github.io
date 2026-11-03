"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add a 2x2x3 block with rank 11
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1,k0+2}
    # Using 11 terms instead of 13 (2 Strassen blocks would give 14, schoolbook gives 12)
    def add_2x2x3(U, V, W, i0, j0, k0, col):
        # 11-term decomposition for 2x2x3 matrix multiplication
        # This is based on the known result that rank(2x2x3) <= 11
        
        # Term 1: A[i0,j0]*B[j0,k0]
        U[i0 * m + j0][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
        
        # Term 2: A[i0+1,j0+1]*B[j0+1,k0]
        U[(i0+1) * m + j0+1][col] = 1
        V[(j0+1) * p + k0][col] = 1
        W[k0 * n + i0+1][col] = 1
        col += 1
        
        # Term 3: A[i0,j0+1]*B[j0+1,k0]
        U[i0 * m + j0+1][col] = 1
        V[(j0+1) * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
        
        # Term 4: A[i0+1,j0]*B[j0,k0]
        U[(i0+1) * m + j0][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i0+1][col] = 1
        col += 1
        
        # Terms 5-11: Handle k0+1 and k0+2 using Strassen-like structure
        # For the 2x2x2 block at k in {k0+1, k0+2}, use 7 terms
        # But we need to be careful about the structure
        
        # Actually, let me use a cleaner approach:
        # 2x2x3 = 2x2x1 + 2x2x2
        # 2x2x1 has rank 4 (schoolbook)
        # 2x2x2 has rank 7 (Strassen)
        # Total: 11 terms
        
        # 2x2x1 part: k = k0+1
        # C[i0,k0+1] = A[i0,j0]*B[j0,k0+1] + A[i0,j0+1]*B[j0+1,k0+1]
        # C[i0+1,k0+1] = A[i0+1,j0]*B[j0,k0+1] + A[i0+1,j0+1]*B[j0+1,k0+1]
        
        # Term 5: A[i0,j0]*B[j0,k0+1]
        U[i0 * m + j0][col] = 1
        V[j0 * p + k0+1][col] = 1
        W[(k0+1) * n + i0][col] = 1
        col += 1
        
        # Term 6: A[i0,j0+1]*B[j0+1,k0+1]
        U[i0 * m + j0+1][col] = 1
        V[(j0+1) * p + k0+1][col] = 1
        W[(k0+1) * n + i0][col] = 1
        col += 1
        
        # Term 7: A[i0+1,j0]*B[j0,k0+1]
        U[(i0+1) * m + j0][col] = 1
        V[j0 * p + k0+1][col] = 1
        W[(k0+1) * n + i0+1][col] = 1
        col += 1
        
        # Term 8: A[i0+1,j0+1]*B[j0+1,k0+1]
        U[(i0+1) * m + j0+1][col] = 1
        V[(j0+1) * p + k0+1][col] = 1
        W[(k0+1) * n + i0+1][col] = 1
        col += 1
        
        # 2x2x2 part: k in {k0+1, k0+2} using Strassen
        # But wait, we already used k0+1 in the 2x2x1 part. Let me reconsider.
        
        # Actually, the decomposition should be:
        # 2x2x3: i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1,k0+2}
        # We can split as: (2x2x1 for k=k0) + (2x2x2 for k in {k0+1,k0+2})
        # 2x2x1: 4 terms
        # 2x2x2: 7 terms
        # Total: 11 terms
        
        # But I already added 4 terms for k=k0 above (terms 1-4).
        # Now add 7 terms for the 2x2x2 block at k in {k0+1, k0+2}
        
        # Strassen for 2x2x2 at k0+1, k0+2:
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0+1][col] += 1
        V[(j0 + 1) * p + k0+2][col] += 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+2) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0+1][col] += 1
        W[(k0+1) * n + i0 + 1][col] += 1
        W[(k0+2) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0+2][col] += 1
        V[(j0 + 1) * p + k0+2][col] -= 1
        W[(k0+2) * n + i0][col] += 1
        W[(k0+2) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0+1][col] += 1
        V[j0 * p + k0+1][col] -= 1
        W[(k0+1) * n + i0][col] += 1
        W[(k0+1) * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0+2][col] += 1
        W[(k0+2) * n + i0][col] += 1
        W[(k0+1) * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0+1][col] += 1
        V[j0 * p + k0+2][col] += 1
        W[(k0+2) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0+1][col] += 1
        V[(j0 + 1) * p + k0+2][col] += 1
        W[(k0+1) * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Build a better decomposition using 2x2x3 blocks where possible
    # 2x2x3 can be done in 11 terms (4 schoolbook for one k + 7 Strassen for two k's)
    # This is better than 3x3x2 Strassen (21) + 3x3x1 schoolbook (9) = 30 for 3x3x3
    
    # Strategy: Partition the 3x5x7 tensor into blocks
    # i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    
    # Use 2x2x3 blocks for i in {0,1}, j in {0,1}, k in {0,1,2}
    # Then handle the rest with Strassen/schoolbook
    
    # Total terms estimate:
    # 2x2x3 for (i={0,1}, j={0,1}, k={0,1,2}): 11 terms
    # 2x2x3 for (i={0,1}, j={0,1}, k={3,4,5}): 11 terms
    # 2x2x1 for (i={0,1}, j={0,1}, k={6}): 4 terms
    # 2x2x3 for (i={0,1}, j={2,3}, k={0,1,2}): 11 terms
    # 2x2x3 for (i={0,1}, j={2,3}, k={3,4,5}): 11 terms
    # 2x2x1 for (i={0,1}, j={2,3}, k={6}): 4 terms
    # 2x2x7 for (i={0,1}, j={4}, k={0..6}): This is 2x2x7, use 3 Strassen + 4 schoolbook = 25 terms
    # 1x5x7 for (i={2}, j={0..4}, k={0..6}): 35 schoolbook terms
    # Total: 11+11+4+11+11+4+25+35 = 112 terms. That's worse than 99!
    
    # Let me reconsider. The current 99-term solution:
    # Part 1: j in {0,1}
    #   3 Strassen 2x2x2 blocks for k in {0,1}, {2,3}, {4,5}: 21 terms
    #   4 schoolbook for i in {0,1}, j in {0,1}, k={6}: 4 terms
    #   14 schoolbook for i={2}, j in {0,1}, k in {0..6}: 14 terms
    #   Subtotal: 39 terms
    # Part 2: j in {2,3,4}
    #   3 Strassen 2x2x2 blocks for i in {0,1}, j in {2,3}, k in {0,1}, {2,3}, {4,5}: 21 terms
    #   4 schoolbook for i in {0,1}, j in {2,3}, k={6}: 4 terms
    #   14 schoolbook for i in {0,1}, j={4}, k in {0..6}: 14 terms
    #   21 schoolbook for i={2}, j in {2,3,4}, k in {0..6}: 21 terms
    #   Subtotal: 60 terms
    # Total: 39 + 60 = 99 terms
    
    # Can we do better? Let's try using 2x2x3 blocks.
    # For i in {0,1}, j in {0,1}:
    #   k in {0,1,2}: 2x2x3 = 11 terms
    #   k in {3,4,5}: 2x2x3 = 11 terms
    #   k in {6}: 2x2x1 = 4 terms
    #   Subtotal: 26 terms (vs 25 in current: 3*7+4=25)
    # That's worse!
    
    # So the current approach is already near-optimal for this partition.
    
    # Let me try a different partition:
    # Use 3x2x2 Strassen blocks for i in {0,1,2}, j in {0,1}, k in {0,1}, {2,3}, {4,5}
    # 3x2x2 can be done in 3*7 = 21 terms (3 Strassen 2x2x2 blocks)
    # Then 3x2x1 for k=6: 6 terms
    # Then 3x3x7 for j in {2,3,4}: This is the hard part
    
    # Actually, let me just return the current 99-term solution. It's hard to improve
    # without a fundamentally better decomposition.
    
    rank_baseline = 99
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    # Part 1: j in {0, 1}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 0, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [0, 1], range(7), col)
    
    # Part 2: j in {2, 3, 4}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 2, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [2, 3], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [4], range(7), col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [2, 3, 4], range(7), col)
    
    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
