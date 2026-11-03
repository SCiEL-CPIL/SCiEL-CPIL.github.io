"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # De Groote rank-23 decomposition for 3x3x3 matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k] for i,j,k in {0,1,2}
    # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    def add_de_groote_3x3x3(U, V, W, i0, j0, k0, col):
        """Add de Groote rank-23 decomposition for a 3x3x3 block."""
        # The 23 rank-1 terms for 3x3x3 matrix multiplication
        # Each term: (u_vec, v_vec, w_vec) where u_vec is 9-dim (3x3), v_vec is 9-dim, w_vec is 9-dim
        # u_vec[i*3+j], v_vec[j*3+k], w_vec[k*3+i]
        
        terms = [
            # Term 0
            ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 1
            ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # Term 2
            ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # Term 3
            ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 4
            ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # Term 5
            ([0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # Term 6
            ([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 7
            ([0,0,0, 0,0,0, 0,1,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # Term 8
            ([0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # Term 9: Strassen-like for remaining
            ([1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 10
            ([0,1,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # Term 11
            ([0,0,1, 0,0,1, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # Term 12
            ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 13
            ([0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # Term 14
            ([0,0,0, 0,0,1, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # Term 15
            ([0,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 16
            ([0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # Term 17
            ([0,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # Term 18
            ([1,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 19
            ([0,1,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # Term 20
            ([0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # Term 21
            ([0,0,0, 1,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # Term 22
            ([0,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ]
        
        # This naive approach doesn't work. Let me use the actual de Groote decomposition.
        # The de Groote decomposition for 3x3x3 has rank 23.
        # Let me use a verified set of 23 terms.
        
        # Actually, let me use a simpler approach: 3x3x3 = 3x2x3 + 3x1x3
        # 3x2x3 can be done in 14 terms (using 2x2x3 = 11, plus adjustments)
        # 3x1x3 = 9 terms (schoolbook)
        # Total: 23 terms
        
        # For 3x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1,2}, k in {0,1,2}
        # This is 3x2x3 matrix multiplication.
        # We can use: 2x2x3 (rank 11) + 1x2x3 (rank 6) = 17? No, that's too many.
        # Actually 3x2x3: we can tile as (2x2 + 1x2) x 3
        # 2x2x3: rank 11
        # 1x2x3: rank 6 (schoolbook: 1*2*3 = 6)
        # Total: 17. Still more than 23-9=14.
        
        # Let me just use schoolbook for the sub-blocks and accept rank 27 for 3x3x3.
        # 3x3x3 schoolbook: 27 terms.
        # 3x2x3 schoolbook: 18 terms.
        # 3x2x2 schoolbook: 12 terms.
        # 3x1x3 schoolbook: 9 terms.
        # 3x1x2 schoolbook: 6 terms.
        # 3x1x1 schoolbook: 3 terms.
        
        # Tiling 3x5x7 as (3x3 + 3x2) x (3x3 + 3x3 + 3x1):
        # 3x3x3: 27
        # 3x3x3: 27
        # 3x3x1: 9
        # 3x2x3: 18
        # 3x2x3: 18
        # 3x2x1: 6
        # Total: 27+27+9+18+18+6 = 105. That's worse!
        
        # Let me use Strassen for 2x2x2 blocks within the 3x3x3:
        # 3x3x3 = 4*(2x2x2) + border terms
        # 4*7 = 28 for the 2x2x2 blocks, plus border: 3*3+3*3+3*3 - 4*2*2 = 27-16 = 11
        # Total: 28+11 = 39. Worse than 27.
        
        # So for 3x3x3, schoolbook (27) is actually better than Strassen tiling.
        
        # Let me reconsider the tiling strategy.
        # 3x5x7: Use 2x2x2 Strassen where possible.
        # 3 = 2+1, 5 = 2+2+1, 7 = 2+2+2+1
        
        # Blocks:
        # (2x2 + 1x2) x (2x2 + 2x2 + 1x2) for the j dimension
        # And k dimension: 2+2+2+1
        
        # This is getting complex. Let me just use the original approach but verify it.
        
        # For now, let me implement a clean schoolbook + Strassen hybrid.
        
        # 3x5x7 = (2x2 + 1x2 + 0x2... no, 3 = 2+1)
        # Actually n=3, so i in {0,1,2}. m=5, j in {0,1,2,3,4}. p=7, k in {0,1,2,3,4,5,6}
        
        # Strategy: Use Strassen for 2x2x2 sub-blocks, schoolbook for the rest.
        # i: {0,1} can use Strassen, i=2 is schoolbook
        # j: {0,1}, {2,3} can use Strassen pairs, j=4 is schoolbook
        # k: {0,1}, {2,3}, {4,5} can use Strassen pairs, k=6 is schoolbook
        
        # For i in {0,1}, j in {0,1}, k in {0,1}: Strassen (7 terms)
        # For i in {0,1}, j in {0,1}, k in {2,3}: Strassen (7 terms)
        # For i in {0,1}, j in {0,1}, k in {4,5}: Strassen (7 terms)
        # For i in {0,1}, j in {0,1}, k in {6}: schoolbook (4 terms)
        # For i in {0,1}, j in {2,3}, k in {0,1}: Strassen (7 terms)
        # For i in {0,1}, j in {2,3}, k in {2,3}: Strassen (7 terms)
        # For i in {0,1}, j in {2,3}, k in {4,5}: Strassen (7 terms)
        # For i in {0,1}, j in {2,3}, k in {6}: schoolbook (4 terms)
        # For i in {0,1}, j in {4}, k in {0,1,2,3,4,5,6}: schoolbook (14 terms)
        # For i in {2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}: schoolbook (35 terms)
        
        # Total: 7*6 + 4*2 + 14 + 35 = 42 + 8 + 14 + 35 = 99. Same as before!
        
        # Can we do better? What if we use 3x3x3 de Groote?
        # If de Groote 3x3x3 has rank 23, then:
        # 3x3x3: 23
        # 3x2x3: ? (let's say 14)
        # 3x2x2: ? (let's say 14)
        # 3x1x3: 9
        # 3x1x2: 6
        # 3x1x1: 3
        
        # Tiling: (3x3 + 3x2) x (3x3 + 3x3 + 3x1)
        # 3x3x3: 23
        # 3x3x3: 23
        # 3x3x1: 9
        # 3x2x3: 14
        # 3x2x3: 14
        # 3x2x1: 6
        # Total: 23+23+9+14+14+6 = 89. Better than 99!
        
        # But I need actual decompositions for 3x2x3 and 3x2x2.
        # 3x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k], i in {0,1,2}, k in {0,1,2}
        # This is 3*2*3 = 18 tensor entries.
        # Can we do it in 14 terms?
        
        # 3x2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k], i in {0,1,2}, k in {0,1}
        # This is 3*2*2 = 12 tensor entries.
        # Can we do it in 14 terms? Schoolbook is 12, so 14 is worse.
        
        # Let me just use schoolbook for 3x2x2 (12 terms) and 3x2x1 (6 terms).
        # And try to find a good 3x2x3 decomposition.
        
        # 3x2x3: Let's tile as (2x2 + 1x2) x 3
        # 2x2x3: rank 11 (known result)
        # 1x2x3: rank 6 (schoolbook)
        # Total: 17. Not 14.
        
        # Or (3x2 + 0x2) x 3 = 3x2x3.
        # Can we do 3x2x3 in fewer than 17?
        # 3x2x3 has 18 entries. The tensor rank is at least 18/2 = 9 (since each term covers at most 2 entries... no, that's not right).
        
        # Let me just use the schoolbook for all sub-blocks and accept rank 105,
        # or stick with the 99-term solution.
        
        # Actually, let me try a different tiling:
        # 3x5x7 = 3x(2+2+1) x (2+2+2+1)
        # = 3x2x2 + 3x2x2 + 3x2x2 + 3x2x1
        # + 3x2x2 + 3x2x2 + 3x2x2 + 3x2x1
        # + 3x1x2 + 3x1x2 + 3x1x2 + 3x1x1
        
        # Using Strassen for 2x2x2:
        # 3x2x2: 3 * 7 = 21 per 2x2x2? No, 3x2x2 means i in {0,1,2}, j in {0,1}, k in {0,1}
        # We can only use Strassen for i in {0,1}. i=2 is schoolbook.
        # 2x2x2: 7 terms
        # 1x2x2: 4 terms
        # 3x2x2: 7+4 = 11 terms
        
        # 3x2x1: i in {0,1,2}, j in {0,1}, k in {0}
        # 2x2x1: 4 terms (schoolbook)
        # 1x2x1: 2 terms
        # 3x2x1: 4+2 = 6 terms
        
        # Tiling: (3x2 + 3x2 + 3x1) x (2+2+2+1)
        # 3x2x2: 11, 3x2x2: 11, 3x2x2: 11, 3x2x1: 6
        # 3x2x2: 11, 3x2x2: 11, 3x2x2: 11, 3x2x1: 6
        # 3x1x2: 6, 3x1x2: 6, 3x1x2: 6, 3x1x1: 3
        
        # Total: 11*3 + 6 + 11*3 + 6 + 6*3 + 3 = 33+6+33+6+18+3 = 99. Same!
        
        # It seems 99 is hard to beat with this approach.
        # Let me just return the 99-term solution.
        
        return col
    
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

    # Build the 99-term solution
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}: Strassen
    for k0 in [0, 2, 4]:
        col = add_strassen(U, V, W, 0, 0, k0, col)
    # i in {0,1}, j in {0,1}, k in {6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [0, 1], [6], col)
    # i in {0,1}, j in {2,3}, k in {0,1}, {2,3}, {4,5}: Strassen
    for k0 in [0, 2, 4]:
        col = add_strassen(U, V, W, 0, 2, k0, col)
    # i in {0,1}, j in {2,3}, k in {6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [2, 3], [6], col)
    # i in {0,1}, j in {4}, k in {0..6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [4], range(7), col)
    # i in {2}, j in {0..4}, k in {0..6}: schoolbook
    col = add_schoolbook(U, V, W, [2], range(5), range(7), col)
    
    # Verify
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                assert total == 1, f"Failed at ({i},{j},{k}): got {total}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
