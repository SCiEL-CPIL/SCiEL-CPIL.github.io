"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
Uses Strassen's algorithm for 2x2 matrix multiplication applied to block structures.
The 2x4 * 4x8 multiplication is decomposed into two 2x2 * 2x8 multiplications.
Each 2x2 * 2x8 multiplication is decomposed into four 2x2 * 2x2 multiplications.
Each 2x2 * 2x2 multiplication uses Strassen's rank-7 decomposition.
Total rank = 2 * 4 * 7 = 56."""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56

    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 16 rows

    # Strassen factors for 2x2 * 2x2
    # u: (X00, X01, X10, X11)
    # v: (Y00, Y01, Y10, Y11)
    # w: (Z00, Z01, Z10, Z11)
    strassen_factors = [
        # M1
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
        # M3
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        # M4
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
        # M5
        ((1, 0, 1, 0), (0, 0, 0, 1), (1, 1, 0, 0)),
        # M6
        ((-1, 0, 0, 1), (1, 0, 1, 0), (0, 0, 0, 1)),
        # M7
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0))
    ]

    col = 0
    # Two A-blocks: A1 (cols 0,1) and A2 (cols 2,3)
    # For each A-block, we have 4 column groups in B (size 2 each)
    for a_block in range(2):
        # A indices offset
        a_j_offset = a_block * 2
        
        for g in range(4):
            # B column indices for this group
            k0 = g * 2
            k1 = g * 2 + 1
            
            # B row indices offset
            # B1 uses rows 0,1. B2 uses rows 2,3.
            # Wait, B is 4x8.
            # B1 is B[0:2, :]. Rows 0,1.
            # B2 is B[2:4, :]. Rows 2,3.
            b_j_offset = a_block * 2
            
            for u, v, w in strassen_factors:
                # Update U
                # u indices correspond to X00, X01, X10, X11
                # X is 2x2 submatrix of A.
                # X00 -> A[i=0, j=a_j_offset]
                # X01 -> A[i=0, j=a_j_offset+1]
                # X10 -> A[i=1, j=a_j_offset]
                # X11 -> A[i=1, j=a_j_offset+1]
                
                # Map u[0] to U[0*m + a_j_offset]
                # Map u[1] to U[0*m + a_j_offset+1]
                # Map u[2] to U[1*m + a_j_offset]
                # Map u[3] to U[1*m + a_j_offset+1]
                
                if u[0] != 0: U[0 * m + a_j_offset][col] += u[0]
                if u[1] != 0: U[0 * m + a_j_offset + 1][col] += u[1]
                if u[2] != 0: U[1 * m + a_j_offset][col] += u[2]
                if u[3] != 0: U[1 * m + a_j_offset + 1][col] += u[3]
                
                # Update V
                # v indices correspond to Y00, Y01, Y10, Y11
                # Y is 2x2 submatrix of B (for specific k columns)
                # Y00 -> B[j=b_j_offset, k=k0]
                # Y01 -> B[j=b_j_offset, k=k1]
                # Y10 -> B[j=b_j_offset+1, k=k0]
                # Y11 -> B[j=b_j_offset+1, k=k1]
                
                # Map v[0] to V[b_j_offset*p + k0]
                # Map v[1] to V[b_j_offset*p + k1]
                # Map v[2] to V[(b_j_offset+1)*p + k0]
                # Map v[3] to V[(b_j_offset+1)*p + k1]
                
                if v[0] != 0: V[b_j_offset * p + k0][col] += v[0]
                if v[1] != 0: V[b_j_offset * p + k1][col] += v[1]
                if v[2] != 0: V[(b_j_offset + 1) * p + k0][col] += v[2]
                if v[3] != 0: V[(b_j_offset + 1) * p + k1][col] += v[3]
                
                # Update W
                # w indices correspond to Z00, Z01, Z10, Z11
                # Z is 2x2 submatrix of C
                # Z00 -> C[i=0, k=k0] -> W[k0*n + 0]
                # Z01 -> C[i=0, k=k1] -> W[k1*n + 0]
                # Z10 -> C[i=1, k=k0] -> W[k0*n + 1]
                # Z11 -> C[i=1, k=k1] -> W[k1*n + 1]
                
                # Map w[0] to W[k0*n + 0]
                # Map w[1] to W[k1*n + 0]
                # Map w[2] to W[k0*n + 1]
                # Map w[3] to W[k1*n + 1]
                
                if w[0] != 0: W[k0 * n + 0][col] += w[0]
                if w[1] != 0: W[k1 * n + 0][col] += w[1]
                if w[2] != 0: W[k0 * n + 1][col] += w[2]
                if w[3] != 0: W[k1 * n + 1][col] += w[3]
                
                col += 1

    assert col == rank, f"column count {col} != rank {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
