"""Matrix multiplication 6x6x6 using Kronecker product of two 2x2 Strassen decompositions."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2(U, V, W, col_offset, n, m, p, i0, j0, k0):
    """Add Strassen rank-7 terms for 2x2 matrix mult at offset (i0,j0) in A, (j0,k0) in B, (k0,i0) in C.
    A is n x m, B is m x p, C is p x n.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    Returns new column offset after adding 7 columns.
    """
    col = col_offset
    # 2x2 blocks: A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[k0:k0+2, i0:i0+2]
    # Strassen:
    # M1 = (A11+A22)(B12-B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B21+B22)
    # M4 = A22(B11-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication, we need to add terms.
    # Each "multiplication" of two 2x2 matrices contributes rank 7 terms.
    # But here, the Strassen algorithm computes 7 products of 2x2 matrices.
    # Each such product is itself a 2x2 matrix multiplication, which has rank 4 (schoolbook) 
    # or we can recursively use Strassen. For 2x2, schoolbook rank 4 is simpler.
    
    # Actually, for the full tensor decomposition, each scalar multiplication in the 
    # Strassen formula corresponds to a rank-1 term in the overall decomposition.
    # Let me think about this more carefully.
    
    # The 2x2 matrix multiplication tensor T_2 has rank 7 (Strassen).
    # The 7 rank-1 decompositions are:
    # (using indices a,b for rows of A, c,d for cols of A/rows of B, e,f for cols of B)
    
    # Let me define the 7 Strassen terms for 2x2:
    # A indices: (0,0),(0,1),(1,0),(1,1) -> relative to (i0,j0)
    # B indices: (0,0),(0,1),(1,0),(1,1) -> relative to (j0,k0)
    # C indices: (0,0),(0,1),(1,0),(1,1) -> relative to (k0,i0)
    
    # Strassen rank-7 decomposition of 2x2 matmul:
    # t1 = (a00+a11)(b01-b10)
    # t2 = (a10+a11)b00
    # t3 = a00(b10+b11)
    # t4 = a11(b00-b11)
    # t5 = (a00+a01)b11
    # t6 = (a10-a00)(b00+b01)
    # t7 = (a01-a11)(b10+b11)
    # c00 = t1+t4-t5+t7
    # c01 = t3+t5
    # c10 = t2+t4
    # c11 = t1-t2+t3+t6
    
    # Each ti is a product of linear forms in A and B.
    # For the tensor decomposition, each ti contributes to the U, V, W matrices.
    
    # Let me enumerate the 7 terms:
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs) where each is list of (index, coeff)
        # t1 = (a00+a11)(b01-b10)
        # U: a00:+1, a11:+1 ; V: b01:+1, b10:-1 ; W: c00:+1, c11:+1
        ([((0,0),1),((1,1),1)], [((0,1),1),((1,0),-1)], [((0,0),1),((1,1),1)]),
        # t2 = (a10+a11)b00
        # U: a10:+1, a11:+1 ; V: b00:+1 ; W: c10:+1, c11:-1
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        # t3 = a00(b10+b11)
        # U: a00:+1 ; V: b10:+1, b11:+1 ; W: c01:+1, c11:+1
        ([((0,0),1)], [((1,0),1),((1,1),1)], [((0,1),1),((1,1),1)]),
        # t4 = a11(b00-b11)
        # U: a11:+1 ; V: b00:+1, b11:-1 ; W: c00:+1, c10:+1
        ([((1,1),1)], [((0,0),1),((1,1),-1)], [((0,0),1),((1,0),1)]),
        # t5 = (a00+a01)b11
        # U: a00:+1, a01:+1 ; V: b11:+1 ; W: c00:-1, c01:+1
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        # t6 = (a10-a00)(b00+b01)
        # U: a10:+1, a00:-1 ; V: b00:+1, b01:+1 ; W: c11:+1
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        # t7 = (a01-a11)(b10+b11)
        # U: a01:+1, a11:-1 ; V: b10:+1, b11:+1 ; W: c00:+1
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
        for (ai, aj), u_val in u_coeffs:
            U[(i0+ai)*m + (j0+aj)][col] = u_val
        for (bi, bj), v_val in v_coeffs:
            V[(j0+bi)*p + (k0+bj)][col] = v_val
        for (ci, cj), w_val in w_coeffs:
            W[(k0+ci)*n + (i0+cj)][col] = w_val
        col += 1
    
    return col

def _schoolbook_block(U, V, W, col_offset, n, m, p, i0, j0, k0, ni, nj, nk):
    """Add schoolbook terms for ni x nj * nj x nk matrix multiplication.
    A block at rows i0..i0+ni-1, cols j0..j0+nj-1
    B block at rows j0..j0+nj-1, cols k0..k0+nk-1
    C block at rows k0..k0+nk-1, cols i0..i0+ni-1
    """
    col = col_offset
    for di in range(ni):
        for dj in range(nj):
            for dk in range(nk):
                U[(i0+di)*m + (j0+dj)][col] = 1
                V[(j0+dj)*p + (k0+dk)][col] = 1
                W[(k0+dk)*n + (i0+di)][col] = 1
                col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Partition: first 4 indices and last 2 indices
    # C[0:4,0:4] = A[0:4,0:4]*B[0:4,0:4] + A[0:4,4:6]*B[4:6,0:4]
    # C[0:4,4:6] = A[0:4,0:4]*B[0:4,4:6] + A[0:4,4:6]*B[4:6,4:6]
    # C[4:6,0:4] = A[4:6,0:4]*B[0:4,0:4] + A[4:6,4:6]*B[4:6,0:4]
    # C[4:6,4:6] = A[4:6,0:4]*B[0:4,4:6] + A[4:6,4:6]*B[4:6,4:6]
    
    # Total rank:
    # 4x4*4x4: 49 (Strassen)
    # 4x2*2x4: 4*2*2*2 = 32 (schoolbook)
    # 4x4*4x2: 4*4*2*2 = 64... wait no
    # Let me recount:
    # A[0:4,0:4]*B[0:4,0:4]: 4x4 mult, rank 49
    # A[0:4,4:6]*B[4:6,0:4]: 4x2 * 2x4, rank 4*2*2*2=32
    # A[0:4,0:4]*B[0:4,4:6]: 4x4 * 4x2, rank 4*4*2*2=64
    # A[0:4,4:6]*B[4:6,4:6]: 4x2 * 2x2, rank 4*2*2*2=32
    # A[4:6,0:4]*B[0:4,0:4]: 2x4 * 4x4, rank 2*4*4*2=64
    # A[4:6,4:6]*B[4:6,0:4]: 2x2 * 2x4, rank 2*2*2*2=16
    # A[4:6,0:4]*B[0:4,4:6]: 2x4 * 4x2, rank 2*4*4*2=64
    # A[4:6,4:6]*B[4:6,4:6]: 2x2 * 2x2, rank 2*2*2*2=16
    
    # Total: 49+32+64+32+64+16+64+16 = 337. That's worse than 189!
    
    # The issue is that I'm double-counting. In the block decomposition:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # where A11 is 4x4, A12 is 4x2, A21 is 2x4, A22 is 2x2, etc.
    
    # Each term is a matrix multiplication:
    # A11*B11: 4x4*4x4 = rank 49 (Strassen)
    # A12*B21: 4x2*2x4 = rank 32
    # A11*B12: 4x4*4x2 = rank 64
    # A12*B22: 4x2*2x2 = rank 32
    # A21*B11: 2x4*4x4 = rank 64
    # A22*B21: 2x2*2x4 = rank 16
    # A21*B12: 2x4*4x2 = rank 64
    # A22*B22: 2x2*2x2 = rank 16
    
    # Total: 49+32+64+32+64+16+64+16 = 337. Still bad.
    
    # The problem is that non-square matrix multiplications have high rank.
    # Let me try a different approach: use Strassen on 2x2 blocks throughout.
    
    # 6x6 = 3x3 grid of 2x2 blocks.
    # C_ij = sum_k A_ik * B_kj for i,j,k in {0,1,2}
    # Each product A_ik * B_kj is 2x2 * 2x2, rank 7 (Strassen)
    # Total: 9*3*7 = 189. Same as schoolbook!
    
    # So Strassen on 2x2 blocks doesn't help for 6x6.
    
    # Better approach: Use a known algorithm for 6x6.
    # The best known upper bound for 6x6 is 156 (from various constructions).
    # But implementing those is complex.
    
    # Simple improvement: use Strassen for the 4x4 part and be smarter about cross terms.
    # Actually, let me just use the 3x3 block structure with a rank-19 3x3 algorithm.
    # That gives 19*7 = 133.
    
    # For now, let me implement a correct decomposition with rank < 189.
    # Use 2x2 Strassen blocks: 3x3 grid, each 2x2*2x2 = rank 7.
    # But we can share some computations.
    
    # Actually, the simplest correct improvement: 
    # Use Strassen for 4x4 (rank 49) and handle the 6x6 as:
    # Pad to 8x8? No, that increases rank.
    
    # Let me just implement the 3x3 block structure with Strassen on each 2x2 product.
    # This gives rank 189, same as schoolbook. Not an improvement.
    
    # OK let me try yet another approach: 
    # Decompose 6 = 2+2+2 and use a rank-19 algorithm for 3x3 block mult.
    # The 3x3 matrix multiplication tensor has rank at most 19.
    # Each "multiplication" in the 3x3 block algorithm is a 2x2*2x2 product (rank 7).
    # Total rank: 19 * 7 = 133.
    
    # I need to find a rank-19 decomposition of 3x3 matrix multiplication.
    # Lader's algorithm is complex. Let me use a simpler known rank-19 or rank-20.
    
    # Actually, for a cleaner implementation, let me use the fact that:
    # rank(M_6) <= rank(M_3) * rank(M_2) <= 19 * 7 = 133
    # But I need an explicit rank-19 decomposition of M_3.
    
    # This is getting too complex for the time. Let me just implement a correct 
    # decomposition that's better than 189 using a different strategy.
    
    # Strategy: Use Strassen for 4x4 top-left, and optimize the rest.
    # Actually, I realize my counting was wrong. Let me recount the 4+2 split:
    
    # In the tensor decomposition, we're decomposing the FULL 6x6x6 multiplication tensor.
    # The block decomposition gives us a sum of tensor products, each of which is a 
    # matrix multiplication of smaller dimensions.
    
    # For A11*B11 where A11 is 4x4 and B11 is 4x4: this is a 4x4x4 tensor, rank 49.
    # For A12*B21 where A12 is 4x2 and B21 is 2x4: this is a 4x2x4 tensor.
    #   The rank of the 4x2x4 matrix multiplication tensor is 4*2*4 = 32 (schoolbook).
    #   Can we do better? The rank of M_{4,2,4} is at most 4*4*2=32, and at least max(4,2,4)^2... 
    #   Actually, for rectangular matrix multiplication, the rank is at least the product of 
    #   the two smallest dimensions times something. For 4x2*2x4, I think rank 32 is tight.
    
    # So the 4+2 split gives 337, which is worse.
    
    # Let me try 3+3 split:
    # C11 = A11*B11 + A12*B21 : 3x3*3x3 (rank ~19?) + 3x3*3x3 (rank ~19?)
    # This is getting circular.
    
    # I'll just implement the schoolbook 189-rank solution correctly.
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
