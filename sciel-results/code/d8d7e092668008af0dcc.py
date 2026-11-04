"""Deterministic recursive block decomposition for 4x4*4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen 7 multiplications for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7
    # C indices: (0,0)=8, (0,1)=9, (1,0)=10, (1,1)=11
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[4, 0] = 1; V[7, 0] = 1
    W[8, 0] = 1; W[11, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[4, 1] = 1
    W[10, 1] = 1; W[11, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[5, 2] = 1; V[7, 2] = -1
    W[9, 2] = 1; W[11, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[6, 3] = 1; V[4, 3] = -1
    W[8, 3] = 1; W[10, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[7, 4] = 1
    W[8, 4] = -1; W[9, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[4, 5] = 1; V[5, 5] = 1
    W[11, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[6, 6] = 1; V[7, 6] = 1
    W[8, 6] = 1
    
    return U, V, W


def _strassen_2x2x3():
    """Return (U, V, W) for 2x2 * 2x3 -> 2x3 with rank 8.
    Uses Strassen-Winograd F(2) adapted for rectangular case.
    U shape: (4, 8), V shape: (6, 8), W shape: (6, 8)
    A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    B indices: (0,0)=4, (0,1)=5, (0,2)=6, (1,0)=7, (1,1)=8, (1,2)=9
    C indices: (0,0)=10, (0,1)=11, (0,2)=12, (1,0)=13, (1,1)=14, (1,2)=15
    """
    # We decompose 2x2*2x3 as:
    # Split B into two 2x2 blocks: B_left = B[:,0:2], B_right = B[:,2:3] padded to 2x2
    # Actually, let's use a direct approach:
    # C = A * B where A is 2x2, B is 2x3
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    #
    # Using Strassen-like on the 2x2 part (columns 0,1) and handling column 2:
    # M1 = (A00+A11)*(B00+B10)  -> contributes to C10, C11 (with signs)
    # M2 = (A10+A11)*B00        -> contributes to C10
    # M3 = A00*(B01-B11)        -> contributes to C01
    # M4 = A11*(B10-B00)        -> contributes to C00
    # M5 = (A00+A01)*B11        -> contributes to C01
    # M6 = (A10-A00)*(B00+B01)  -> contributes to C11
    # M7 = (A01-A11)*(B10+B11)  -> contributes to C00
    # M8 = A00*B02 + A01*B12    -> this needs 2 terms... 
    #
    # Alternative: Use the fact that for 2x2*2x3, we can do:
    # Treat it as 2x2*2x2 (cols 0,1) with Strassen (7 mults) + 2x2*2x1 (col 2) with 4 mults
    # But 2x2*2x1 can be done in 4 mults (schoolbook for 2x1 is already 4)
    # Total: 7 + 4 = 11. That's not 8.
    #
    # Better approach: The rank of 2x2*2x3 is known to be 8.
    # Let me construct it directly.
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # 8 multiplications:
    # M1 = (A00+A11)*(B00+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # M8 = (A00*B02 + A01*B12) -- this is 2 mults, so we need to be smarter
    #
    # Actually, let me reconsider. The rank of the 2x2x3 matrix multiplication tensor is 8.
    # Let me use a known decomposition.
    #
    # For 2x2 * 2x3 -> 2x3:
    # We can write C = A*B as:
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # 8 rank-1 terms:
    # t1 = (a+d)(e+h)  -> C00 += 1, C10 += 1
    # t2 = (c+d)e      -> C10 += 1, C11 -= 1  ... wait, let me be more careful
    #
    # Let me use the standard construction. The tensor T_{abc} where a indexes A (4), b indexes B (6), c indexes C (6).
    # T[a,b,c] = 1 if the corresponding entry of A times entry of B contributes to entry of C.
    #
    # A entries: a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
    # B entries: b0=B[0,0], b1=B[0,1], b2=B[0,2], b3=B[1,0], b4=B[1,1], b5=B[1,2]
    # C entries: c0=C[0,0], c1=C[0,1], c2=C[0,2], c3=C[1,0], c4=C[1,1], c5=C[1,2]
    #
    # C[0,0] = a0*b0 + a1*b3
    # C[0,1] = a0*b1 + a1*b4
    # C[0,2] = a0*b2 + a1*b5
    # C[1,0] = a2*b0 + a3*b3
    # C[1,1] = a2*b1 + a3*b4
    # C[1,2] = a2*b2 + a3*b5
    #
    # 8-term decomposition:
    # r1: (a0+a3)*(b0+b3) -> C00+=1, C10+=1
    # r2: (a2+a3)*b0      -> C10+=1, C00-=1  ... hmm, let me verify
    #
    # Actually, let me just use the following verified 8-term decomposition:
    # r1: (a0+a3)*(b0+b3)
    #   U: a0=1, a3=1; V: b0=1, b3=1; W: c0=1, c3=1
    # r2: (a2+a3)*b0
    #   U: a2=1, a3=1; V: b0=1; W: c3=1, c0=-1
    #   Check: contributes a2*b0 to c3 (correct for C10) and -a3*b0 to c0
    #   C00 needs a0*b0 + a1*b3. From r1: a0*b0 + a3*b0 + a0*b3 + a3*b3
    #   From r2: -a3*b0. Net for c0: a0*b0 + a0*b3. Still missing a1*b3 and need to subtract a3*b3.
    #   This is getting complicated. Let me use a different known decomposition.
    #
    # Known rank-8 for 2x2*2x3 (from literature):
    # Let me just use the block approach: Strassen for cols 0,1 (7 terms) + schoolbook for col 2 (4 terms) = 11.
    # That's rank 11, not 8. The rank 8 is a more clever construction.
    #
    # For now, let me use rank 11 for the 2x2x3 part. Total: 7 + 11 = 18 for the 2x2x3 block.
    # Wait, I'm overcomplicating. Let me just do:
    # - 2x2x2 block: Strassen rank 7
    # - 2x2x3 block: schoolbook rank 12 (2*2*3)
    # Total for top-left 2x2 * full 2x5: 7 + 12 = 19
    # - Remaining 2x2 * 2x5 (bottom-left A, full B): schoolbook 2*2*5 = 20
    # Total: 19 + 20 = 39
    #
    # Hmm, that's rank 39. Let me think more carefully.
    #
    # Actually, the simplest structural improvement:
    # 4x4*4x5 = (2x2 block diagonal) approach:
    # Split A into 4 blocks of 2x2, B into 4 blocks of 2x2 and one 2x1.
    # This is getting complex. Let me just do a clean block decomposition.
    
    # Simple approach: Use Strassen for each 2x2*2x2 block in the 4x4*4x4 part (4 blocks, rank 7 each = 28)
    # Plus schoolbook for 4x4*4x1 (16 terms)
    # Total: 28 + 16 = 44
    
    # Even simpler and cleaner: just use Strassen recursively.
    # 4x4*4x4 can be done with 4 Strassen 2x2*2x2 blocks? No, that's not how it works.
    # 4x4*4x4 = 16 blocks of 1x1*1x1 if naive.
    # With 2x2 Strassen: 4x4 = 4 blocks of 2x2, so we need 4*4=16 sub-products of 2x2*2x2.
    # Each 2x2*2x2 takes rank 7. Total: 16*7 = 112. That's worse than schoolbook 64!
    # 
    # The issue: Strassen helps when you multiply the 2x2 blocks together, not when you just replace each scalar mult.
    #
    # Correct recursive approach:
    # 4x4*4x4: Split into 2x2 blocks. A has 4 blocks, B has 4 blocks.
    # C has 4 blocks, each C_ij = sum_k A_ik * B_kj (2x2 matrix products).
    # Using Strassen on the 2x2 block structure: 7 "block multiplications" of 2x2*2x2.
    # Each 2x2*2x2 can be done with rank 7 (Strassen again) or rank 4 (schoolbook).
    # If we use Strassen recursively: 7 * 7 = 49 for 4x4*4x4.
    # Plus 4x4*4x1: 16 schoolbook terms.
    # Total: 49 + 16 = 65.
    
    # Let me implement the 4x4*4x4 with recursive Strassen (rank 49) + 4x4*4x1 schoolbook (rank 16) = 65.
    pass


def _schoolbook_block(n, m, p, offset_u, offset_v, offset_w, U, V, W, start_rank):
    """Add schoolbook decomposition for n*m*p block. Returns new start_rank."""
    r = start_rank
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[offset_u + i * m + j, r] = 1
                V[offset_v + j * p + k, r] = 1
                W[offset_w + k * n + i, r] = 1
                r += 1
    return r


def _strassen_block_2x2x2(offset_u, offset_v, offset_w, U, V, W, start_rank):
    """Add Strassen 2x2*2x2 decomposition at given offsets. Returns new start_rank."""
    su, sv, sw = _strassen_2x2()
    for r in range(7):
        U[offset_u:offset_u+4, start_rank + r] += su[:, r]
        V[offset_v:offset_v+4, start_rank + r] += sv[:, r]
        W[offset_w:offset_w+4, start_rank + r] += sw[:, r]
    return start_rank + 7


def _recursive_strassen_4x4x4(U, V, W, start_rank):
    """
    Decompose 4x4 * 4x4 -> 4x4 using recursive Strassen.
    Split into 2x2 blocks. 7 block-multiplications, each 2x2*2x2 with Strassen (rank 7).
    Total rank: 49.
    
    A blocks: A00, A01, A10, A11 (each 2x2)
    B blocks: B00, B01, B10, B11 (each 2x2)
    C blocks: C00, C01, C10, C11 (each 2x2)
    
    Strassen on blocks:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each M is a 2x2*2x2 product, done with Strassen (rank 7).
    Total: 7 * 7 = 49 rank.
    """
    # Block offsets within the 4x4 matrices:
    # A: 4x4, blocks are 2x2.
    #   A00: rows 0-1, cols 0-1 -> offset (0,0) in A, linear index 0*4+0=0 to 1*4+1=5
    #   A01: rows 0-1, cols 2-3 -> offset (0,2), linear 0*4+2=2 to 1*4+3=7
    #   A10: rows 2-3, cols 0-1 -> offset (2,0), linear 2*4+0=8 to 3*4+1=13
    #   A11: rows 2-3, cols 2-3 -> offset (2,2), linear 2*4+2=10 to 3*4+3=15
    
    # For U (shape 16 x rank), U[i*4+j, r] corresponds to A[i,j]
    # A00: i in {0,1}, j in {0,1} -> U rows 0,1,4,5
    # A01: i in {0,1}, j in {2,3} -> U rows 2,3,6,7
    # A10: i in {2,3}, j in {0,1} -> U rows 8,9,12,13
    # A11: i in {2,3}, j in {2,3} -> U rows 10,11,14,15
    
    # Similarly for V (B) and W (C).
    
    # Block row/col indices for 2x2 sub-blocks:
    # Block (bi, bj) of a 4x4 matrix:
    #   bi=0, bj=0: rows 0,1; cols 0,1 -> linear indices 0,1,4,5
    #   bi=0, bj=1: rows 0,1; cols 2,3 -> linear indices 2,3,6,7
    #   bi=1, bj=0: rows 2,3; cols 0,1 -> linear indices 8,9,12,13
    #   bi=1, bj=1: rows 2,3; cols 2,3 -> linear indices 10,11,14,15
    
    def block_indices(bi, bj, dim):
        """Get the 4 linear indices for block (bi,bj) in a dim x dim matrix."""
        rows = [bi*2, bi*2+1]
        cols = [bj*2, bj*2+1]
        return [r * dim + c for r in rows for c in cols]
    
    # Get block index lists
    A00 = block_indices(0, 0, 4)  # [0,1,4,5]
    A01 = block_indices(0, 1, 4)  # [2,3,6,7]
    A10 = block_indices(1, 0, 4)  # [8,9,12,13]
    A11 = block_indices(1, 1, 4)  # [10,11,14,15]
    
    B00 = block_indices(0, 0, 4)  # [0,1,4,5]
    B01 = block_indices(0, 1, 4)  # [2,3,6,7]
    B10 = block_indices(1, 0, 4)  # [8,9,12,13]
    B11 = block_indices(1, 1, 4)  # [10,11,14,15]
    
    C00 = block_indices(0, 0, 4)  # [0,1,4,5]
    C01 = block_indices(0, 1, 4)  # [2,3,6,7]
    C10 = block_indices(1, 0, 4)  # [8,9,12,13]
    C11 = block_indices(1, 1, 4)  # [10,11,14,15]
    
    su, sv, sw = _strassen_2x2()
    
    r = start_rank
    
    # M1 = (A00+A11)*(B00+B11)
    for sr in range(7):
        for idx in A00:
            U[idx, r + sr] += su[idx % 4 if idx < 4 else (idx-4) % 4, sr]  # This is wrong, let me fix
        # Actually, I need to map the 4 indices of A00 to the 4 positions in the Strassen U.
        # A00 indices [0,1,4,5] map to Strassen positions [0,1,2,3] (i.e., A[0,0],A[0,1],A[1,0],A[1,1])
        # A11 indices [10,11,14,15] map to same Strassen positions.
        pass
    
    # Let me redo this more carefully.
    # The 4 linear indices of block (bi,bj) in order correspond to:
    # [bi*2, bj*2], [bi*2, bj*2+1], [bi*2+1, bj*2], [bi*2+1, bj*2+1]
    # which are exactly the 4 positions in a 2x2 matrix: (0,0), (0,1), (1,0), (1,1)
    # So the mapping is: block_indices[0]->pos0, [1]->pos1, [2]->pos2, [3]->pos3
    
    r = start_rank
    
    # Helper to add a Strassen 2x2*2x2 for specific blocks
    def add_strassen_mult(A_blocks, B_blocks, C_blocks, signs, base_r):
        """Add Strassen decomposition where A = sum of A_blocks, B = sum of B_blocks, result goes to C_blocks with signs.
        A_blocks: list of 4-index lists (each representing a 2x2 block)
        B_blocks: list of 4-index lists
        C_blocks: list of (4-index list, sign) pairs
        """
        for sr in range(7):
            # U part: A = sum of A_blocks
            for A_blk in A_blocks:
                for pos in range(4):
                    U[A_blk[pos], base_r + sr] += su[pos, sr]
            # V part: B = sum of B_blocks  
            for B_blk in B_blocks:
                for pos in range(4):
                    V[B_blk[pos], base_r + sr] += sv[pos, sr]
            # W part: C = sum of C_blocks with signs
            for C_blk, sign in C_blocks:
                for pos in range(4):
                    W[C_blk[pos], base_r + sr] += sign * sw[pos, sr]
        return base_r + 7
    
    # M1 = (A00+A11)*(B00+B11)
    # Contributes to: C00(+1), C11(+1)
    r = add_strassen_mult([A00, A11], [B00, B11], [(C00, 1), (C11, 1)], None, r)
    
    # M2 = (A10+A11)*B00
    # Contributes to: C10(+1), C11(-1)
    r = add_strassen_mult([A10, A11], [B00], [(C10, 1), (C11, -1)], None, r)
    
    # M3 = A00*(B01-B11)
    # Contributes to: C01(+1), C11(+1)
    r = add_strassen_mult([A00], [B01, [-1]*0 + B11], [(C01, 1), (C11, 1)], None, r)
    # Wait, B01-B11 means B = B01 + (-1)*B11. Let me handle signs in B_blocks.
    # I need to modify add_strassen_mult to accept signed blocks for B.
    
    # Let me redo with a cleaner helper that accepts signed blocks.
    # Actually, let me just inline it for clarity.
    
    # Reset and do it properly
    r = start_rank
    
    def add_strassen_signed(A_blocks_signed, B_blocks_signed, C_blocks_signed, base_r):
        """
        A_blocks_signed: list of (block_indices, sign)
        B_blocks_signed: list of (block_indices, sign)
        C_blocks_signed: list of (block_indices, sign)
        """
        for sr in range(7):
            for A_blk, A_sign in A_blocks_signed:
                for pos in range(4):
                    U[A_blk[pos], base_r + sr] += A_sign * su[pos, sr]
            for B_blk, B_sign in B_blocks_signed:
                for pos in range(4):
                    V[B_blk[pos], base_r + sr] += B_sign * sv[pos, sr]
            for C_blk, C_sign in C_blocks_signed:
                for pos in range(4):
                    W[C_blk[pos], base_r + sr] += C_sign * sw[pos, sr]
        return base_r + 7
    
    # M1 = (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
    r = add_strassen_signed(
        [(A00, 1), (A11, 1)],
        [(B00, 1), (B11, 1)],
        [(C00, 1), (C11, 1)],
        r
    )
    
    # M2 = (A10+A11)*B00 -> C10(+1), C11(-1)
    r = add_strassen_signed(
        [(A10, 1), (A11, 1)],
        [(B00, 1)],
        [(C10, 1), (C11, -1)],
        r
    )
    
    # M3 = A00*(B01-B11) -> C01(+1), C11(+1)
    r = add_strassen_signed(
        [(A00, 1)],
        [(B01, 1), (B11, -1)],
        [(C01, 1), (C11, 1)],
        r
    )
    
    # M4 = A11*(B10-B00) -> C00(+1), C10(+1)
    r = add_strassen_signed(
        [(A11, 1)],
        [(B10, 1), (B00, -1)],
        [(C00, 1), (C10, 1)],
        r
    )
    
    # M5 = (A00+A01)*B11 -> C00(-1), C01(+1)
    r = add_strassen_signed(
        [(A00, 1), (A01, 1)],
        [(B11, 1)],
        [(C00, -1), (C01, 1)],
        r
    )
    
    # M6 = (A10-A00)*(B00+B01) -> C11(+1)
    r = add_strassen_signed(
        [(A10, 1), (A00, -1)],
        [(B00, 1), (B01, 1)],
        [(C11, 1)],
        r
    )
    
    # M7 = (A01-A11)*(B10+B11) -> C00(+1)
    r = add_strassen_signed(
        [(A01, 1), (A11, -1)],
        [(B10, 1), (B11, 1)],
        [(C00, 1)],
        r
    )
    
    return r


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: 4x4*4x5 = 4x4*4x4 (Strassen recursive, rank 49) + 4x4*4x1 (schoolbook, rank 16)
    # Total rank: 65
    
    R = 65
    U = np.zeros((n * m, R), dtype=float)  # (16, 65)
    V = np.zeros((m * p, R), dtype=float)  # (20, 65)
    W = np.zeros((p * n, R), dtype=float)  # (20, 65)
    
    # Part 1: 4x4 * 4x4 -> 4x4 (columns 0-3 of B, rows 0-3 of C)
    # B columns 0-3: V indices j*5+k for k=0..3
    # C rows 0-3: W indices k*4+i for k=0..3
    r = _recursive_strassen_4x4x4(U, V[:, 0:4], W[:, 0:4], 0)
    # r should be 49
    
    # Part 2: 4x4 * 4x1 -> 4x1 (column 4 of B, column 4 of C)
    # B column 4: V indices j*5+4 for j=0..3 -> linear indices 4, 9, 14, 19
    # C column 4: W indices 4*4+i = 16+i for i=0..3 -> linear indices 16,17,18,19
    r2 = r  # start at 49
    for i in range(n):  # 4
        for j in range(m):  # 4
            U[i * m + j, r2] = 1
            V[j * p + 4, r2] = 1  # j*5+4
            W[4 * n + i, r2] = 1  # 16+i
            r2 += 1
    # r2 should be 49 + 16 = 65
    
    # Verify within deadline
    if time.time() < deadline:
        # Quick verification: check a few random entries
        # The tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Should equal 1 iff a,b,c form a valid (i,j,k) triple where C[i,k] gets A[i,j]*B[j,k]
        # i.e., a = i*m+j, b = j*p+k, c = k*n+i
        import random
        random.seed(42)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Verify all entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        # Also check that no spurious entries are generated
        if valid:
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        # Check if this (a,b,c) is a valid triple
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        is_valid = (j == j2) and (k == k2) and (i == i2)
                        s = 0.0
                        for r in range(R):
                            s += U[a, r] * V[b, r] * W[c, r]
                        if is_valid:
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        else:
                            if abs(s) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
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
