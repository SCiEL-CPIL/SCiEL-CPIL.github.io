"""Matrix multiplication 4x4 * 4x7 tensor decomposition.
Uses Strassen for 4x4*4x4 (rank 56) + Strassen for 4x4*4x4 (rank 56) 
+ 4x4*4x3 portion (rank 48) = 160? No wait.
Better: split B into B1 (4x4) and B2 (4x3).
C = A*B1 + A*B2.
Use Strassen for A*B1 (rank 56) and schoolbook for A*B2 (rank 48).
Total rank = 56 + 48 = 104.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_4x4_4x4():
    """Return (U, V, W) for 4x4 * 4x4 -> 4x4 matrix multiplication using Strassen.
    U shape: (16, 56), V shape: (16, 56), W shape: (16, 56)
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    n = 4
    rank = 56
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen for 4x4*4x4: split each 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = A*B
    # 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B10+B11)
    # M7 = (A01-A11)(B00+B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2*2x2 Strassen, each Mx is computed via schoolbook (4 multiplications)
    # So total = 7 * 4 = 28 scalar multiplications.
    # But we need rank for 4x4*4x4, which is 16*16=256 entries? No.
    # The rank is the number of scalar multiplications = 28? No, 4x4*4x4 has 16*16*16? 
    # No, matrix mult A(4x4)*B(4x4)->C(4x4): tensor is 4x4x4x4x4x4? 
    # No, the multiplication tensor M_{n,m,p} has dimensions n*m x m*p x p*n.
    # For 4x4*4x4: 16 x 16 x 16. Rank is min number of rank-1 terms.
    # Strassen for 4x4*4x4: treat as 2x2 block matrices, each block 2x2.
    # Each 2x2*2x2 uses 7 multiplications (Strassen), each 2x2*2x2 multiplication 
    # requires 4 scalar multiplications in schoolbook, but Strassen replaces that.
    # So 4x4*4x4 via Strassen: 7 * (2x2*2x2) = 7*4 = 28? No.
    # Actually, each "multiplication" in the block Strassen is a 2x2*2x2 multiplication,
    # which itself has rank 7 (Strassen) or 4 (schoolbook).
    # So total rank = 7 * 7 = 49? Or 7 * 4 = 28?
    # Using Strassen recursively: 4x4 = 2 blocks of 2x2, so 7 multiplications of 2x2*2x2.
    # Each 2x2*2x2 has rank 7 (Strassen) or 4 (schoolbook).
    # If we use Strassen for the 2x2 level: 7*7 = 49.
    # If we use schoolbook for 2x2: 7*4 = 28.
    # But 28 < 49, so schoolbook for 2x2 is better.
    # Wait, but the known rank of 2x2*2x2 is 7 (Strassen), and schoolbook is 4.
    # 4 < 7, so schoolbook is better for 2x2!
    # So 4x4*4x4 via block Strassen with schoolbook 2x2: rank = 7*4 = 28.
    
    # Let me just use the known result: 4x4*4x4 has rank at most 49 (Strassen^2) 
    # or 28 (Strassen + schoolbook). Let me go with 28.
    
    # Actually, let me just build it explicitly.
    # Split indices: for i in 0..3, block_i = i//2, sub_i = i%2
    # For the 7 block multiplications, each produces a 2x2*2x2 result.
    # Each 2x2*2x2 multiplication (schoolbook) has 4 rank-1 terms.
    
    # Block structure:
    # A blocks: A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B blocks: B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # M1 = (A00+A11)(B00+B11): 2x2*2x2
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B10+B11)
    # M7 = (A01-A11)(B00+B10)
    
    # Each Mx is 2x2*2x2, schoolbook rank 4.
    # Total rank = 7*4 = 28.
    
    # Let me build U, V, W for this.
    # U row index: i*4+j for A[i,j], V row index: j*4+k for B[j,k], W row index: k*4+i for C[k,i]... 
    # Wait, W index is k*n+i where n=4, so W[k*4+i].
    
    # For each block multiplication Mx, we have a 2x2*2x2 schoolbook.
    # Let me define helper: given a set of (ai, aj) pairs for U, (bj, bk) for V, (ck, ci) for W,
    # and a coefficient, add to U, V, W at column c.
    
    col = 0
    # Define the 7 block multiplications with their A-blocks, B-blocks, and C-block contributions
    # M1: A = A00+A11, B = B00+B11, contributes to C00 (+1), C11 (+1)
    # M2: A = A10+A11, B = B00, contributes to C10 (+1), C11 (-1)
    # M3: A = A00, B = B01-B11, contributes to C01 (+1), C11 (+1)
    # M4: A = A11, B = B10-B00, contributes to C00 (+1), C10 (+1)
    # M5: A = A00+A10, B = B11, contributes to C00 (-1), C01 (+1)
    # M6: A = A11-A10, B = B10+B11, contributes to C11 (+1)
    # M7: A = A01-A11, B = B00+B10, contributes to C00 (+1), C11 (-1)
    
    # Block indices: block 0 -> rows/cols 0,1; block 1 -> rows/cols 2,3
    # A00: i in {0,1}, j in {0,1}
    # A01: i in {0,1}, j in {2,3}
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    
    def block_range(block):
        return [block*2, block*2+1]
    
    # For each of the 7 multiplications, and for each of the 4 schoolbook terms within:
    mults = [
        # (A_blocks, B_blocks, C_contributions)
        # A_blocks: list of (block, sign)
        # B_blocks: list of (block, sign)
        # C_contributions: list of (c_block, sign)
        ([(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)]),  # M1
        ([(1,1),(1,1)], [(0,1)], [(1,1),(1,-1)]),        # M2
        ([(0,1)], [(0,1),(1,-1)], [(0,1),(1,1)]),        # M3
        ([(1,1)], [(1,1),(0,-1)], [(0,1),(1,1)]),        # M4
        ([(0,1),(1,1)], [(1,1)], [(0,-1),(0,1)]),        # M5
        ([(1,1),(1,-1)], [(1,1),(1,1)], [(1,1)]),        # M6
        ([(0,1),(1,-1)], [(0,1),(1,1)], [(0,1),(1,-1)]), # M7
    ]
    
    for a_blocks, b_blocks, c_blocks in mults:
        # Each block multiplication is 2x2*2x2 schoolbook: 4 terms
        # For sub-indices si, sj in {0,1}:
        for si in range(2):
            for sj in range(2):
                # U: for each (block_a, sign_a) in a_blocks, for row i = block_a*2+si, col j = sj... 
                # Wait, the U matrix indexes A[i,j]. For block multiplication, A is 2x2 with indices 
                # (local_i, local_j) where local_i, local_j in {0,1}.
                # A_block[local_i, local_j] = A[block_i*2+local_i, block_j*2+local_j]
                # So for a given (a_block, sign), the A element at (si, sj) is at global (a_block*2+si, sj)
                # Wait no: sj is the column index within the 2x2 block, but the block also has a column offset.
                # A_block has rows [block*2, block*2+1] and cols [block*2, block*2+1]? No!
                # A00: rows 0-1, cols 0-1. A01: rows 0-1, cols 2-3.
                # So A_block(bi, bj) has rows bi*2..bi*2+1, cols bj*2..bj*2+1.
                # For the 2x2*2x2 multiplication, the A matrix is 2x2 with local indices (li, lj).
                # Global A index: (bi*2+li, bj*2+lj).
                # Similarly for B: B block (bj, bk) has rows bj*2..bj*2+1, cols bk*2..bk*2+1.
                # Local B index: (lj, lk) -> global (bj*2+lj, bk*2+lk).
                # C block (ci, ck) has rows ci*2..ci*2+1, cols ck*2..ck*2+1.
                # Local C index: (lk, li) -> global (ci*2+lk, ck*2+li).
                # Wait, C = A*B, so C[ci, ck] = sum_j A[ci, j]*B[j, ck].
                # The local indices: C_local[lk, li] where lk is the local row of B (which is local col of A)
                # Hmm, let me be more careful.
                
                # In the 2x2*2x2 block multiplication:
                # A_local[li, lj], B_local[lj, lk], C_local[li, lk] = sum_lj A_local[li,lj]*B_local[lj,lk]
                # Schoolbook: for each lj, C_local[li,lk] += A_local[li,lj]*B_local[lj,lk]
                # So for each (li, lj, lk), we have a term.
                
                # But in our decomposition, for each of the 7 block multiplications, we have:
                # The "A" is a sum of blocks, the "B" is a sum of blocks.
                # The product expands to sum of (A_block_a)(B_block_b) terms.
                # Each such term is a 2x2*2x2 multiplication, which has 4 schoolbook terms.
                
                # So for each pair (a_block, b_block), we have a 2x2*2x2 schoolbook.
                # And the result contributes to C blocks with signs.
                
                pass  # This is getting complex. Let me just build it more directly.
    
    # Let me take a different approach: just enumerate all the rank-1 terms directly.
    # For the 4x4*4x4 Strassen decomposition, there are 28 rank-1 terms.
    # Let me just hardcode them based on the Strassen formulas.
    
    # Actually, this is getting too complex. Let me just use the schoolbook for 4x4*4x4 (rank 256)
    # and see if I can do better.
    
    # Wait, the problem says 4x4 * 4x7. Let me re-read.
    # "multiplying a 4 by 4 matrix by a 4 by 7 matrix"
    # So A is 4x4, B is 4x7, C is 4x7.
    # n=4, m=4, p=7.
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (28, rank)
    
    # Schoolbook: rank = 4*4*7 = 112.
    
    # Can we do better? Split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2.
    # A*B1: 4x4*4x4, rank at most 28 (Strassen+schoolbook) or 49 (Strassen^2).
    # A*B2: 4x4*4x3, rank at most 4*4*3 = 48 (schoolbook).
    # Total: 28 + 48 = 76 or 49 + 48 = 97.
    
    # Let me just do schoolbook for everything: rank 112.
    # Or try to be clever.
    
    # For now, let me just return the schoolbook decomposition to be safe.
    # The previous program scored 1.0, which is the seed.
    # Let me try rank 104 as mentioned in the knowledge.
    
    # 4x4*4x4 with Strassen: rank 49 (if using Strassen recursively) or 28.
    # 4x4*4x3 with schoolbook: rank 48.
    # Total: 49 + 48 = 97 or 28 + 48 = 76.
    
    # Let me just build the schoolbook for 4x4*4x7 (rank 112) first, then try to improve.
    n, m, p = 4, 4, 7
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
    return U, V, W

def construct():
    """Construct the tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys.
    """
    n, m, p = DIMENSIONS
    # Split B into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    # Use Strassen for A*B1 (4x4*4x4) and schoolbook for A*B2 (4x4*4x3)
    
    # For 4x4*4x4 Strassen: rank 49 (conservative, using Strassen at both levels)
    # For 4x4*4x3 schoolbook: rank 48
    # Total: 97
    
    # Let me build the Strassen 4x4*4x4 part.
    # I'll use the recursive Strassen: 4x4 = 2x2 blocks, 2x2*2x2 = 7 multiplications,
    # each 2x2*2x2 done via schoolbook (4 mults) -> 28 total.
    # Or Strassen at 2x2 level: 7*7=49.
    # Let me use 28 (schoolbook for 2x2).
    
    rank_4x4 = 28
    rank_4x3 = 4 * 4 * 3  # 48
    total_rank = rank_4x4 + rank_4x3  # 76
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Part 1: A * B1 where B1 = B[:, 0:4] (4x4)
    # Strassen 4x4*4x4, rank 28
    col = 0
    # 7 block multiplications, each with 4 schoolbook terms
    # Block structure: 2x2 blocks of 2x2
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    # C blocks: C00, C01, C10, C11 (2x2 each)
    
    # For each of the 7 multiplications Mx = Ax * Bx:
    # Ax is a sum of A-blocks, Bx is a sum of B-blocks
    # The product contributes to C-blocks with signs
    
    # Let me define the 7 multiplications:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A10)B11 -> C00 -= M5, C01 += M5
    # M6 = (A11-A10)(B10+B11) -> C11 += M6
    # M7 = (A01-A11)(B00+B10) -> C00 += M7, C11 -= M7
    
    # Each Mx is a 2x2*2x2 multiplication.
    # For 2x2*2x2 schoolbook: 4 terms.
    # A_local[li,lj] * B_local[lj,lk] contributes to C_local[li,lk]
    
    # For each (a_block_list, b_block_list, c_contrib_list):
    mults = [
        # (a_blocks: [(block, sign)], b_blocks: [(block, sign)], c_contribs: [(block, sign)])
        # A-blocks: 0=A00, 1=A01, 2=A10, 3=A11
        # B-blocks: 0=B00, 1=B01, 2=B10, 3=B11
        # C-blocks: 0=C00, 1=C01, 2=C10, 3=C11
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),   # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),         # M2
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),         # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),         # M4
        ([(0,1),(2,1)], [(3,1)], [(0,-1),(1,1)]),         # M5
        ([(3,1),(2,-1)], [(2,1),(3,1)], [(3,1)]),         # M6
        ([(1,1),(3,-1)], [(0,1),(2,1)], [(0,1),(3,-1)]),  # M7
    ]
    
    for a_blks, b_blks, c_blks in mults:
        # Expand: for each pair (ab, bb), we have a 2x2*2x2 schoolbook
        for ab in a_blks:
            for bb in b_blks:
                a_block, a_sign = ab
                b_block, b_sign = bb
                # 2x2*2x2 schoolbook: 4 terms
                for li in range(2):
                    for lj in range(2):
                        for lk in range(2):
                            # U: A element at global (a_block_row*2+li, a_block_col*2+lj)
                            # Wait, A-block ab has row block and col block.
                            # A-block 0 (A00): row block 0, col block 0
                            # A-block 1 (A01): row block 0, col block 1
                            # A-block 2 (A10): row block 1, col block 0
                            # A-block 3 (A11): row block 1, col block 1
                            a_row_block = a_block // 2
                            a_col_block = a_block % 2
                            # B-block bb: row block (col block of A), col block
                            b_row_block = b_block // 2
                            b_col_block = b_block % 2
                            # The A element is at global row a_row_block*2+li, col a_col_block*2+lj
                            # The B element is at global row b_row_block*2+lj, col b_col_block*2+lk
                            # For this to be a valid term, we need a_col_block == b_row_block (they share the j index)
                            # In Strassen, the blocks are aligned, so this should hold.
                            
                            u_row = a_row_block * 2 + li
                            u_col_local = a_col_block * 2 + lj
                            v_row = b_row_block * 2 + lj
                            v_col_local = b_col_block * 2 + lk
                            
                            # U index: u_row * 4 + u_col_local
                            # V index: v_row * 7 + v_col_local (since p=7 for B)
                            # Wait, V is (m*p, rank) = (4*7, rank) = (28, rank)
                            # V[j*p+k] where j is row of B (0..3), k is col of B (0..6)
                            # For B1 (4x4), k ranges 0..3.
                            v_row_global = b_row_block * 2 + lj  # 0..3
                            v_col_global = b_col_block * 2 + lk  # 0..3
                            
                            # W index: k*n+i where k is col of C (0..3), i is row of C (0..3)
                            # C-block c_block: row block and col block
                            # For each c_blk in c_blks:
                            # C element at global row c_row_block*2+li, col c_col_block*2+lk
                            
                            u_idx = u_row * 4 + u_col_local
                            v_idx = v_row_global * 7 + v_col_global
                            
                            coeff = a_sign * b_sign
                            
                            for c_blk, c_sign in c_blks:
                                c_row_block = c_blk // 2
                                c_col_block = c_blk % 2
                                w_row = c_col_block * 2 + lk
                                w_col = c_row_block * 2 + li
                                w_idx = w_row * 4 + w_col
                                
                                U[u_idx][col] += coeff * c_sign
                                V[v_idx][col] = 1
                                W[w_idx][col] = 1
                            
                            col += 1
    # col should be 28 now
    
    # Part 2: A * B2 where B2 = B[:, 4:7] (4x3)
    # Schoolbook: rank 48
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 7 + (4 + k)  # k is 0..2, but B col is 4..6
                w_idx = (4 + k) * 4 + i  # C col is 4..6, row is i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
