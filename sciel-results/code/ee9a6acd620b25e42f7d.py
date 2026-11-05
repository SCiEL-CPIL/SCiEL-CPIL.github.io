import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # All 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def check_time():
        return time.time() > deadline
    
    def make_vec(size):
        return [0] * size
    
    # Strassen decomposition for 2x2x2 at given offsets
    def strassen_2x2x2(off_a, off_b, off_c):
        # M1 = (A00+A11)(B00+B11)
        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
        u_vec[(off_a+0)*5 + (off_b+0)] = 1
        u_vec[(off_a+1)*5 + (off_b+1)] = 1
        v_vec[(off_b+0)*5 + (off_c+0)] = 1
        v_vec[(off_b+1)*5 + (off_c+1)] = 1
        w_vec[(off_c+0)*5 + (off_a+0)] = 1
        w_vec[(off_c+1)*5 + (off_a+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M2 = (A10+A11)B00
        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
        u_vec[(off_a+1)*5 + (off_b+0)] = 1
        u_vec[(off_a+1)*5 + (off_b+1)] = 1
        v_vec[(off_b+0)*5 + (off_c+0)] = 1
        w_vec[(off_c+0)*5 + (off_a+1)] = 1
        w_vec[(off_c+1)*5 + (off_a+1)] = -1
        add_col(u_vec, v_vec, w_vec)
        
        # M3 = A00(B01-B11)
        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
        u_vec[(off_a+0)*5 + (off_b+0)] = 1
        v_vec[(off_b+0)*5 + (off_c+1)] = 1
        v_vec[(off_b+1)*5 + (off_c+1)] = -1
        w_vec[(off_c+1)*5 + (off_a+0)] = 1
        w_vec[(off_c+1)*5 + (off_a+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M4 = A11(B10-B00)
        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
        u_vec[(off_a+1)*5 + (off_b+1)] = 1
        v_vec[(off_b+1)*5 + (off_c+0)] = 1
        v_vec[(off_b+0)*5 + (off_c+0)] = -1
        w_vec[(off_c+0)*5 + (off_a+0)] = 1
        w_vec[(off_c+1)*5 + (off_a+0)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M5 = (A00+A01)B11
        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
        u_vec[(off_a+0)*5 + (off_b+0)] = 1
        u_vec[(off_a+0)*5 + (off_b+1)] = 1
        v_vec[(off_b+1)*5 + (off_c+1)] = 1
        w_vec[(off_c+0)*5 + (off_a+0)] = -1
        w_vec[(off_c+1)*5 + (off_a+0)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M6 = (A10-A00)(B00+B01)
        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
        u_vec[(off_a+1)*5 + (off_b+0)] = 1
        u_vec[(off_a+0)*5 + (off_b+0)] = -1
        v_vec[(off_b+0)*5 + (off_c+0)] = 1
        v_vec[(off_b+0)*5 + (off_c+1)] = 1
        w_vec[(off_c+1)*5 + (off_a+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M7 = (A01-A11)(B10+B11)
        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
        u_vec[(off_a+0)*5 + (off_b+1)] = 1
        u_vec[(off_a+1)*5 + (off_b+1)] = -1
        v_vec[(off_b+1)*5 + (off_c+0)] = 1
        v_vec[(off_b+1)*5 + (off_c+1)] = 1
        w_vec[(off_c+0)*5 + (off_a+0)] = 1
        add_col(u_vec, v_vec, w_vec)
    
    # Partition 5 into 4+1
    # Core: 4x4x4, Boundary: 1x1x1 and cross terms
    # 4x4x4 can be done with 4 Strassen 2x2x2 blocks (rank 7*4 = 28)
    # But we can do better: partition 4 into 2+2
    
    # For 4x4x4: partition into four 2x2x2 blocks
    # Block (0,0,0): offsets (0,0,0)
    # Block (0,0,1): offsets (0,2,0) - no wait, let me think about this more carefully
    
    # Actually for 4x4x4, we partition the 4x4 A matrix into four 2x2 blocks,
    # same for B and C. The 16 block interactions give us:
    # A_{i,j} * B_{j,k} -> C_{i,k} for i,j,k in {0,1} where each index represents a 2x2 block
    
    # For the 4x4x4 core, we use:
    # For each of the 8 (i,j) pairs of A blocks and (j,k) pairs of B blocks:
    #   We need A_{i,j} * B_{j,k} to contribute to C_{i,k}
    # This gives 8*8 = 64 schoolbook terms, but we can use Strassen on each 2x2x2 interaction
    
    # Better approach: Treat the 4x4 as 2x2 blocks of 2x2.
    # The 4x4x4 tensor can be decomposed using Strassen at the block level.
    # At the block level, we have a 2x2x2 "meta" tensor where each entry is a 2x2x2 tensor.
    # Using Strassen at the meta level: 7 meta-terms, each of which is a 2x2x2 tensor.
    # Each meta-term involves products of 2x2 blocks, which we can compute with Strassen.
    
    # Actually, let me just use the straightforward approach:
    # 4x4x4 = 4 Strassen 2x2x2 blocks (for the diagonal 2x2x2 blocks)
    # plus cross terms handled by schoolbook
    
    # Let me reconsider. The cleanest approach for 5x5x5:
    # Partition: 4 + 1
    # Core 4x4x4: use 4 independent Strassen 2x2x2 (for the four 2x2x2 diagonal blocks)
    #   = 4 * 7 = 28 rank
    # Cross terms:
    #   A[0:4, 0:4] * B[4, 0:4] -> C[0:4, 4]: 4*4 = 16 terms
    #   A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]: 4*4 = 16 terms  
    #   A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]: 4*4 = 16 terms
    #   A[0:4, 4] * B[4, 0:4] -> C[0:4, 4]: 4*1 = 4 terms
    #   A[0:4, 4] * B[0:4, 4] -> C[0:4, 4]: 4*4 = 16 terms
    #   A[4, 0:4] * B[0:4, 4] -> C[4, 4]: 4*1 = 4 terms
    #   A[4, 4] * B[0:4, 0:4] -> C[4, 0:4]: 4*4 = 16 terms
    #   A[4, 4] * B[4, 0:4] -> C[4, 0:4]: 1*4 = 4 terms
    #   A[4, 4] * B[0:4, 4] -> C[4, 4]: 1*4 = 4 terms
    #   A[4, 4] * B[4, 4] -> C[4, 4]: 1 term
    
    # Wait, I'm overcomplicating this. Let me just enumerate all the (i,j,k) triples
    # where the block sizes are determined by the 4+1 partition.
    
    # Partition indices: first 4, then 1
    # Blocks:
    # (0,0): rows 0-3, cols 0-3 of A (4x4)
    # (0,1): rows 0-3, col 4 of A (4x1)
    # (1,0): row 4, cols 0-3 of A (1x4)
    # (1,1): row 4, col 4 of A (1x1)
    # Same for B and C.
    
    # For each (i_block, j_block, k_block) combination:
    # A[i_block, j_block] * B[j_block, k_block] -> C[i_block, k_block]
    
    # The 4x4 core (blocks 0,0 for A and B, 0,0 for C):
    # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Use 4 Strassen 2x2x2 decompositions:
    
    # 2x2x2 at (0,0,0)
    if not check_time():
        strassen_2x2x2(0, 0, 0)
    # 2x2x2 at (0,2,0) - A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    # Wait, I need to be more careful. The 4x4x4 core has indices 0-3 in all three dims.
    # I can partition it as:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  : Strassen at (0,0,0)
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]  : schoolbook 2*2*2=8
    # A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]  : schoolbook 2*2*2=8
    # A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]  : ...
    
    # This is getting complicated. Let me just use a cleaner decomposition.
    
    # Actually, the simplest correct approach:
    # Use Strassen for the 2x2x2 sub-blocks within the 4x4x4 core.
    # The 4x4x4 core can be seen as having 8 non-trivial block interactions.
    
    # Let me just do it block by block with the 4+1 partition:
    
    parts = [4, 1]
    offsets = [0, 4]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if check_time():
                    break
                
                size_a = parts[i]
                size_b = parts[j]
                size_c = parts[k]
                
                off_a = offsets[i]
                off_b = offsets[j]
                off_c = offsets[k]
                
                # We need to decompose A[off_a:off_a+size_a, off_b:off_b+size_b] * 
                # B[off_b:off_b+size_b, off_c:off_c+size_c] -> C[off_c:off_c+size_c, off_a:off_a+size_a]
                
                if size_a == 4 and size_b == 4 and size_c == 4:
                    # Use 4 Strassen 2x2x2 for the four 2x2x2 diagonal blocks
                    # plus schoolbook for the 4 off-diagonal 2x2x2 blocks
                    # Diagonal blocks: (0,0), (2,2) for each dimension
                    # Block (0,0,0): A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
                    strassen_2x2x2(0, 0, 0)
                    # Block (0,2,2): A[0:2,2:4]*B[2:4,2:4]->C[2:4,0:2]
                    strassen_2x2x2(0, 2, 2)
                    # Block (2,0,0): A[2:4,0:2]*B[0:2,0:2]->C[0:2,2:4]
                    strassen_2x2x2(2, 0, 0)
                    # Block (2,2,2): A[2:4,2:4]*B[2:4,2:4]->C[2:4,2:4]
                    strassen_2x2x2(2, 2, 2)
                    
                    # Off-diagonal blocks (schoolbook):
                    # A[0:2,0:2]*B[0:2,2:4]->C[2:4,0:2]
                    for u in range(2):
                        for v in range(2):
                            for w in range(2):
                                u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                                u_vec[(0+u)*5 + (0+v)] = 1
                                v_vec[(0+v)*5 + (2+w)] = 1
                                w_vec[(2+w)*5 + (0+u)] = 1
                                add_col(u_vec, v_vec, w_vec)
                    
                    # A[0:2,2:4]*B[2:4,0:2]->C[0:2,0:2]
                    for u in range(2):
                        for v in range(2):
                            for w in range(2):
                                u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                                u_vec[(0+u)*5 + (2+v)] = 1
                                v_vec[(2+v)*5 + (0+w)] = 1
                                w_vec[(0+w)*5 + (0+u)] = 1
                                add_col(u_vec, v_vec, w_vec)
                    
                    # A[2:4,0:2]*B[0:2,2:4]->C[2:4,2:4]
                    for u in range(2):
                        for v in range(2):
                            for w in range(2):
                                u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                                u_vec[(2+u)*5 + (0+v)] = 1
                                v_vec[(0+v)*5 + (2+w)] = 1
                                w_vec[(2+w)*5 + (2+u)] = 1
                                add_col(u_vec, v_vec, w_vec)
                    
                    # A[2:4,2:4]*B[2:4,0:2]->C[0:2,2:4]
                    for u in range(2):
                        for v in range(2):
                            for w in range(2):
                                u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                                u_vec[(2+u)*5 + (2+v)] = 1
                                v_vec[(2+v)*5 + (0+w)] = 1
                                w_vec[(0+w)*5 + (2+u)] = 1
                                add_col(u_vec, v_vec, w_vec)
                    
                elif size_a == 1 and size_b == 1 and size_c == 1:
                    # Single element
                    u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                    u_vec[off_a*5 + off_b] = 1
                    v_vec[off_b*5 + off_c] = 1
                    w_vec[off_c*5 + off_a] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                elif size_a == 4 and size_b == 1 and size_c == 1:
                    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
                    # Wait, this doesn't make sense dimensionally.
                    # A is 5x5, so A[i,j] where i is row, j is col.
                    # B is 5x5, B[j,k] where j is row, k is col.
                    # C is 5x5, C[i,k].
                    # The product term is sum_j A[i,j] * B[j,k] for C[i,k].
                    # For block (i_block=0, j_block=1, k_block=0):
                    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
                    # This is a 4x1 times 1x4 giving 4x4, so 16 terms
                    for u in range(4):
                        for w in range(4):
                            u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                            u_vec[(off_a+u)*5 + off_b] = 1
                            v_vec[off_b*5 + (off_c+w)] = 1
                            w_vec[(off_c+w)*5 + (off_a+u)] = 1
                            add_col(u_vec, v_vec, w_vec)
                            
                elif size_a == 4 and size_b == 4 and size_c == 1:
                    # A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
                    for u in range(4):
                        for v in range(4):
                            u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                            u_vec[(off_a+u)*5 + (off_b+v)] = 1
                            v_vec[(off_b+v)*5 + off_c] = 1
                            w_vec[off_c*5 + (off_a+u)] = 1
                            add_col(u_vec, v_vec, w_vec)
                            
                elif size_a == 1 and size_b == 4 and size_c == 4:
                    # A[4, 0:4] * B[0:4, 0:4] -> C[0:4, 4]
                    for v in range(4):
                        for w in range(4):
                            u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                            u_vec[off_a*5 + (off_b+v)] = 1
                            v_vec[(off_b+v)*5 + (off_c+w)] = 1
                            w_vec[(off_c+w)*5 + off_a] = 1
                            add_col(u_vec, v_vec, w_vec)
                            
                elif size_a == 4 and size_b == 1 and size_c == 4:
                    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 4]
                    for u in range(4):
                        for w in range(4):
                            u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                            u_vec[(off_a+u)*5 + off_b] = 1
                            v_vec[off_b*5 + (off_c+w)] = 1
                            w_vec[(off_c+w)*5 + (off_a+u)] = 1
                            add_col(u_vec, v_vec, w_vec)
                            
                elif size_a == 1 and size_b == 1 and size_c == 4:
                    # A[4, 4] * B[4, 0:4] -> C[0:4, 4]
                    for w in range(4):
                        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                        u_vec[off_a*5 + off_b] = 1
                        v_vec[off_b*5 + (off_c+w)] = 1
                        w_vec[(off_c+w)*5 + off_a] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                elif size_a == 1 and size_b == 4 and size_c == 1:
                    # A[4, 0:4] * B[0:4, 4] -> C[4, 4]
                    for v in range(4):
                        u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                        u_vec[off_a*5 + (off_b+v)] = 1
                        v_vec[(off_b+v)*5 + off_c] = 1
                        w_vec[off_c*5 + off_a] = 1
                        add_col(u_vec, v_vec, w_vec)
                        
                elif size_a == 4 and size_b == 4 and size_c == 4:
                    # Already handled above
                    pass
                else:
                    # General fallback: schoolbook
                    for u in range(size_a):
                        for v in range(size_b):
                            for w in range(size_c):
                                u_vec = make_vec(25); v_vec = make_vec(25); w_vec = make_vec(25)
                                u_vec[(off_a+u)*5 + (off_b+v)] = 1
                                v_vec[(off_b+v)*5 + (off_c+w)] = 1
                                w_vec[(off_c+w)*5 + (off_a+u)] = 1
                                add_col(u_vec, v_vec, w_vec)
    
    # Convert to row-major format
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        for row_idx in range(25):
            U[row_idx][col_idx] = U_cols[col_idx][row_idx]
            V[row_idx][col_idx] = V_cols[col_idx][row_idx]
            W[row_idx][col_idx] = W_cols[col_idx][row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
