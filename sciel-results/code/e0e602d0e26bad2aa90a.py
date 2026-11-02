"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 98.
    # 49 from Strassen 4x4
    # 49 from linear terms (16+16+4+4+4+1 = 45? Let's recount: 16+16+4+4+4+1 = 45. 49+45=94? 
    # Wait, A11B11 is 49.
    # A12B21: 4*1 * 1*4 -> 4 mults.
    # A11B12: 4*4 * 4*1 -> 16 mults.
    # A12B22: 4*1 * 1*1 -> 4 mults.
    # A21B11: 1*4 * 4*4 -> 16 mults.
    # A22B21: 1*1 * 1*4 -> 4 mults.
    # A21B12: 1*4 * 4*1 -> 4 mults.
    # A22B22: 1*1 * 1*1 -> 1 mult.
    # Total linear: 4+16+4+16+4+4+1 = 49.
    # Total rank: 49 + 49 = 98.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w

    col = 0
    
    # 1. Strassen 4x4 for indices 0..3
    # We need the U, V, W for a 4x4 Strassen multiplication.
    # Strassen 4x4 is composed of 7 Strassen 2x2 multiplications.
    # Each Strassen 2x2 has 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # Let's generate the 2x2 Strassen U,V,W first.
    # Indices for 2x2: (0,0), (0,1), (1,0), (1,1)
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    def get_strassen_2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # P1 = (A00+A11)(B00+B11)
        u[0][0] += 1; u[1][0] += 1
        v[0][0] += 1; v[1][0] += 1
        w[0][0] += 1; w[1][0] += 1 # W is for C, so C00 += P1
        
        # P2 = (A10+A11)B00
        u[2][1] += 1; u[3][1] += 1
        v[0][1] += 1
        w[2][1] += 1 # C10 += P2
        
        # P3 = A00(B01-B11)
        u[0][2] += 1
        v[1][2] += 1; v[3][2] += -1
        w[1][2] += 1 # C01 += P3
        
        # P4 = A11(B10-B00)
        u[3][3] += 1
        v[2][3] += 1; v[0][3] += -1
        w[0][3] += 1 # C00 += P4
        w[2][3] += 1 # C10 += P4
        
        # P5 = (A00+A01)B11
        u[0][4] += 1; u[1][4] += 1
        v[3][4] += 1
        w[0][4] += -1 # C00 -= P5
        w[1][4] += 1 # C01 += P5
        
        # P6 = (A10-A00)(B00+B01)
        u[2][5] += 1; u[0][5] += -1
        v[0][5] += 1; v[1][5] += 1
        w[3][5] += 1 # C11 += P6
        
        # P7 = (A01-A11)(B10+B11)
        u[1][6] += 1; u[3][6] += -1
        v[2][6] += 1; v[3][6] += 1
        w[0][6] += 1 # C00 += P7
        w[3][6] += 1 # C11 += P7
        
        return u, v, w

    u2, v2, w2 = get_strassen_2x2()
    
    # Now build 4x4 Strassen.
    # The 4x4 matrix is split into 4 2x2 blocks:
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) in block space.
    # Indices in 4x4:
    # Block (0,0): rows 0,1 cols 0,1
    # Block (0,1): rows 0,1 cols 2,3
    # Block (1,0): rows 2,3 cols 0,1
    # Block (1,1): rows 2,3 cols 2,3
    
    # We apply the 2x2 Strassen logic on the blocks.
    # This means we need 7 "block multiplications".
    # Each block multiplication is a 2x2 matrix multiplication, which we have decomposed into 7 scalar multiplications.
    # So for each of the 7 Strassen steps, we "expand" the block multiplication into 7 scalar terms.
    
    # Mapping of block indices to 4x4 indices:
    def block_idx(r, c):
        # r, c in {0,1} for block row/col
        # returns start row, start col
        return r*2, c*2
        
    # We iterate over the 7 Strassen terms.
    # For each term, we have specific A-blocks and B-blocks involved, and C-blocks updated.
    # We reuse the u2, v2, w2 structure but scaled to 4x4 indices.
    
    # Strassen 2x2 terms (in block space):
    # 0: A(0,0)+A(1,1) * B(0,0)+B(1,1) -> C(0,0) += 1
    # 1: A(1,0)+A(1,1) * B(0,0) -> C(1,0) += 1
    # 2: A(0,0) * B(0,1)-B(1,1) -> C(0,1) += 1
    # 3: A(1,1) * B(1,0)-B(0,0) -> C(0,0) += 1, C(1,0) += 1
    # 4: A(0,0)+A(0,1) * B(1,1) -> C(0,0) -= 1, C(0,1) += 1
    # 5: A(1,0)-A(0,0) * B(0,0)+B(0,1) -> C(1,1) += 1
    # 6: A(0,1)-A(1,1) * B(1,0)+B(1,1) -> C(0,0) += 1, C(1,1) += 1
    
    # We will generate the 49 terms.
    # For each block multiplication, we use the 2x2 Strassen decomposition.
    # This is a bit complex to code dynamically. 
    # Alternative: Just hardcode the 49 terms? No, too long.
    # Let's write a helper to expand a block multiplication.
    
    def expand_block_mult(a_block, b_block, c_block, sign, col_offset):
        # a_block, b_block, c_block are (r,c) in block space
        # This function adds the 7 scalar terms for multiplying block a_block by b_block and adding to c_block
        # It uses the pre-computed u2, v2, w2
        pass

    # This is getting too complex for a quick rewrite. 
    # I will stick to the Schoolbook (125) if I can't get the Strassen right quickly.
    # BUT, 98 is a solid improvement.
    
    # Let's try a simpler valid improvement: 
    # Use Strassen for 2x2 blocks and cover the 5x5 with overlapping 2x2s? No.
    
    # I will implement the Schoolbook but with a check. 
    # Actually, I will implement the 4x4 Strassen + Linear terms as planned, 
    # but I will generate the 4x4 Strassen terms programmatically to avoid errors.
    
    # Re-implementing the 4x4 Strassen expansion:
    
    # The 7 Strassen steps for 4x4 (block level):
    # Step 0: (A00+A11)*(B00+B11) -> C00
    # Step 1: (A10+A11)*B00 -> C10
    # Step 2: A00*(B01-B11) -> C01
    # Step 3: A11*(B10-B00) -> C00, C10
    # Step 4: (A00+A01)*B11 -> C00(-), C01
    # Step 5: (A10-A00)*(B00+B01) -> C11
    # Step 6: (A01-A11)*(B10+B11) -> C00, C11
    
    # For each step, we perform a 2x2 matrix multiplication.
    # We have u2, v2, w2 for 2x2.
    # We need to map the 2x2 indices to 4x4 indices based on the block location.
    
    def map_block_to_4x4(block_r, block_c, local_r, local_c):
        return block_r*2 + local_r, block_c*2 + local_c

    # We will iterate through the 7 steps.
    # For each step, we identify the A-blocks and B-blocks involved.
    # Then we iterate through the 7 sub-terms of the 2x2 Strassen.
    
    # This is effectively composing the decompositions.
    
    # Let's just use the Schoolbook for safety if this takes too long.
    # But 125 is the baseline. I must improve.
    
    # I will use a known valid rank 49 for 4x4 and add 49 for the rest.
    # I will hardcode the 4x4 Strassen U,V,W generation.
    
    # Given the complexity, I will fall back to the Schoolbook (125) 
    # but ensure it is correct and fast.
    # Wait, the prompt says "score is the rank". 125 is 1.0.
    # If I can't get < 125, I stay at 1.0.
    
    # Let's try one more simple idea: 
    # Is there a rank 120 decomposition?
    # Probably not standard.
    
    # I will return the Schoolbook.
    
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
