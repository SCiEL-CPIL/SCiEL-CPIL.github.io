"""Tensor decomposition for 2x4 * 4x4 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x4 * 4x4."""
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # We use a decomposition based on treating the 2x4 matrix as two 2x2 blocks
    # and the 4x4 matrix as four 2x2 blocks.
    # The product C = A * B is a 2x4 matrix, split into two 2x2 blocks C_L, C_R.
    
    # A = [[A_11, A_12], [A_21, A_22]] where each A_ij is 2x2
    # B = [[B_11, B_12], [B_21, B_22]] where each B_ij is 2x2
    
    # C_L = A_11 * B_11 + A_12 * B_21
    # C_R = A_11 * B_12 + A_12 * B_22
    
    # We use Strassen's algorithm for each 2x2 * 2x2 multiplication.
    # Strassen uses 7 multiplications for 2x2 * 2x2.
    
    # Total: 4 sub-products, each using 7 Strassen mults = 28 rank
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen's 7 intermediate products for X * Y where X, Y are 2x2:
    # M1 = (X11 + X22) * (Y11 + Y22)
    # M2 = (X21 + X22) * Y12
    # M3 = X11 * Y21
    # M4 = X22 * Y22
    # M5 = (X11 + X12) * Y22
    # M6 = (X21 - X11) * (Y11 + Y12)
    # M7 = (X12 - X22) * (Y21 + Y22)
    #
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M2 + M5
    # Z21 = M3 + M6
    # Z22 = M1 - M2 + M3 + M4
    
    def add_strassen(x_block_row, y_block_col, z_block_col, start_rank):
        """Add Strassen decomposition for one 2x2*2x2 product.
        x_block_row: which 2-row block of A (0 or 1)
        y_block_col: which 2-col block of B (0 or 1)
        z_block_col: which 2-col block of C (0 or 1)
        start_rank: starting rank index
        """
        # A indices: row i (0,1), col j (0,3) -> A[i*m + j]
        # B indices: row j (0,3), col k (0,3) -> B[j*p + k]
        # C indices: row i (0,1), col k (0,3) -> C[k*n + i] (trace convention)
        
        # X = A[x_block_row * 2 : (x_block_row+1) * 2, :]
        # X11 = A[xbr*2, 0], X12 = A[xbr*2, 1], X21 = A[xbr*2+1, 0], X22 = A[xbr*2+1, 1]
        # Y = B[:, y_block_col * 2 : (ybc+1) * 2]
        # Y11 = B[0, ybc*2], Y12 = B[0, ybc*2+1], Y21 = B[1, ybc*2], Y22 = B[1, ybc*2+1]
        
        # For Strassen, X is 2x2 from A rows xbr*2, xbr*2+1 and cols 0,1
        # But wait, A is 2x4, so A has rows 0,1 and cols 0,1,2,3
        # We're splitting A into left and right halves? No.
        # Let me re-think.
        
        # A is 2x4. We can split A into A_left (2x2) and A_right (2x2).
        # A = [A_left | A_right]
        # B is 4x4. Split into B_top (2x4) and B_bottom (2x4), or into 2x2 blocks.
        # Let's split B into a 2x2 grid of 2x2 blocks:
        # B = [[B_11, B_12], [B_21, B_22]]
        
        # C = A * B = [A_left * B_top + A_right * B_bottom]
        # where B_top = [B_11, B_12] (2x4) and B_bottom = [B_21, B_22] (2x4)
        # C is 2x4, so C = [C_left | C_right] where C_left = A_left * B_11 + A_right * B_21
        # C_right = A_left * B_12 + A_right * B_22
        
        # Hmm, this is getting complex. Let me just do it directly.
        
        # Actually, let me reconsider. The simplest approach:
        # A is 2x4, B is 4x4, C is 2x4.
        # We can view this as: C = A * B
        # Split A into two 2x2 blocks: A_L = A[:, 0:2], A_R = A[:, 2:4]
        # Split B into four 2x2 blocks: B_11 = B[0:2, 0:2], B_12 = B[0:2, 2:4],
        #                                 B_21 = B[2:4, 0:2], B_22 = B[2:4, 2:4]
        # C = A_L * [B_11; B_21] + A_R * [B_12; B_22]
        # C is 2x4, split into C_L = C[:, 0:2] and C_R = C[:, 2:4]
        # C_L = A_L * B_11 + A_R * B_21
        # C_R = A_L * B_12 + A_R * B_22
        
        # Each of these is a sum of two 2x2 * 2x2 products.
        # Using Strassen for each 2x2*2x2, that's 7 mults per product.
        # 4 products * 7 = 28 rank.
        
        pass
    
    # Let me just hardcode the 28-rank decomposition directly.
    # I'll use the standard Strassen-based approach.
    
    # For each of the 4 sub-products (A_L*B_11, A_R*B_21, A_L*B_12, A_R*B_22),
    # apply Strassen's 7-mult algorithm.
    
    # A_L = A[:, 0:2], indices in U: row i, col 0 or 1 -> U[i*4 + 0], U[i*4 + 1]
    # A_R = A[:, 2:4], indices in U: row i, col 2 or 3 -> U[i*4 + 2], U[i*4 + 3]
    # B_11 = B[0:2, 0:2], indices in V: row 0 or 1, col 0 or 1 -> V[0*4+0], V[0*4+1], V[1*4+0], V[1*4+1]
    # B_12 = B[0:2, 2:4], indices in V: row 0 or 1, col 2 or 3 -> V[0*4+2], V[0*4+3], V[1*4+2], V[1*4+3]
    # B_21 = B[2:4, 0:2], indices in V: row 2 or 3, col 0 or 1 -> V[2*4+0], V[2*4+1], V[3*4+0], V[3*4+1]
    # B_22 = B[2:4, 2:4], indices in V: row 2 or 3, col 2 or 3 -> V[2*4+2], V[2*4+3], V[3*4+2], V[3*4+3]
    # C_L = C[:, 0:2], indices in W: col 0 or 1 -> W[k*2 + i] for k=0,1
    # C_R = C[:, 2:4], indices in W: col 2 or 3 -> W[k*2 + i] for k=2,3
    
    # Strassen for X*Y where X, Y are 2x2:
    # Let X have entries X00, X01, X10, X11 and Y have Y00, Y01, Y10, Y11
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y01
    # M3 = X00*Y10
    # M4 = X11*Y11
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M2 + M5
    # Z10 = M3 + M6
    # Z11 = M1 - M2 + M3 + M4
    
    def strassen_decomp(x_idx, y_idx, z_idx, start):
        """Add Strassen factors for one 2x2*2x2 product.
        x_idx: list of 4 U indices for X entries [X00, X01, X10, X11]
        y_idx: list of 4 V indices for Y entries [Y00, Y01, Y10, Y11]
        z_idx: list of 4 W indices for Z entries [Z00, Z01, Z10, Z11]
        start: starting rank index
        Returns the next rank index.
        """
        # M1 = (X00+X11)(Y00+Y11)
        r = start
        U[x_idx[0]][r] += 1  # X00
        U[x_idx[2]][r] += 1  # X11
        V[y_idx[0]][r] += 1  # Y00
        V[y_idx[2]][r] += 1  # Y11
        W[z_idx[0]][r] += 1  # Z00 += M1
        W[z_idx[3]][r] += 1  # Z11 += M1
        r += 1
        
        # M2 = (X10+X11)Y01
        U[x_idx[1]][r] += 1  # X10
        U[x_idx[2]][r] += 1  # X11
        V[y_idx[1]][r] += 1  # Y01
        W[z_idx[1]][r] += 1  # Z01 += M2
        W[z_idx[3]][r] -= 1  # Z11 -= M2
        r += 1
        
        # M3 = X00*Y10
        U[x_idx[0]][r] += 1  # X00
        V[y_idx[2]][r] += 1  # Y10
        W[z_idx[2]][r] += 1  # Z10 += M3
        W[z_idx[3]][r] += 1  # Z11 += M3
        r += 1
        
        # M4 = X11*Y11
        U[x_idx[2]][r] += 1  # X11
        V[y_idx[3]][r] += 1  # Y11
        W[z_idx[0]][r] += 1  # Z00 += M4
        W[z_idx[3]][r] += 1  # Z11 += M4
        r += 1
        
        # M5 = (X00+X01)Y11
        U[x_idx[0]][r] += 1  # X00
        U[x_idx[1]][r] += 1  # X01
        V[y_idx[3]][r] += 1  # Y11
        W[z_idx[0]][r] -= 1  # Z00 -= M5
        W[z_idx[1]][r] += 1  # Z01 += M5
        r += 1
        
        # M6 = (X10-X00)(Y00+Y01)
        U[x_idx[1]][r] += 1  # X10
        U[x_idx[0]][r] -= 1  # -X00
        V[y_idx[0]][r] += 1  # Y00
        V[y_idx[1]][r] += 1  # Y01
        W[z_idx[2]][r] += 1  # Z10 += M6
        r += 1
        
        # M7 = (X01-X11)(Y10+Y11)
        U[x_idx[1]][r] += 1  # X01
        U[x_idx[2]][r] -= 1  # -X11
        V[y_idx[2]][r] += 1  # Y10
        V[y_idx[3]][r] += 1  # Y11
        W[z_idx[0]][r] += 1  # Z00 += M7
        r += 1
        
        return r
    
    # Now set up the four sub-products:
    # Product 1: A_L * B_11 -> contributes to C_L
    # A_L entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0, 1, 4, 5
    x1 = [0, 1, 4, 5]  # X00, X01, X10, X11
    # B_11 entries: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices: 0, 1, 4, 5
    y1 = [0, 1, 4, 5]  # Y00, Y01, Y10, Y11
    # C_L entries: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: W[k*2+i]
    # C[0,0] -> k=0,i=0 -> W[0]
    # C[0,1] -> k=1,i=0 -> W[2]
    # C[1,0] -> k=0,i=1 -> W[1]
    # C[1,1] -> k=1,i=1 -> W[3]
    z1 = [0, 2, 1, 3]  # Z00, Z01, Z10, Z11
    
    r = strassen_decomp(x1, y1, z1, 0)
    
    # Product 2: A_R * B_21 -> contributes to C_L
    # A_R entries: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices: 2, 3, 6, 7
    x2 = [2, 3, 6, 7]
    # B_21 entries: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices: 8, 9, 12, 13
    y2 = [8, 9, 12, 13]
    # C_L same as before
    z2 = [0, 2, 1, 3]
    
    r = strassen_decomp(x2, y2, z2, r)
    
    # Product 3: A_L * B_12 -> contributes to C_R
    x3 = [0, 1, 4, 5]
    # B_12 entries: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices: 2, 3, 6, 7
    y3 = [2, 3, 6, 7]
    # C_R entries: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices:
    # C[0,2] -> k=2,i=0 -> W[4]
    # C[0,3] -> k=3,i=0 -> W[6]
    # C[1,2] -> k=2,i=1 -> W[5]
    # C[1,3] -> k=3,i=1 -> W[7]
    z3 = [4, 6, 5, 7]
    
    r = strassen_decomp(x3, y3, z3, r)
    
    # Product 4: A_R * B_22 -> contributes to C_R
    x4 = [2, 3, 6, 7]
    # B_22 entries: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices: 10, 11, 14, 15
    y4 = [10, 11, 14, 15]
    z4 = [4, 6, 5, 7]
    
    r = strassen_decomp(x4, y4, z4, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
