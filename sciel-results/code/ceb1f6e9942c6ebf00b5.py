"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Use 3 blocks of (4x4)*(4x3)
    # For each block, pad B to 4x4 and use Strassen 4x4x4 (rank 28 with schoolbook 2x2)
    # Total rank = 3 * 28 = 84
    
    # Let's build the Strassen 4x4x4 decomposition with schoolbook 2x2 base (rank 28)
    # Then for each 4x3 block, we use 28 terms but only 3 of the 4 output columns matter.
    # Actually, the rank is still 28 per block because the decomposition has 28 terms.
    # The 4th column of the output will be zero, but we still have 28 rank-1 terms.
    # Wait, can we reduce the rank? The 4th column being zero means some terms might be redundant.
    # But for now, let's just use 28 per block.
    
    # Strassen 4x4x4 with schoolbook 2x2:
    # Split 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, which with schoolbook is 4 rank-1 terms.
    # So 7 * 4 = 28 terms for the 4x4x4 product.
    
    # For the 4x4 * 4x3 case, B has 3 columns. We can pad B to 4x4 with a zero 4th column.
    # Then B12 and B22 have a 4th column that is zero.
    # This might allow reducing the rank, but for now let's just use the 28 terms.
    
    # Let's build the 28-term decomposition for one 4x4*4x3 block.
    # We'll create U, V, W for this block, then combine 3 blocks.
    
    def strassen_block_decomp(n_block=4, m_block=4, p_block=3):
        """Return U, V, W for 4x4 * 4x3 using Strassen with schoolbook 2x2."""
        # The 4x4 * 4x3 can be seen as 4x4 * 4x4 with the last column of B being zero.
        # Let's just build the 28-term Strassen 4x4x4 and note that the last column is zero.
        # U has shape (4*4, 28) = (16, 28)
        # V has shape (4*3, 28) = (12, 28) -- wait, V should be (m*p, rank) = (4*3, 28)
        # But in Strassen, B is 4x4. If we pad to 4x4, V is (4*4, 28) = (16, 28).
        # But we only need 3 columns of B. Let's think.
        # Actually, the tensor decomposition is for C = A * B where A is 4x4 and B is 4x3.
        # The standard form is: C[i,k] = sum_t U[i,t] * V[t,k] where U is (4*4, rank) and V is (4*3, rank)?
        # No, the convention is: C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * ... wait.
        
        # Let's use the convention from the seed:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # For n=4, m=4, p=3 (one block):
        # U has shape (4*4, rank) = (16, rank)
        # V has shape (4*3, rank) = (12, rank)
        # W has shape (3*4, rank) = (12, rank)
        
        # Strassen for 4x4 * 4x4 gives C[i,k] for i,k in 0..3.
        # We want C[i,k] for i in 0..3, k in 0..2.
        # So we can use the Strassen 4x4x4 decomposition and just ignore the k=3 output.
        # But the rank is still 28.
        
        # Let's build the 28-term decomposition.
        # Each of the 7 Strassen products is a 2x2 * 2x2 product, which has 4 rank-1 terms.
        # Total 28 terms.
        
        rank = 28
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(12)]  # 4*3
        W = [[0]*rank for _ in range(12)]  # 3*4
        
        # We need to map the Strassen terms to this convention.
        # Let's define the 7 Strassen products:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # Each Pq is a 2x2 matrix. Pq[r,c] = sum_{t} Uq[r,t] * Vq[c,t] where t indexes the 4 schoolbook terms.
        # For a 2x2 * 2x2 product:
        # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
        # Z = X*Y, Z[0,0] = x00*y00 + x01*y10
        # Z[0,1] = x00*y01 + x01*y11
        # Z[1,0] = x10*y00 + x11*y10
        # Z[1,1] = x10*y01 + x11*y11
        
        # So for each Strassen product, we have 4 rank-1 terms.
        # The U part for the 2x2 product: for term t, U has entries corresponding to the X entries.
        # For X = [[x00, x01], [x10, x11]], the 4 terms are:
        # t=0: x00 * y00 -> U entries: x00 at (0,0), 0 at (0,1), 0 at (1,0), 0 at (1,1)
        # t=1: x01 * y10 -> U entries: 0 at (0,0), x01 at (0,1), 0 at (1,0), 0 at (1,1)
        # t=2: x10 * y00 -> U entries: 0 at (0,0), 0 at (0,1), x10 at (1,0), 0 at (1,1)
        # t=3: x11 * y10 -> U entries: 0 at (0,0), 0 at (0,1), 0 at (1,0), x11 at (1,1)
        
        # Wait, that's not right. For Z[0,0] = x00*y00 + x01*y10, the terms are:
        # term 0: x00 * y00
        # term 1: x01 * y10
        # So U for term 0 has x00 at position (0,0) of the 2x2 block.
        # U for term 1 has x01 at position (0,1) of the 2x2 block.
        
        # Let's define a helper to add a 2x2 Strassen product.
        
        def add_2x2_product(base_col, X_coeffs, Y_coeffs, X_pos, Y_pos, Z_pos):
            """
            Add 4 rank-1 terms for a 2x2 * 2x2 product.
            X_coeffs: dict mapping (r,c) to coefficient for X[r,c]
            Y_coeffs: dict mapping (r,c) to coefficient for Y[r,c]
            X_pos: mapping from (r,c) to index in the full U array
            Y_pos: mapping from (r,c) to index in the full V array
            Z_pos: mapping from (r,c) to index in the full W array
            """
            # The 4 terms for 2x2*2x2:
            # term 0: X[0,0]*Y[0,0]
            # term 1: X[0,1]*Y[1,0]
            # term 2: X[1,0]*Y[0,0]
            # term 3: X[1,1]*Y[1,0]
            # Wait, Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
            # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
            # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
            # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
            
            # So the 4 rank-1 terms are:
            # t0: X[0,0] * Y[0,0], contributes to Z[0,0] and Z[0,1] (via Y[0,1])
            # Wait, no. Each rank-1 term is of the form u_i * v_j * w_k.
            # For the schoolbook 2x2*2x2, the 4 terms are:
            # t0: X[0,0]*Y[0,0] -> contributes to Z[0,0]
            # t1: X[0,0]*Y[0,1] -> contributes to Z[0,1]
            # t2: X[0,1]*Y[1,0] -> contributes to Z[0,0]
            # t3: X[0,1]*Y[1,1] -> contributes to Z[0,1]
            # t4: X[1,0]*Y[0,0] -> contributes to Z[1,0]
            # t5: X[1,0]*Y[0,1] -> contributes to Z[1,1]
            # t6: X[1,1]*Y[1,0] -> contributes to Z[1,0]
            # t7: X[1,1]*Y[1,1] -> contributes to Z[1,1]
            # That's 8 terms, not 4. I made a mistake.
            # The rank of 2x2*2x2 is 4, but the schoolbook decomposition has 4 terms?
            # No, schoolbook for 2x2*2x2 has 4 multiplications:
            # M1 = X[0,0]*Y[0,0]
            # M2 = X[0,0]*Y[0,1]
            # M3 = X[0,1]*Y[1,0]
            # M4 = X[0,1]*Y[1,1]
            # M5 = X[1,0]*Y[0,0]
            # M6 = X[1,0]*Y[0,1]
            # M7 = X[1,1]*Y[1,0]
            # M8 = X[1,1]*Y[1,1]
            # That's 8. But the rank of 2x2*2x2 is 4? No, the rank is 4 for 2x2*2x2?
            # Actually, the rank of the 2x2*2x2 tensor is 4. The schoolbook has 4 terms?
            # Let me check: Z = X*Y. Z[0,0] = X00*Y00 + X01*Y10
            # Z[0,1] = X00*Y01 + X01*Y11
            # Z[1,0] = X10*Y00 + X11*Y10
            # Z[1,1] = X10*Y01 + X11*Y11
            # The 4 rank-1 terms for Strassen-like? No, the rank of 2x2*2x2 is 4.
            # The schoolbook has 4 multiplications? No, it has 4 multiplications for 2x2?
            # 2x2 matrix multiplication: C = A*B
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            # That's 4 multiplications if we reuse? No, that's 4 distinct multiplications?
            # A00*B00, A01*B10, A00*B01, A01*B11, A10*B00, A11*B10, A10*B01, A11*B11
            # That's 8. But we can compute it with 4 multiplications? No, the rank is 4 for 2x2*2x2.
            # Actually, the rank of 2x2*2x2 is 4. The schoolbook decomposition has rank 4?
            # Let's just use the 4-term Strassen-like decomposition for 2x2*2x2? No, Strassen is for 2x2.
            # Strassen for 2x2*2x2 has 7 multiplications, which is worse than 4.
            # The schoolbook for 2x2*2x2 has 4 multiplications? No, it has 4 multiplications if we consider:
            # M1 = A00*B00
            # M2 = A01*B10
            # M3 = A00*B01
            # M4 = A01*B11
            # M5 = A10*B00
            # M6 = A11*B10
            # M7 = A10*B01
            # M8 = A11*B11
            # That's 8. But the rank is 4. So there must be a 4-term decomposition.
            # Actually, I think I'm overcomplicating. The rank of 2x2*2x2 is 4.
            # Let's just use the 4-term decomposition for 2x2*2x2.
            # But I don't know it. Let's just use the schoolbook with 4 terms? No, schoolbook is 4 multiplications?
            # I think the schoolbook for 2x2 has 4 multiplications. Let me count again.
            # C00 = A00*B00 + A01*B10: 2 multiplications
            # C01 = A00*B01 + A01*B11: 2 multiplications
            # C10 = A10*B00 + A11*B10: 2 multiplications
            # C11 = A10*B01 + A11*B11: 2 multiplications
            # Total 8 multiplications. But the rank is 4. So there's a better decomposition.
            # For now, let's just use the schoolbook for 2x2 with 4 terms? No, that's not correct.
            # Let's just use the 4-term decomposition for 2x2*2x2 that I know:
            # M1 = (A00+A01)(B00+B10) - A00*B10 - A01*B00
            # This is getting too complex. Let's just use the schoolbook for the entire 4x4*4x3.
            # Rank = 4*4*3 = 48 per block, total 144. Same as before.
            
            # Let's just use the schoolbook for the whole thing.
            pass
        
        # Let's just return schoolbook for now.
        rank = 4 * 4 * 3  # 48
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(12)]
        W = [[0]*rank for _ in range(12)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # For now, just use schoolbook for the whole 4x4*4x9.
    rank = 4 * 4 * 9  # 144
    U = [[0]*rank for _ in range(4*4)]
    V = [[0]*rank for _ in range(4*9)]
    W = [[0]*rank for _ in range(9*4)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                U[i*4+j][col] = 1
                V[j*9+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
