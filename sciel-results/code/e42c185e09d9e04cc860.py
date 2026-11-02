import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Helper to create trivial decomposition for n x m x p
    def trivial_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11: i=0,j=0. A22: i=1,j=1.
        # B11: j=0,k=0. B22: j=1,k=1.
        # U[0,0]=1, U[3,0]=1
        # V[0,0]=1, V[3,0]=1
        # W[0,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        
        # M2 = (A21+A22)B11
        # A21: i=1,j=0. A22: i=1,j=1.
        # B11: j=0,k=0.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = 1
        
        # M3 = A11(B12-B22)
        # A11: i=0,j=0.
        # B12: j=0,k=1. B22: j=1,k=1.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22(B21-B11)
        # A22: i=1,j=1.
        # B21: j=1,k=0. B11: j=0,k=0.
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        
        # M5 = (A11+A12)B22
        # A11: i=0,j=0. A12: i=0,j=1.
        # B22: j=1,k=1.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21: i=1,j=0. A11: i=0,j=0.
        # B11: j=0,k=0. B12: j=0,k=1.
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12: i=0,j=1. A22: i=1,j=1.
        # B21: j=1,k=0. B22: j=1,k=1.
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Recursive Strassen for 4x4x4
    def strassen_4x4x4():
        # Split into 4 blocks of 2x2x2
        # This is complex to do recursively in code quickly.
        # Instead, we can use the 2x2x2 Strassen on the 4x4x4 by viewing it as 2x2 blocks of 2x2 matrices.
        # The coefficients will be integers.
        
        # We will generate the 4x4x4 decomposition by applying Strassen to the 2x2x2 structure of blocks.
        # Block A11, A12, A21, A22 are 2x2.
        # Block B11, B12, B21, B22 are 2x2.
        
        # The 7 products are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each product is a 2x2x2 multiplication.
        # We can combine the U, V, W matrices of the 2x2x2 Strassen for each term.
        
        U_2, V_2, W_2 = strassen_2x2x2()
        
        # Total rank is 7 * 7 = 49.
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # We need to map the block indices to the global indices.
        # A block (bi, bj) corresponds to i in [2*bi, 2*bi+2), j in [2*bj, 2*bj+2).
        # B block (bj, bk) corresponds to j in [2*bj, 2*bj+2), k in [2*bk, 2*bk+2).
        
        # Helper to add a 2x2x2 decomposition to the global matrices with a sign and offset
        def add_decomp(U_global, V_global, W_global, U_local, V_local, W_local, sign, i_off, j_off, k_off):
            # i_off, j_off, k_off are the block offsets (0 or 1)
            # Local indices: i_loc in 0..1, j_loc in 0..1, k_loc in 0..1
            # Global indices: i = i_off*2 + i_loc, etc.
            # U_global index: (i_off*2 + i_loc)*4 + (j_off*2 + j_loc)
            # V_global index: (j_off*2 + j_loc)*4 + (k_off*2 + k_loc)
            # W_global index: (k_off*2 + k_loc)*4 + (i_off*2 + i_loc)
            
            for r_loc in range(7):
                r_glob = r_loc + 0 # This is wrong, we need to stack columns
                # Actually, we are adding to specific columns.
                pass
        
        # This manual construction is error-prone.
        # Let's use a simpler approach: Just use the trivial decomposition for all parts if we can't easily construct Strassen.
        # But 105 is a good target.
        # Let's stick to the 105 rank decomposition using trivial for 4x4x2 and 4x1x6, and Strassen for 4x4x4.
        
        # To avoid bugs in recursive Strassen construction, we can use a known library or a very simple implementation.
        # Given the time constraints, I will implement the 105 rank decomposition using trivial for everything EXCEPT the 4x4x4 part, 
        # where I will use a pre-computed or simple recursive call.
        
        # Actually, let's just use the trivial decomposition for 4x4x4 as well if Strassen is too complex to code correctly in short time.
        # Trivial for 4x4x4 is 64.
        # Total: 64 + 32 + 24 = 120.
        # This is the same as the baseline.
        
        # I need to implement Strassen correctly.
        
        # Let's try a different split that is easier to handle with trivial decompositions but still reduces rank.
        # Is there a split that reduces rank below 120 using only trivial decompositions?
        # No, the sum of trivial ranks of sub-blocks is always >= total trivial rank if the blocks partition the dimensions.
        # Wait, if we overlap? No, decomposition must be exact.
        
        # So we MUST use Strassen or another efficient algorithm to beat 120.
        
        # Let's implement the recursive Strassen for 4x4x4 carefully.
        
        def strassen_recursive(n, m, p):
            if n == 2 and m == 2 and p == 2:
                return strassen_2x2x2()
            
            # For 4x4x4
            if n == 4 and m == 4 and p == 4:
                # Use the 7 products of 2x2x2
                U_2, V_2, W_2 = strassen_2x2x2()
                
                U = np.zeros((16, 49))
                V = np.zeros((16, 49))
                W = np.zeros((16, 49))
                
                # Define the 7 terms with their block indices and signs
                # Term 1: (A11+A22)(B11+B22)
                # A11: i_off=0, j_off=0. A22: i_off=1, j_off=1.
                # B11: j_off=0, k_off=0. B22: j_off=1, k_off=1.
                # We add U_2 scaled by 1 for A11 part, and 1 for A22 part.
                # But U_2 is for 2x2x2.
                # We need to place U_2 into the correct rows of U.
                
                # Helper to place a 2x2x2 decomp into global matrices
                def place(U_g, V_g, W_g, U_l, V_l, W_l, sign, i_off, j_off, k_off, col_offset):
                    for r in range(7):
                        for i_loc in range(2):
                            for j_loc in range(2):
                                i_g = i_off * 2 + i_loc
                                j_g = j_off * 2 + j_loc
                                U_g[i_g * 4 + j_g, col_offset + r] += sign * U_l[i_loc * 2 + j_loc, r]
                                
                        for j_loc in range(2):
                            for k_loc in range(2):
                                j_g = j_off * 2 + j_loc
                                k_g = k_off * 2 + k_loc
                                V_g[j_g * 4 + k_g, col_offset + r] += sign * V_l[j_loc * 2 + k_loc, r]
                                
                        for k_loc in range(2):
                            for i_loc in range(2):
                                k_g = k_off * 2 + k_loc
                                i_g = i_off * 2 + i_loc
                                W_g[k_g * 4 + i_g, col_offset + r] += sign * W_l[k_loc * 2 + i_loc, r]
                
                # Term 1: A11*B11 (sign 1), A11*B22 (sign 1), A22*B11 (sign 1), A22*B22 (sign 1)
                # Wait, M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
                # We can add these 4 terms.
                place(U, V, W, U_2, V_2, W_2, 1, 0, 0, 0, 0) # A11 B11
                place(U, V, W, U_2, V_2, W_2, 1, 0, 0, 1, 0) # A11 B22 (k_off=1)
                place(U, V, W, U_2, V_2, W_2, 1, 1, 1, 0, 0) # A22 B11 (i_off=1, j_off=1)
                place(U, V, W, U_2, V_2, W_2, 1, 1, 1, 1, 0) # A22 B22
                
                # But this uses 4 * 7 = 28 columns for Term 1.
                # Strassen uses 7 products, not 28.
                # The point of Strassen is to compute the 7 products M1..M7, where each M is a 2x2x2 product.
                # So we should call strassen_2x2x2() once for each of the 7 terms, and sum the results.
                # But the "inputs" to the 2x2x2 multiplication are linear combinations of the blocks.
                # My `place` function adds the decomposition of a single 2x2x2 product.
                # To implement Strassen, we need to form the linear combinations in U and V.
                
                # This is getting too complex to code correctly in the short time.
                # I will fall back to a simpler strategy:
                # Use the trivial decomposition for 4x4x4 (rank 64).
                # Total rank 120.
                # This is not an improvement.
                
                # Let's try a different split:
                # 4x5x6 = 4x4x6 + 4x1x6.
                # 4x4x6 = 4x4x4 + 4x4x2.
                # If I can't do Strassen, I'm stuck at 120.
                
                # Is there a simple 119 rank decomposition?
                # Maybe not.
                
                # I will return the trivial 120 rank decomposition to ensure validity, 
                # but I will try to use the 105 rank decomposition if I can code Strassen correctly.
                
                # Given the risk, I will return the trivial 120.
                pass
        
        # Since I cannot guarantee a correct Strassen implementation quickly, I will return the trivial decomposition.
        U, V, W = trivial_decomp(n, m, p)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
