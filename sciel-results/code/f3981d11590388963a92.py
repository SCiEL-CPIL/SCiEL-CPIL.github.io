"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 4
    
    # Use Strassen-like approach with block decomposition
    # A is 2x4, B is 4x4, C is 2x4
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For each 2x2 multiplication, use Strassen's algorithm (rank 7)
    # But we can share intermediate computations.
    
    # Strassen for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x4 * 4x4, we have 4 independent 2x2 multiplications in the output,
    # but they share the same input blocks.
    
    # Let's define the Strassen intermediates for each pair of 2x2 blocks:
    # For computing C_ij, we need products of A blocks with B blocks.
    
    # Actually, let's think of this differently. We have:
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # C[2:4, 0:2] = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:4]*B[2:4, 0:2]
    # C[2:4, 2:4] = A[2:4, 0:2]*B[0:2, 2:4] + A[2:4, 2:4]*B[2:4, 2:4]
    
    # Each 2x2*2x2 can be done with 7 multiplications (Strassen).
    # Total: 4*7 = 28 multiplications.
    
    # But we can potentially share some computations.
    # Let's first implement the 28-rank solution using Strassen blocks.
    
    # Define Strassen factors for a single 2x2 multiplication
    # For matrices X, Y (2x2), compute Z = X*Y using 7 products:
    # P1 = (X00+X11)*(Y00+Y11)
    # P2 = (X10+X11)*Y00
    # P3 = X00*(Y01-Y11)
    # P4 = X11*(Y10-Y00)
    # P5 = (X00+X01)*Y11
    # P6 = (X10-X00)*(Y00+Y01)
    # P7 = (X01-X11)*(Y10+Y11)
    #
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # For our case, we need to compute 4 such products:
    # C00 = A00*B00 + A01*B10  (this is a sum of two 2x2 products)
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Wait, that's not quite right. Let me reconsider.
    # A is 2x4, so A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # B is 4x4, so B = [[B_topleft, B_topright], [B_botleft, B_botright]]
    # C = A*B = [A_left*B_top + A_right*B_bot]
    # where B_top = [B_topleft, B_topright] (2x4) and B_bot = [B_botleft, B_botright] (2x4)
    
    # So:
    # C_left (2x2) = A_left * B_topleft + A_right * B_botleft
    # C_right (2x2) = A_left * B_topright + A_right * B_botright
    
    # Each of these is a sum of two 2x2*2x2 products.
    # We can use Strassen for each 2x2*2x2, giving 7*4 = 28 total.
    
    # Let's construct the 28-rank decomposition.
    # For each of the 4 sub-products, we use 7 Strassen terms.
    
    # Sub-product 1: A_left * B_topleft
    # Sub-product 2: A_right * B_botleft
    # Sub-product 3: A_left * B_topright
    # Sub-product 4: A_right * B_botright
    
    rank = 28
    U = np.zeros((n * m, rank))  # 8x28
    V = np.zeros((m * p, rank))  # 16x28
    W = np.zeros((p * n, rank))  # 8x28
    
    # Strassen coefficients for 2x2 multiplication
    # For X*Y = Z:
    # P1 = (X00+X11)*(Y00+Y11) -> contributes to Z00(+1), Z11(+1)
    # P2 = (X10+X11)*Y00       -> contributes to Z10(+1), Z11(-1)
    # P3 = X00*(Y01-Y11)       -> contributes to Z01(+1), Z11(+1)
    # P4 = X11*(Y10-Y00)       -> contributes to Z00(+1), Z10(+1)
    # P5 = (X00+X01)*Y11       -> contributes to Z00(-1), Z01(+1)
    # P6 = (X10-X00)*(Y00+Y01) -> contributes to Z11(+1)
    # P7 = (X01-X11)*(Y10+Y11) -> contributes to Z00(+1)
    
    # For each sub-product, define the Strassen factors
    # X indices in U: for A_left, rows 0-3 (0,1,2,3 correspond to A[0,0],A[0,1],A[1,0],A[1,1])
    # For A_right, rows 4-7 (A[0,2],A[0,3],A[1,2],A[1,3])
    
    # Y indices in V: for B_topleft, rows 0-3 (B[0,0],B[0,1],B[1,0],B[1,1])
    # For B_botleft, rows 4-7 (B[2,0],B[2,1],B[3,0],B[3,1])
    # For B_topright, rows 8-11 (B[0,2],B[0,3],B[1,2],B[1,3])
    # For B_botright, rows 12-15 (B[2,2],B[2,3],B[3,2],B[3,3])
    
    # Z indices in W: for C_left, rows 0-3 (C[0,0],C[0,1],C[1,0],C[1,1])
    # For C_right, rows 4-7 (C[0,2],C[0,3],C[1,2],C[1,3])
    
    # Define Strassen factor matrices for a single 2x2 product
    # X_factor[r, x_idx] where x_idx is 0..3 for X[0,0],X[0,1],X[1,0],X[1,1]
    # Y_factor[r, y_idx] where y_idx is 0..3 for Y[0,0],Y[0,1],Y[1,0],Y[1,1]
    # Z_factor[r, z_idx] where z_idx is 0..3 for Z[0,0],Z[0,1],Z[1,0],Z[1,1]
    
    def get_strassen_factors():
        """Return U, V, W factors for a single 2x2 Strassen multiplication."""
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # P1 = (X00+X11)*(Y00+Y11)
        U_s[0, 0] = 1; U_s[2, 0] = 1  # X00, X11
        V_s[0, 0] = 1; V_s[2, 0] = 1  # Y00, Y11
        W_s[0, 0] = 1; W_s[3, 0] = 1  # Z00, Z11
        
        # P2 = (X10+X11)*Y00
        U_s[1, 1] = 1; U_s[2, 1] = 1  # X10, X11
        V_s[0, 1] = 1                 # Y00
        W_s[1, 1] = 1; W_s[3, 1] = -1 # Z10, Z11
        
        # P3 = X00*(Y01-Y11)
        U_s[0, 2] = 1                 # X00
        V_s[1, 2] = 1; V_s[2, 2] = -1 # Y01, Y11
        W_s[0, 2] = 1; W_s[3, 2] = 1  # Z01, Z11
        
        # P4 = X11*(Y10-Y00)
        U_s[2, 3] = 1                 # X11
        V_s[1, 3] = -1; V_s[2, 3] = 1 # Y10, Y00
        W_s[0, 3] = 1; W_s[1, 3] = 1  # Z00, Z10
        
        # P5 = (X00+X01)*Y11
        U_s[0, 4] = 1; U_s[1, 4] = 1  # X00, X01
        V_s[2, 4] = 1                 # Y11
        W_s[0, 4] = -1; W_s[0, 4] = -1 # Z00, Z01 - wait, need to fix
        W_s[0, 4] = -1                # Z00
        W_s[0, 4] = -1                # Z01 - this is wrong, let me redo
        
        # Let me restart with correct indices
        # Z[0,0] is index 0, Z[0,1] is index 1, Z[1,0] is index 2, Z[1,1] is index 3
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # P1 = (X00+X11)*(Y00+Y11) -> Z00 += 1, Z11 += 1
        U_s[0, 0] = 1; U_s[3, 0] = 1  # X00, X11
        V_s[0, 0] = 1; V_s[3, 0] = 1  # Y00, Y11
        W_s[0, 0] = 1; W_s[3, 0] = 1  # Z00, Z11
        
        # P2 = (X10+X11)*Y00 -> Z10 += 1, Z11 -= 1
        U_s[2, 1] = 1; U_s[3, 1] = 1  # X10, X11
        V_s[0, 1] = 1                 # Y00
        W_s[2, 1] = 1; W_s[3, 1] = -1 # Z10, Z11
        
        # P3 = X00*(Y01-Y11) -> Z01 += 1, Z11 += 1
        U_s[0, 2] = 1                 # X00
        V_s[1, 2] = 1; V_s[3, 2] = -1 # Y01, Y11
        W_s[1, 2] = 1; W_s[3, 2] = 1  # Z01, Z11
        
        # P4 = X11*(Y10-Y00) -> Z00 += 1, Z10 += 1
        U_s[3, 3] = 1                 # X11
        V_s[2, 3] = 1; V_s[0, 3] = -1 # Y10, Y00
        W_s[0, 3] = 1; W_s[2, 3] = 1  # Z00, Z10
        
        # P5 = (X00+X01)*Y11 -> Z00 -= 1, Z01 += 1
        U_s[0, 4] = 1; U_s[1, 4] = 1  # X00, X01
        V_s[3, 4] = 1                 # Y11
        W_s[0, 4] = -1; W_s[1, 4] = 1 # Z00, Z01
        
        # P6 = (X10-X00)*(Y00+Y01) -> Z11 += 1
        U_s[2, 5] = 1; U_s[0, 5] = -1 # X10, X00
        V_s[0, 5] = 1; V_s[1, 5] = 1  # Y00, Y01
        W_s[3, 5] = 1                 # Z11
        
        # P7 = (X01-X11)*(Y10+Y11) -> Z00 += 1
        U_s[1, 6] = 1; U_s[3, 6] = -1 # X01, X11
        V_s[2, 6] = 1; V_s[3, 6] = 1  # Y10, Y11
        W_s[0, 6] = 1                 # Z00
        
        return U_s, V_s, W_s
    
    U_s, V_s, W_s = get_strassen_factors()
    
    # Now assemble the full 28-rank decomposition
    # Sub-product 1: A_left * B_topleft -> contributes to C_left
    # A_left rows in U: 0,1,2,3
    # B_topleft rows in V: 0,1,2,3
    # C_left rows in W: 0,1,2,3
    
    for r in range(7):
        col = r
        U[0:4, col] = U_s[:, r]
        V[0:4, col] = V_s[:, r]
        W[0:4, col] = W_s[:, r]
    
    # Sub-product 2: A_right * B_botleft -> contributes to C_left
    # A_right rows in U: 4,5,6,7
    # B_botleft rows in V: 4,5,6,7
    # C_left rows in W: 0,1,2,3
    
    for r in range(7):
        col = 7 + r
        U[4:8, col] = U_s[:, r]
        V[4:8, col] = V_s[:, r]
        W[0:4, col] = W_s[:, r]
    
    # Sub-product 3: A_left * B_topright -> contributes to C_right
    # A_left rows in U: 0,1,2,3
    # B_topright rows in V: 8,9,10,11
    # C_right rows in W: 4,5,6,7
    
    for r in range(7):
        col = 14 + r
        U[0:4, col] = U_s[:, r]
        V[8:12, col] = V_s[:, r]
        W[4:8, col] = W_s[:, r]
    
    # Sub-product 4: A_right * B_botright -> contributes to C_right
    # A_right rows in U: 4,5,6,7
    # B_botright rows in V: 12,13,14,15
    # C_right rows in W: 4,5,6,7
    
    for r in range(7):
        col = 21 + r
        U[4:8, col] = U_s[:, r]
        V[12:16, col] = V_s[:, r]
        W[4:8, col] = W_s[:, r]
    
    # Verify the decomposition
    # Check that sum_r U[:,r] * V[:,r] * W[:,r] gives the correct tensor
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # The multiplication tensor should satisfy:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # And T = 0 otherwise
    
    # Let's verify by checking a few entries
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                # This entry should be 1
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                if abs(val - 1) > 1e-10:
                    errors += 1
    
    # Also check that other entries are 0 (sample a few)
    for _ in range(100):
        a_idx = np.random.randint(0, n*m)
        b_idx = np.random.randint(0, m*p)
        c_idx = np.random.randint(0, p*n)
        # Check if this is a valid entry
        i = a_idx // m
        j = a_idx % m
        j2 = b_idx // p
        k = b_idx % p
        k2 = c_idx // n
        i2 = c_idx % n
        if i == i2 and j == j2 and k == k2:
            continue  # This is a valid entry, should be 1
        val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
        if abs(val) > 1e-10:
            errors += 1
    
    if errors > 0:
        # Fall back to schoolbook if verification fails
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
    else:
        # Success, use the 28-rank decomposition
        pass
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
