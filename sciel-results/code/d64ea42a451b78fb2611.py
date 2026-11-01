import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # U[i*4+j, r] - A indices
    # V[j*4+k, r] - B indices  
    # W[k*4+i, r] - C indices (trace convention: W[k*n+i, r])
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    def get_U_index(i, j):
        return i * m + j
    
    def get_V_index(j, k):
        return j * p + k
    
    def get_W_index(k, i):
        return k * n + i
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we split into 2x2 blocks of 2x2 submatrices
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = [[C11, C12], [C21, C22]] where each Ck is 2x2
    #
    # Each 2x2 block multiplication uses 7 scalar multiplications
    # Total: 7 * 7 = 49 multiplications
    
    # Define the 7 Strassen terms for each block multiplication
    # For block (bi, bj) in A and block (bk, bl) in B:
    # The 7 products are indexed by t in 0..6
    
    def strassen_A_coeffs(block_row, block_col):
        """Return dict mapping (i,j) -> coeff for each of the 7 Strassen products
        block_row, block_col: which 2x2 block of A (0 or 1 for row/col)
        """
        coeffs = [{} for _ in range(7)]
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        br, bc = block_row, block_col
        
        # M1: A11 + A22
        coeffs[0][(0,0)] = 1
        coeffs[0][(1,1)] = 1
        # M2: A21 + A22
        coeffs[1][(1,0)] = 1
        coeffs[1][(1,1)] = 1
        # M3: A11
        coeffs[2][(0,0)] = 1
        # M4: A22
        coeffs[3][(1,1)] = 1
        # M5: A11 + A12
        coeffs[4][(0,0)] = 1
        coeffs[4][(0,1)] = 1
        # M6: A21 - A11
        coeffs[5][(1,0)] = 1
        coeffs[5][(0,0)] = -1
        # M7: A12 - A22
        coeffs[6][(0,1)] = 1
        coeffs[6][(1,1)] = -1
        
        return coeffs
    
    def strassen_B_coeffs(block_row, block_col):
        """Return dict mapping (j,k) -> coeff for each of the 7 Strassen products
        block_row, block_col: which 2x2 block of B (0 or 1 for row/col)
        """
        coeffs = [{} for _ in range(7)]
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        br, bc = block_row, block_col
        
        # M1: B11 + B22
        coeffs[0][(0,0)] = 1
        coeffs[0][(1,1)] = 1
        # M2: B11
        coeffs[1][(0,0)] = 1
        # M3: B12 - B22
        coeffs[2][(0,1)] = 1
        coeffs[2][(1,1)] = -1
        # M4: B21 - B11
        coeffs[3][(1,0)] = 1
        coeffs[3][(0,0)] = -1
        # M5: B22
        coeffs[4][(1,1)] = 1
        # M6: B11 + B12
        coeffs[5][(0,0)] = 1
        coeffs[5][(0,1)] = 1
        # M7: B21 + B22
        coeffs[6][(1,0)] = 1
        coeffs[6][(1,1)] = 1
        
        return coeffs
    
    # C block contributions from each Strassen product
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    c_contrib = [
        {(0,0): 1, (1,1): 1},       # M1 -> C11, C22
        {(1,0): 1, (1,1): -1},      # M2 -> C21, C22
        {(0,1): 1, (1,1): 1},       # M3 -> C12, C22
        {(0,0): 1, (1,0): 1},       # M4 -> C11, C21
        {(0,0): -1, (0,1): 1},      # M5 -> C11, C12
        {(1,1): 1},                 # M6 -> C22
        {(0,0): 1},                 # M7 -> C11
    ]
    
    # Now build the 4x4 decomposition
    # The 4x4 matrix A has blocks A[br][bc] for br,bc in {0,1}
    # Each block is 2x2 with elements A[br*2+i][bc*2+j] for i,j in {0,1}
    
    # For each pair of blocks (br, bc) in A and (bk, bl) in B:
    # We need to compute the contribution to C blocks
    # C[br][bl] gets contributions from summing over bk
    # But in Strassen, we group by the 7 products
    
    # Actually, let's think more carefully:
    # C = A * B
    # C[br][bl] = sum_{bk} A[br][bk] * B[bk][bl]
    # 
    # Using Strassen at the block level:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Where M1 = (A11+A22)(B11+B22), etc.
    # Each M is a 2x2 matrix product of 2x2 matrices
    # So each M requires 7 scalar multiplications (Strassen again)
    
    # Total rank = 4 * 7 * 7 / ... no wait
    # There are 4 block products in the Strassen formula, but they share M's
    # Actually there are 7 M's, each is a 2x2 matrix product
    # Each 2x2 matrix product has rank 7
    # Total rank = 7 * 7 = 49
    
    # For each of the 7 Strassen products M_t:
    # M_t = X_t * Y_t where X_t and Y_t are 2x2 matrices
    # X_t is a linear combination of A blocks
    # Y_t is a linear combination of B blocks
    
    # Let's define for each t in 0..6:
    # X_t[br][bc] = sum of coefficients for A[br][bc]
    # Y_t[bk][bl] = sum of coefficients for B[bk][bl]
    
    # From Strassen:
    # M1 = (A11+A22)(B11+B22)
    #   X1[0][0] = 1, X1[1][1] = 1
    #   Y1[0][0] = 1, Y1[1][1] = 1
    # M2 = (A21+A22)B11
    #   X2[1][0] = 1, X2[1][1] = 1
    #   Y2[0][0] = 1
    # M3 = A11(B12-B22)
    #   X3[0][0] = 1
    #   Y3[0][1] = 1, Y3[1][1] = -1
    # M4 = A22(B21-B11)
    #   X4[1][1] = 1
    #   Y4[1][0] = 1, Y4[0][0] = -1
    # M5 = (A11+A12)B22
    #   X5[0][0] = 1, X5[0][1] = 1
    #   Y5[1][1] = 1
    # M6 = (A21-A11)(B11+B12)
    #   X6[1][0] = 1, X6[0][0] = -1
    #   Y6[0][0] = 1, Y6[0][1] = 1
    # M7 = (A12-A22)(B21+B22)
    #   X7[0][1] = 1, X7[1][1] = -1
    #   Y7[1][0] = 1, Y7[1][1] = 1
    
    X_coeffs = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): 1},      # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},      # M5
        {(1,0): 1, (0,0): -1},     # M6
        {(0,1): 1, (1,1): -1},     # M7
    ]
    
    Y_coeffs = [
        {(0,0): 1, (1,1): 1},      # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},     # M3
        {(1,0): 1, (0,0): -1},     # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},      # M6
        {(1,0): 1, (1,1): 1},      # M7
    ]
    
    # Now for each t in 0..6, X_t * Y_t is a 2x2 matrix product
    # We decompose this 2x2 product using Strassen (rank 7)
    # The result contributes to C blocks according to c_contrib[t]
    
    # For the inner Strassen (computing X_t * Y_t):
    # X_t is a 2x2 matrix with elements X_t[br][bc] for br,bc in {0,1}
    # Y_t is a 2x2 matrix with elements Y_t[bk][bl] for bk,bl in {0,1}
    # Each element is a scalar (coefficient)
    
    # To compute X_t * Y_t = Z_t where Z_t[br][bl] = sum_{bk} X_t[br][bk] * Y_t[bk][bl]
    # We use Strassen on the 2x2 matrices
    
    # For the inner Strassen, the "matrices" are actually coefficient matrices
    # X_t[br][bk] and Y_t[bk][bl] are the coefficients
    
    # The 7 inner Strassen products for X_t * Y_t:
    # N1 = (X11+X22)(Y11+Y22)
    # N2 = (X21+X22)Y11
    # N3 = X11(Y12-Y22)
    # N4 = X22(Y21-Y11)
    # N5 = (X11+X12)Y22
    # N6 = (X21-X11)(Y11+Y12)
    # N7 = (X12-X22)(Y21+Y22)
    
    # Z11 = N1 + N4 - N5 + N7
    # Z12 = N3 + N5
    # Z21 = N2 + N4
    # Z22 = N1 - N2 + N3 + N6
    
    # For each t (outer Strassen product) and s (inner Strassen product):
    # We have a rank-1 term
    # U part: linear combination of A elements
    # V part: linear combination of B elements
    # W part: linear combination of C elements
    
    # Let's build it step by step
    
    for t in range(7):
        # X_t coefficients: X_t[br][bk] for br,bk in {0,1}
        X_t = {}
        for (br, bk), coeff in X_coeffs[t].items():
            X_t[(br, bk)] = coeff
        
        # Y_t coefficients: Y_t[bk][bl] for bk,bl in {0,1}
        Y_t = {}
        for (bk, bl), coeff in Y_coeffs[t].items():
            Y_t[(bk, bl)] = coeff
        
        # Inner Strassen on X_t * Y_t
        for s in range(7):
            # Determine the coefficients for the s-th inner Strassen product
            # N_s = f_s(X_t) * g_s(Y_t)
            
            # f_s(X_t) is a 2x2 matrix (coefficients for X_t elements)
            # g_s(Y_t) is a 2x2 matrix (coefficients for Y_t elements)
            
            if s == 0:  # N1 = (X11+X22)(Y11+Y22)
                f_s = {(0,0): 1, (1,1): 1}
                g_s = {(0,0): 1, (1,1): 1}
            elif s == 1:  # N2 = (X21+X22)Y11
                f_s = {(1,0): 1, (1,1): 1}
                g_s = {(0,0): 1}
            elif s == 2:  # N3 = X11(Y12-Y22)
                f_s = {(0,0): 1}
                g_s = {(0,1): 1, (1,1): -1}
            elif s == 3:  # N4 = X22(Y21-Y11)
                f_s = {(1,1): 1}
                g_s = {(1,0): 1, (0,0): -1}
            elif s == 4:  # N5 = (X11+X12)Y22
                f_s = {(0,0): 1, (0,1): 1}
                g_s = {(1,1): 1}
            elif s == 5:  # N6 = (X21-X11)(Y11+Y12)
                f_s = {(1,0): 1, (0,0): -1}
                g_s = {(0,0): 1, (0,1): 1}
            else:  # s == 6: N7 = (X12-X22)(Y21+Y22)
                f_s = {(0,1): 1, (1,1): -1}
                g_s = {(1,0): 1, (1,1): 1}
            
            # Now compute the actual coefficients for A, B, C elements
            # U coefficients: for each (i,j) in A, what's the coefficient?
            # i = br*2 + ir, j = bk*2 + jr where ir,jr in {0,1}
            
            U_coeffs = {}
            for (br, bk), f_coeff in f_s.items():
                if (br, bk) in X_t:
                    x_coeff = X_t[(br, bk)]
                    total_coeff = f_coeff * x_coeff
                    for ir in range(2):
                        for jr in range(2):
                            i = br * 2 + ir
                            j = bk * 2 + jr
                            U_coeffs[(i, j)] = U_coeffs.get((i, j), 0) + total_coeff
            
            V_coeffs = {}
            for (bk, bl), g_coeff in g_s.items():
                if (bk, bl) in Y_t:
                    y_coeff = Y_t[(bk, bl)]
                    total_coeff = g_coeff * y_coeff
                    for ik in range(2):
                        for jl in range(2):
                            k = bk * 2 + ik
                            l = bl * 2 + jl
                            V_coeffs[(k, l)] = V_coeffs.get((k, l), 0) + total_coeff
            
            # W coefficients: C[br][bl] gets contribution from N_s
            # Z[br][bl] = sum of N_s contributions
            # Z11 = N1 + N4 - N5 + N7
            # Z12 = N3 + N5
            # Z21 = N2 + N4
            # Z22 = N1 - N2 + N3 + N6
            
            W_coeffs = {}
            if s == 0:  # N1 -> Z11, Z22
                for (br, bl), c_coeff in c_contrib[t].items():
                    if (br, bl) in [(0,0), (1,1)]:
                        w_coeff = 1 * c_coeff
                        for ir in range(2):
                            for il in range(2):
                                i = br * 2 + ir
                                l = bl * 2 + il
                                W_coeffs[(i, l)] = W_coeffs.get((i, l), 0) + w_coeff
            elif s == 1:  # N2 -> Z21, Z22
                for (br, bl), c_coeff in c_contrib[t].items():
                    if (br, bl) in [(1,0), (1,1)]:
                        w_coeff = (1 if (br,bl) == (1,0) else -1) * c_coeff
                        for ir in range(2):
                            for il in range(2):
                                i = br * 2 + ir
                                l = bl * 2 + il
                                W_coeffs[(i, l)] = W_coeffs.get((i, l), 0) + w_coeff
            elif s == 2:  # N3 -> Z12, Z22
                for (br, bl), c_coeff in c_contrib[t].items():
                    if (br, bl) in [(0,1), (1,1)]:
                        w_coeff = c_coeff
                        for ir in range(2):
                            for il in range(2):
                                i = br * 2 + ir
                                l = bl * 2 + il
                                W_coeffs[(i, l)] = W_coeffs.get((i, l), 0) + w_coeff
            elif s == 3:  # N4 -> Z11, Z21
                for (br, bl), c_coeff in c_contrib[t].items():
                    if (br, bl) in [(0,0), (1,0)]:
                        w_coeff = c_coeff
                        for ir in range(2):
                            for il in range(2):
                                i = br * 2 + ir
                                l = bl * 2 + il
                                W_coeffs[(i, l)] = W_coeffs.get((i, l), 0) + w_coeff
            elif s == 4:  # N5 -> Z11, Z12
                for (br, bl), c_coeff in c_contrib[t].items():
                    if (br, bl) == (0,0):
                        w_coeff = -1 * c_coeff
                    elif (br, bl) == (0,1):
                        w_coeff = c_coeff
                    else:
                        continue
                    for ir in range(2):
                        for il in range(2):
                            i = br * 2 + ir
                            l = bl * 2 + il
                            W_coeffs[(i, l)] = W_coeffs.get((i, l), 0) + w_coeff
            elif s == 5:  # N6 -> Z22
                for (br, bl), c_coeff in c_contrib[t].items():
                    if (br, bl) == (1,1):
                        w_coeff = c_coeff
                        for ir in range(2):
                            for il in range(2):
                                i = br * 2 + ir
                                l = bl * 2 + il
                                W_coeffs[(i, l)] = W_coeffs.get((i, l), 0) + w_coeff
            else:  # s == 6: N7 -> Z11
                for (br, bl), c_coeff in c_contrib[t].items():
                    if (br, bl) == (0,0):
                        w_coeff = c_coeff
                        for ir in range(2):
                            for il in range(2):
                                i = br * 2 + ir
                                l = bl * 2 + il
                                W_coeffs[(i, l)] = W_coeffs.get((i, l), 0) + w_coeff
            
            # Assign to rank index
            rank_idx = t * 7 + s
            
            # Fill U
            for (i, j), coeff in U_coeffs.items():
                u_idx = get_U_index(i, j)
                U[u_idx][rank_idx] += coeff
            
            # Fill V
            for (k, l), coeff in V_coeffs.items():
                v_idx = get_V_index(k, l)
                V[v_idx][rank_idx] += coeff
            
            # Fill W
            for (i, l), coeff in W_coeffs.items():
                w_idx = get_W_index(l, i)  # W[k*n+i, r] where k=l, i=i
                W[w_idx][rank_idx] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
