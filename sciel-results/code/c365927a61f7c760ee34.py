import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Construct a rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    The approach splits the 2x4 matrix A into two 2x2 blocks [A1, A2] (columns 0:2, 2:4)
    and the 4x4 matrix B into four 2x2 blocks:
        B = [B11 B12]
            [B21 B22]
    where B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4].
    
    Then C = A * B where:
        C[:, 0:2] = A1*B11 + A2*B21
        C[:, 2:4] = A1*B12 + A2*B22
    
    Each 2x2 * 2x2 product uses Strassen's algorithm with rank 7.
    Total rank = 4 * 7 = 28.
    
    U has shape (8, 28): U[i*m + j, r] where i in [0,2), j in [0,4)
    V has shape (16, 28): V[j*p + k, r] where j in [0,4), k in [0,4)
    W has shape (8, 28): W[k*n + i, r] where k in [0,4), i in [0,2)
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen formulas for 2x2 * 2x2:
    # Given X (2x2) and Y (2x2):
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y10 - Y11)
    # M4 = X11 * (Y01 + Y11)
    # M5 = (X00 + X10) * Y11
    # M6 = (X10 - X00) * (Y00 + Y10)
    # M7 = (X01 - X11) * (Y11 + Y01)
    # 
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each of the 4 block products, we assign 7 columns to U, V, W.
    # The 4 products are:
    # Product 1: A1 * B11 -> contributes to C[:, 0:2]
    # Product 2: A2 * B21 -> contributes to C[:, 0:2]
    # Product 3: A1 * B12 -> contributes to C[:, 2:4]
    # Product 4: A2 * B22 -> contributes to C[:, 2:4]
    
    # For each product, we need to define:
    # - Which entries of A (U) and B (V) are used in each Strassen term
    # - Which entries of C (W) receive contributions from each Strassen term
    
    # Let's define the Strassen coefficients:
    # For term t in {0..6}, the U coefficients for X entries (X00, X01, X10, X11):
    # M1: X00=1, X11=1, others 0; Y00=1, Y11=1, others 0
    # M2: X10=1, X11=1, others 0; Y00=1, others 0
    # M3: X00=1, others 0; Y10=1, Y11=-1
    # M4: X11=1, others 0; Y01=1, Y11=1
    # M5: X00=1, X10=1, others 0; Y11=1, others 0
    # M6: X10=1, X00=-1, others 0; Y00=1, Y10=1
    # M7: X01=1, X11=-1, others 0; Y11=1, Y01=1
    
    # And the W coefficients for Z entries (Z00, Z01, Z10, Z11):
    # M1: Z00=1, Z11=1, others 0
    # M2: Z10=1, Z11=-1, others 0
    # M3: Z01=1, Z11=1, others 0
    # M4: Z00=1, Z10=1, others 0
    # M5: Z00=-1, Z01=1, others 0
    # M6: Z11=1, others 0
    # M7: Z00=1, others 0
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: X00=1, X01=0, X10=0, X11=1
        [0, 0, 1, 1],   # M2: X00=0, X01=0, X10=1, X11=1
        [1, 0, 0, 0],   # M3: X00=1, X01=0, X10=0, X11=0
        [0, 0, 0, 1],   # M4: X00=0, X01=0, X10=0, X11=1
        [1, 0, 1, 0],   # M5: X00=1, X01=0, X10=1, X11=0
        [-1, 0, 1, 0],  # M6: X00=-1, X01=0, X10=1, X11=0
        [0, 1, 0, -1],  # M7: X00=0, X01=1, X10=0, X11=-1
    ]
    
    strassen_V = [
        [1, 0, 0, 1],   # M1: Y00=1, Y01=0, Y10=0, Y11=1
        [1, 0, 0, 0],   # M2: Y00=1, Y01=0, Y10=0, Y11=0
        [0, 0, 1, -1],  # M3: Y00=0, Y01=0, Y10=1, Y11=-1
        [0, 1, 0, 1],   # M4: Y00=0, Y01=1, Y10=0, Y11=1
        [0, 0, 0, 1],   # M5: Y00=0, Y01=0, Y10=0, Y11=1
        [1, 0, 1, 0],   # M6: Y00=1, Y01=0, Y10=1, Y11=0
        [0, 1, 0, 1],   # M7: Y00=0, Y01=1, Y10=0, Y11=1
    ]
    
    strassen_W = [
        [1, 0, 0, 1],   # M1: Z00=1, Z01=0, Z10=0, Z11=1
        [0, 0, 1, -1],  # M2: Z00=0, Z01=0, Z10=1, Z11=-1
        [0, 1, 0, 1],   # M3: Z00=0, Z01=1, Z10=0, Z11=1
        [1, 0, 1, 0],   # M4: Z00=1, Z01=0, Z10=1, Z11=0
        [-1, 1, 0, 0],  # M5: Z00=-1, Z01=1, Z10=0, Z11=0
        [0, 0, 0, 1],   # M6: Z00=0, Z01=0, Z10=0, Z11=1
        [1, 0, 0, 0],   # M7: Z00=1, Z01=0, Z10=0, Z11=0
    ]
    
    # The 4 block products with their mappings:
    # Product 1: A1 * B11
    #   A1 = A[:, 0:2], so A[i, j] for i in [0,2), j in [0,2)
    #   U index for A[i,j] = i*4 + j
    #   B11 = B[0:2, 0:2], so B[j, k] for j in [0,2), k in [0,2)
    #   V index for B[j,k] = j*4 + k
    #   Z1 = P1 = C[:, 0:2], so C[i, k] for i in [0,2), k in [0,2)
    #   W index for C[i,k] = k*2 + i
    #
    # Product 2: A2 * B21
    #   A2 = A[:, 2:4], so A[i, j] for i in [0,2), j in [2,4)
    #   U index for A[i,j] = i*4 + j
    #   B21 = B[2:4, 0:2], so B[j, k] for j in [2,4), k in [0,2)
    #   V index for B[j,k] = j*4 + k
    #   Z2 = P2 = C[:, 0:2], same as above
    #
    # Product 3: A1 * B12
    #   A1 = A[:, 0:2]
    #   B12 = B[0:2, 2:4], so B[j, k] for j in [0,2), k in [2,4)
    #   Z3 = P3 = C[:, 2:4], so C[i, k] for i in [0,2), k in [2,4)
    #   W index for C[i,k] = k*2 + i
    #
    # Product 4: A2 * B22
    #   A2 = A[:, 2:4]
    #   B22 = B[2:4, 2:4]
    #   Z4 = P4 = C[:, 2:4]
    
    # Define the mapping for each product:
    # For X (which is A block): X[row, col] corresponds to U[i*4 + j]
    #   where i is the row in the 2x2 block, j is the actual column in A
    # For Y (which is B block): Y[row, col] corresponds to V[j*4 + k]
    #   where j is the actual row in B, k is the actual column in B
    # For Z (which is C block): Z[row, col] corresponds to W[k*2 + i]
    #   where i is the row in C, k is the actual column in C
    
    products = [
        {
            'name': 'A1*B11',
            'U_cols': [0, 1],     # j values for A1
            'V_rows': [0, 1],     # j values for B11
            'V_cols': [0, 1],     # k values for B11
            'W_rows': [0, 1],     # i values for C
            'W_cols': [0, 1],     # k values for C
        },
        {
            'name': 'A2*B21',
            'U_cols': [2, 3],     # j values for A2
            'V_rows': [2, 3],     # j values for B21
            'V_cols': [0, 1],     # k values for B21
            'W_rows': [0, 1],     # i values for C
            'W_cols': [0, 1],     # k values for C
        },
        {
            'name': 'A1*B12',
            'U_cols': [0, 1],     # j values for A1
            'V_rows': [0, 1],     # j values for B12
            'V_cols': [2, 3],     # k values for B12
            'W_rows': [0, 1],     # i values for C
            'W_cols': [2, 3],     # k values for C
        },
        {
            'name': 'A2*B22',
            'U_cols': [2, 3],     # j values for A2
            'V_rows': [2, 3],     # j values for B22
            'V_cols': [2, 3],     # k values for B22
            'W_rows': [0, 1],     # i values for C
            'W_cols': [2, 3],     # k values for C
        },
    ]
    
    for prod_idx, prod in enumerate(products):
        col_offset = prod_idx * 7
        U_cols = prod['U_cols']
        V_rows = prod['V_rows']
        V_cols = prod['V_cols']
        W_rows = prod['W_rows']
        W_cols = prod['W_cols']
        
        for t in range(7):
            col = col_offset + t
            
            # U coefficients: X[row, col] -> U[i*4 + j]
            # row 0 -> i=0, row 1 -> i=1
            # col 0 -> U_cols[0], col 1 -> U_cols[1]
            U_coeffs = strassen_U[t]
            U_coeffs = [U_coeffs[0], U_coeffs[1], U_coeffs[2], U_coeffs[3]]  # [X00, X01, X10, X11]
            
            # X00: row=0, col=0 -> i=0, j=U_cols[0] -> U[0*4 + U_cols[0]]
            # X01: row=0, col=1 -> i=0, j=U_cols[1] -> U[0*4 + U_cols[1]]
            # X10: row=1, col=0 -> i=1, j=U_cols[0] -> U[1*4 + U_cols[0]]
            # X11: row=1, col=1 -> i=1, j=U_cols[1] -> U[1*4 + U_cols[1]]
            
            if U_coeffs[0] != 0:
                U[0 * 4 + U_cols[0], col] += U_coeffs[0]
            if U_coeffs[1] != 0:
                U[0 * 4 + U_cols[1], col] += U_coeffs[1]
            if U_coeffs[2] != 0:
                U[1 * 4 + U_cols[0], col] += U_coeffs[2]
            if U_coeffs[3] != 0:
                U[1 * 4 + U_cols[1], col] += U_coeffs[3]
            
            # V coefficients: Y[row, col] -> V[j*4 + k]
            # row 0 -> j=V_rows[0], row 1 -> j=V_rows[1]
            # col 0 -> k=V_cols[0], col 1 -> k=V_cols[1]
            V_coeffs = strassen_V[t]
            V_coeffs = [V_coeffs[0], V_coeffs[1], V_coeffs[2], V_coeffs[3]]  # [Y00, Y01, Y10, Y11]
            
            # Y00: row=0, col=0 -> j=V_rows[0], k=V_cols[0] -> V[V_rows[0]*4 + V_cols[0]]
            # Y01: row=0, col=1 -> j=V_rows[0], k=V_cols[1] -> V[V_rows[0]*4 + V_cols[1]]
            # Y10: row=1, col=0 -> j=V_rows[1], k=V_cols[0] -> V[V_rows[1]*4 + V_cols[0]]
            # Y11: row=1, col=1 -> j=V_rows[1], k=V_cols[1] -> V[V_rows[1]*4 + V_cols[1]]
            
            if V_coeffs[0] != 0:
                V[V_rows[0] * 4 + V_cols[0], col] += V_coeffs[0]
            if V_coeffs[1] != 0:
                V[V_rows[0] * 4 + V_cols[1], col] += V_coeffs[1]
            if V_coeffs[2] != 0:
                V[V_rows[1] * 4 + V_cols[0], col] += V_coeffs[2]
            if V_coeffs[3] != 0:
                V[V_rows[1] * 4 + V_cols[1], col] += V_coeffs[3]
            
            # W coefficients: Z[row, col] -> W[k*2 + i]
            # row 0 -> i=W_rows[0], row 1 -> i=W_rows[1]
            # col 0 -> k=W_cols[0], col 1 -> k=W_cols[1]
            W_coeffs = strassen_W[t]
            W_coeffs = [W_coeffs[0], W_coeffs[1], W_coeffs[2], W_coeffs[3]]  # [Z00, Z01, Z10, Z11]
            
            # Z00: row=0, col=0 -> i=W_rows[0], k=W_cols[0] -> W[W_cols[0]*2 + W_rows[0]]
            # Z01: row=0, col=1 -> i=W_rows[0], k=W_cols[1] -> W[W_cols[1]*2 + W_rows[0]]
            # Z10: row=1, col=0 -> i=W_rows[1], k=W_cols[0] -> W[W_cols[0]*2 + W_rows[1]]
            # Z11: row=1, col=1 -> i=W_rows[1], k=W_cols[1] -> W[W_cols[1]*2 + W_rows[1]]
            
            if W_coeffs[0] != 0:
                W[W_cols[0] * 2 + W_rows[0], col] += W_coeffs[0]
            if W_coeffs[1] != 0:
                W[W_cols[1] * 2 + W_rows[0], col] += W_coeffs[1]
            if W_coeffs[2] != 0:
                W[W_cols[0] * 2 + W_rows[1], col] += W_coeffs[2]
            if W_coeffs[3] != 0:
                W[W_cols[1] * 2 + W_rows[1], col] += W_coeffs[3]
    
    # Convert to list of lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
