"""Rewritten program: block-Strassen decomposition for 2x2 * 2x13 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 2, 13]

# Strassen coefficients for 2x2 multiplication
# M1 = A00 * B00
# M2 = (A00 + A10) * B01
# M3 = (A01 + A11) * B10
# M4 = A11 * B11
# M5 = (A00 + A11) * (B00 + B11)
# M6 = (A10 - A00) * (B01 + B11)
# M7 = (A01 - A11) * (B00 + B10)
# C00 = M5 + M4 - M2 + M6
# C01 = M1 + M2
# C10 = M3 + M4
# C11 = M5 + M1 - M3 + M7

# A coefficients (U): row index = i*2 + j (i=0..1, j=0..1)
# B coefficients (V): row index = j*2 + k (j=0..1, k=0..1)
# C coefficients (W): row index = k*2 + i (k=0..1, k is col, i is row)

U_2x2 = np.array([
    [1,  1,  0,  0,  1,  0,  0],   # A00
    [0,  0,  1,  0,  1,  0, -1],   # A01
    [0,  1,  0,  0,  0,  1,  0],   # A10
    [0,  0,  0,  1,  1,  0, -1],   # A11
])

V_2x2 = np.array([
    [1,  0,  0,  0,  1,  0,  0],   # B00
    [0,  1,  0,  0,  0,  1,  0],   # B01
    [0,  0, 1,  0,  0,  0,  1],   # B10
    [0,  0,  0,  1,  1,  1,  0],   # B11
])

W_2x2 = np.array([
    [0,  0,  0,  0,  1,  1,  0],   # C00
    [1,  1,  0,  0,  0,  0,  0],   # C01
    [0,  0,  1,  1,  0,  0,  0],   # C10
    [0,  0,  0,  0,  1,  0,  1],   # C11
])

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # We'll split columns of B into blocks of 2, plus one remainder column
    # 13 = 2*6 + 1
    num_blocks_2x2 = p // 2  # 6
    remainder = p % 2  # 1
    
    # Each 2x2 block contributes 7 rank terms
    # Each 2x1 remainder contributes 4 rank terms (schoolbook)
    total_rank = num_blocks_2x2 * 7 + remainder * 4  # 6*7 + 1*4 = 46
    
    U = np.zeros((n * m, total_rank))  # (4, 46)
    V = np.zeros((m * p, total_rank))  # (26, 46)
    W = np.zeros((p * n, total_rank))  # (26, 46)
    
    col = 0
    for b in range(num_blocks_2x2):
        # This block handles B columns 2b and 2b+1
        block_start = 2 * b
        
        # U: A coefficients are the same for each block
        U[:, col:col+7] = U_2x2
        
        # V: B coefficients, shifted for this block
        # V row index = j * p + k, where j is B row (0..1), k is B col (0..12)
        # For this block, k = block_start + local_k (local_k = 0, 1)
        for local_k in range(2):
            for local_j in range(2):
                row = local_j * p + (block_start + local_k)
                V[row, col:col+7] = V_2x2[local_j * 2 + local_k]
        
        # W: C coefficients
        # W row index = k * n + i, where k is C col (0..12), i is C row (0..1)
        # For this block, k = block_start + local_k
        for local_k in range(2):
            for local_i in range(2):
                row = (block_start + local_k) * n + local_i
                W[row, col:col+7] = W_2x2[local_k * 2 + local_i]
        
        col += 7
    
    if remainder == 1:
        # Last column is column 12
        last_col = num_blocks_2x2 * 2  # 12
        
        # Schoolbook for 2x1:
        # T1 = A00 * B0, T2 = A01 * B1, T3 = A10 * B0, T4 = A11 * B1
        # C0 = T1 + T2, C1 = T3 + T4
        # U: A coefficients
        U[0, col] = 1   # A00
        U[1, col+1] = 1 # A01
        U[2, col+2] = 1 # A10
        U[3, col+3] = 1 # A11
        
        # V: B coefficients (only one column)
        V[0 * p + last_col, col] = 1   # B0, col 12
        V[1 * p + last_col, col+1] = 1 # B1, col 12
        V[0 * p + last_col, col+2] = 1 # B0, col 12
        V[1 * p + last_col, col+3] = 1 # B1, col 12
        
        # W: C coefficients
        # C row 0, col 12: W index = 12*2 + 0 = 24
        # C row 1, col 12: W index = 12*2 + 1 = 25
        W[last_col * n + 0, col] = 1   # C0
        W[last_col * n + 0, col+1] = 1 # C0
        W[last_col * n + 1, col+2] = 1 # C1
        W[last_col * n + 1, col+3] = 1 # C1
    
    # Convert to lists for the expected return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
