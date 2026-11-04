"""
Matrix multiplication 4x4 by 4x8 using Kronecker product of 2x2 Strassen.
Rank = 7 * 7 = 49.
"""
DIMENSIONS = [4, 4, 8]

# 2x2 Strassen coefficients
# U: shape (4, 7) for input matrix A (flattened row-major)
# V: shape (4, 7) for input matrix B (flattened row-major)
# W: shape (4, 7) for output matrix C (flattened row-major)
# Convention: U[i*2+j, k], V[i*2+j, k], W[i*2+j, k]

_STRASSEN_U = [
    [1, 0, 0, 0, 0, 0, 0],  # a
    [0, 1, 0, 0, 0, 0, 0],  # b
    [0, 0, 1, 0, 0, 0, 0],  # c
    [0, 0, 0, 1, 0, 0, 0],  # d
]
# Wait, this is just the identity. Let me use the actual Strassen decomposition.

# Strassen for 2x2:
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B21-B22)
# M4 = A22(B12-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)
# C11 = M1 - M4 + M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M3 + M2 + M6

# Flattened indices for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
# A11=0, A12=1, A21=2, A22=3
# B11=0, B12=1, B21=2, B22=3
# C11=0, C12=1, C21=2, C22=3

_STRASSEN_U = [
    [1, 1, 1, 0, 1, -1, -1],  # a00
    [0, 0, 0, 0, 1, 0, 1],    # a01
    [0, 1, 0, 0, 0, 1, 0],    # a10
    [1, 1, 0, 1, 0, 0, -1],   # a11
]

_STRASSEN_V = [
    [1, 1, 0, -1, 0, 1, 0],   # b00
    [1, 0, 1, -1, 0, 1, 0],   # b01
    [0, 0, 1, 0, 1, 0, 1],    # b10
    [1, 0, -1, 0, 1, 0, 1],   # b11
]

_STRASSEN_W = [
    [1, 0, 0, -1, 1, 0, 1],   # c00
    [0, 0, 1, 0, 1, 0, 0],    # c01
    [0, 1, 0, 1, 0, 0, 0],    # c10
    [1, 0, -1, 0, 0, 1, 0],   # c11
]


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We're computing C = A * B where A is 4x4, B is 4x8, C is 4x8
    # We can think of this as two separate 4x4 multiplications:
    # C_left = A * B_left (B_left is 4x4)
    # C_right = A * B_right (B_right is 4x4)
    
    # Using Kronecker product of 2x2 Strassen:
    # 4x4 multiplication can be decomposed using 7x7 = 49 terms
    # Each 4x4 multiplication uses 49 terms
    # Two 4x4 multiplications use 98 terms total
    # But we can share the A-side factors!
    
    # Actually, let's think about this differently.
    # The tensor for (4x4) * (4x8) -> (4x8) has:
    # U: 4*4 = 16 rows
    # V: 4*8 = 32 rows  
    # W: 8*4 = 32 rows (using trace convention W[k*n+i, r] where k is 0..7, i is 0..3)
    
    # Using Kronecker: treat 4x4 as 2x(2x2) blocks
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13], ...] 
    # where each Bj is 2x4
    
    # Actually, let's just do two independent 4x4 multiplications.
    # For C_left = A * B[:4] and C_right = A * B[4:]
    # Each uses the Kronecker product of 2x2 Strassen.
    
    rank = 49 * 2  # 98 total terms
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Generate 4x4 Kronecker Strassen decomposition
    # For 4x4: indices are (block_row, block_col, local_row, local_col)
    # Flattened: ((br*2+lr)*2+lc) = br*4 + lr*2 + lc
    
    # We need to build U_4x4, V_4x4, W_4x4 of shape (16, 49)
    # using Kronecker product of the 2x2 Strassen matrices
    
    # The Kronecker product structure:
    # For each pair of 2x2 Strassen terms (k1, k2), we get a 4x4 term
    # U_4x4[index, term] = U_2x2[block_idx, k1] * U_2x2[local_idx, k2]
    # where block_idx is the 2x2 block index (0-3) and local_idx is the local 2x2 index
    
    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]
    
    for k1 in range(7):
        for k2 in range(7):
            term = k1 * 7 + k2
            for br in range(2):  # block row
                for bc in range(2):  # block col
                    for lr in range(2):  # local row
                        for lc in range(2):  # local col
                            # Index in 4x4 matrix
                            idx = (br * 2 + lr) * 2 + lc
                            # Block index in 2x2: br*2+bc
                            block_idx = br * 2 + bc
                            # Local index in 2x2: lr*2+lc
                            local_idx = lr * 2 + lc
                            
                            # U factor
                            U4[idx][term] = _STRASSEN_U[block_idx][k1] * _STRASSEN_U[local_idx][k2]
                            V4[idx][term] = _STRASSEN_V[block_idx][k1] * _STRASSEN_V[local_idx][k2]
                            W4[idx][term] = _STRASSEN_W[block_idx][k1] * _STRASSEN_W[local_idx][k2]
    
    # Now fill in U, V, W for both halves
    # For the left half (B columns 0-3):
    for i in range(16):
        for t in range(49):
            U[i][t] = U4[i][t]
            V[i][t] = V4[i][t]  # V index for B_left: i corresponds to 4x4 index
            W[i][t] = W4[i][t]  # W index for C_left: i corresponds to 4x8 index (first 4 cols)
    
    # For the right half (B columns 4-7):
    # V indices shift by 4 (since B has 8 columns, each 2x2 block in B uses 4 cols)
    # Actually, B is 4x8. In the Kronecker decomposition, B is treated as 2x(2x4)
    # But for the second 4x4 multiplication, we use B[4:8]
    # V index for B_right: the 4x4 index i maps to V row i (for B_left) or i+4? 
    # No, V has 32 rows for 4x8. For the left half, V rows 0-15. For right half, V rows 16-31?
    # Wait, V is (m*p, rank) = (4*8, rank) = (32, rank)
    # For B_left (4x4), the V indices are 0..15
    # For B_right (4x4), the V indices are 16..31
    # Similarly, W for C (4x8) has 32 rows. C_left uses W rows 0..15, C_right uses W rows 16..31
    
    for i in range(16):
        for t in range(49):
            U[i][t + 49] = U4[i][t]
            V[i + 16][t + 49] = V4[i][t]
            W[i + 16][t + 49] = W4[i][t]
    
    return {"U": U, "V": V, "W": W}
