"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    
    # Block 1: B cols 0,1 -> C rows 0,1
    # Strassen part (7 mults) + Outer product part (4 mults) = 11
    # A rows 0,1. B cols 0,1 (global). C rows 0,1 (global).
    
    # Strassen on A[0:2, 0:2] and B[0:2, 0:2]
    # 1. (A00+A11)(B00+B11)
    U[0, col_offset] += 1; U[3, col_offset] += 1
    V[0, col_offset] += 1; V[4, col_offset] += 1
    W[0, col_offset] += 1; W[3, col_offset] += 1
    col_offset += 1
    
    # 2. (A10+A11)B00
    U[3, col_offset] += 1; U[4, col_offset] += 1
    V[0, col_offset] += 1
    W[1, col_offset] += 1
    col_offset += 1
    
    # 3. A00(B10-B00)
    U[0, col_offset] += 1
    V[3, col_offset] += 1; V[0, col_offset] -= 1
    W[2, col_offset] += 1
    col_offset += 1
    
    # 4. A11(B01-B11)
    U[4, col_offset] += 1
    V[1, col_offset] += 1; V[4, col_offset] -= 1
    W[3, col_offset] += 1
    col_offset += 1
    
    # 5. (A00+A01)B11
    U[0, col_offset] += 1; U[1, col_offset] += 1
    V[4, col_offset] += 1
    W[2, col_offset] += 1
    col_offset += 1
    
    # 6. (A10-A00)(B00+B01)
    U[3, col_offset] += 1; U[0, col_offset] -= 1
    V[0, col_offset] += 1; V[1, col_offset] += 1
    W[1, col_offset] += 1; W[3, col_offset] += 1
    col_offset += 1
    
    # 7. (A01-A11)(B10+B11)
    U[1, col_offset] += 1; U[4, col_offset] -= 1
    V[3, col_offset] += 1; V[4, col_offset] += 1
    W[0, col_offset] += 1; W[2, col_offset] += 1
    col_offset += 1
    
    # Outer product: A[:,2] * B[2,:]
    # C00 += A02*B20
    U[2, col_offset] += 1; V[6, col_offset] += 1; W[0, col_offset] += 1
    col_offset += 1
    # C01 += A02*B21
    U[2, col_offset] += 1; V[7, col_offset] += 1; W[1, col_offset] += 1
    col_offset += 1
    # C10 += A12*B20
    U[5, col_offset] += 1; V[6, col_offset] += 1; W[2, col_offset] += 1
    col_offset += 1
    # C11 += A12*B21
    U[5, col_offset] += 1; V[7, col_offset] += 1; W[3, col_offset] += 1
    col_offset += 1
    
    # Block 2: B cols 2,3 -> C rows 2,3
    # A rows 0,1. B cols 2,3 (global). C rows 2,3 (global).
    # Strassen on A[0:2, 0:2] and B[0:2, 2:4] (local cols 0,1)
    
    # 1. (A00+A11)(B20+B31) -> B global (2,0) and (3,1)? No.
    # B is 3x7.
    # Block 2 uses B cols 2 and 3.
    # B[0,2], B[0,3], B[1,2], B[1,3], B[2,2], B[2,3].
    # Local B indices:
    # B_loc[0,0] = B_global[0,2] = V[0*3+2] = V[2]
    # B_loc[0,1] = B_global[0,3] = V[0*3+3] = V[3]
    # B_loc[1,0] = B_global[1,2] = V[1*3+2] = V[5]
    # B_loc[1,1] = B_global[1,3] = V[1*3+3] = V[6]
    # B_loc[2,0] = B_global[2,2] = V[2*3+2] = V[8]
    # B_loc[2,1] = B_global[2,3] = V[2*3+3] = V[9]
    
    # C rows 2,3.
    # C_loc[0,0] = C_global[0,2] = W[2*2+0] = W[4]
    # C_loc[0,1] = C_global[0,3] = W[2*2+1] = W[5]
    # C_loc[1,0] = C_global[1,2] = W[2*2+2] = W[6]
    # C_loc[1,1] = C_global[1,3] = W[2*2+3] = W[7]
    
    # 1. (A00+A11)(B20+B31)
    U[0, col_offset] += 1; U[3, col_offset] += 1
    V[2, col_offset] += 1; V[6, col_offset] += 1
    W[4, col_offset] += 1; W[7, col_offset] += 1
    col_offset += 1
    
    # 2. (A10+A11)B20
    U[3, col_offset] += 1; U[4, col_offset] += 1
    V[2, col_offset] += 1
    W[5, col_offset] += 1
    col_offset += 1
    
    # 3. A00(B30-B20)
    U[0, col_offset] += 1
    V[5, col_offset] += 1; V[2, col_offset] -= 1
    W[6, col_offset] += 1
    col_offset += 1
    
    # 4. A11(B21-B31)
    U[4, col_offset] += 1
    V[3, col_offset] += 1; V[6, col_offset] -= 1
    W[7, col_offset] += 1
    col_offset += 1
    
    # 5. (A00+A01)B31
    U[0, col_offset] += 1; U[1, col_offset] += 1
    V[6, col_offset] += 1
    W[6, col_offset] += 1
    col_offset += 1
    
    # 6. (A10-A00)(B20+B21)
    U[3, col_offset] += 1; U[0, col_offset] -= 1
    V[2, col_offset] += 1; V[3, col_offset] += 1
    W[5, col_offset] += 1; W[7, col_offset] += 1
    col_offset += 1
    
    # 7. (A01-A11)(B30+B31)
    U[1, col_offset] += 1; U[4, col_offset] -= 1
    V[5, col_offset] += 1; V[6, col_offset] += 1
    W[4, col_offset] += 1; W[6, col_offset] += 1
    col_offset += 1
    
    # Outer product: A[:,2] * B[2,2:4]
    # C00 += A02*B22
    U[2, col_offset] += 1; V[8, col_offset] += 1; W[4, col_offset] += 1
    col_offset += 1
    # C01 += A02*B23
    U[2, col_offset] += 1; V[9, col_offset] += 1; W[5, col_offset] += 1
    col_offset += 1
    # C10 += A12*B22
    U[5, col_offset] += 1; V[8, col_offset] += 1; W[6, col_offset] += 1
    col_offset += 1
    # C11 += A12*B23
    U[5, col_offset] += 1; V[9, col_offset] += 1; W[7, col_offset] += 1
    col_offset += 1
    
    # Block 3: B cols 4,5 -> C rows 4,5
    # B global cols 4,5.
    # B_loc[0,0] = V[4], B_loc[0,1] = V[5]
    # B_loc[1,0] = V[7], B_loc[1,1] = V[8]
    # B_loc[2,0] = V[10], B_loc[2,1] = V[11]
    # C rows 4,5.
    # C_loc[0,0] = W[8], C_loc[0,1] = W[9]
    # C_loc[1,0] = W[10], C_loc[1,1] = W[11]
    
    # 1.
    U[0, col_offset] += 1; U[3, col_offset] += 1
    V[4, col_offset] += 1; V[8, col_offset] += 1
    W[8, col_offset] += 1; W[11, col_offset] += 1
    col_offset += 1
    
    # 2.
    U[3, col_offset] += 1; U[4, col_offset] += 1
    V[4, col_offset] += 1
    W[9, col_offset] += 1
    col_offset += 1
    
    # 3.
    U[0, col_offset] += 1
    V[7, col_offset] += 1; V[4, col_offset] -= 1
    W[10, col_offset] += 1
    col_offset += 1
    
    # 4.
    U[4, col_offset] += 1
    V[5, col_offset] += 1; V[8, col_offset] -= 1
    W[11, col_offset] += 1
    col_offset += 1
    
    # 5.
    U[0, col_offset] += 1; U[1, col_offset] += 1
    V[8, col_offset] += 1
    W[10, col_offset] += 1
    col_offset += 1
    
    # 6.
    U[3, col_offset] += 1; U[0, col_offset] -= 1
    V[4, col_offset] += 1; V[5, col_offset] += 1
    W[9, col_offset] += 1; W[11, col_offset] += 1
    col_offset += 1
    
    # 7.
    U[1, col_offset] += 1; U[4, col_offset] -= 1
    V[7, col_offset] += 1; V[8, col_offset] += 1
    W[8, col_offset] += 1; W[10, col_offset] += 1
    col_offset += 1
    
    # Outer product
    U[2, col_offset] += 1; V[10, col_offset] += 1; W[8, col_offset] += 1
    col_offset += 1
    U[2, col_offset] += 1; V[11, col_offset] += 1; W[9, col_offset] += 1
    col_offset += 1
    U[5, col_offset] += 1; V[10, col_offset] += 1; W[10, col_offset] += 1
    col_offset += 1
    U[5, col_offset] += 1; V[11, col_offset] += 1; W[11, col_offset] += 1
    col_offset += 1
    
    # Block 4: B col 6 -> C row 6
    # Schoolbook 2x3x1. Rank 6.
    # C0 += A00*B06 + A01*B16 + A02*B26
    # C1 += A10*B06 + A11*B16 + A12*B26
    # B global col 6.
    # B_loc[0,0] = V[6], B_loc[1,0] = V[9], B_loc[2,0] = V[12]
    # C_loc[0,0] = W[12], C_loc[1,0] = W[13]
    
    U[0, col_offset] += 1; V[6, col_offset] += 1; W[12, col_offset] += 1
    col_offset += 1
    U[1, col_offset] += 1; V[9, col_offset] += 1; W[12, col_offset] += 1
    col_offset += 1
    U[2, col_offset] += 1; V[12, col_offset] += 1; W[12, col_offset] += 1
    col_offset += 1
    
    U[3, col_offset] += 1; V[6, col_offset] += 1; W[13, col_offset] += 1
    col_offset += 1
    U[4, col_offset] += 1; V[9, col_offset] += 1; W[13, col_offset] += 1
    col_offset += 1
    U[5, col_offset] += 1; V[12, col_offset] += 1; W[13, col_offset] += 1
    col_offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
