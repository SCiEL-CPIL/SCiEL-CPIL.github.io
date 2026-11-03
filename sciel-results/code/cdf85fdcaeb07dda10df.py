"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Approach: View the multiplication as a 2x2 block matrix multiplication over
    the ring of 2x2 matrices (treating the 4 middle indices as 2x2 blocks).
    A (2x4) = [A00 A01; A10 A11] where each Aij is 1x2
    B (4x8) = [B00 B01 B02 B03; B10 B11 B12 B13; B20 B21 B22 B23; B30 B31 B32 B33]
              = [[B00 B01], [B02 B03]; [B10 B11], [B12 B13]; [B20 B21], [B22 B23]; [B30 B31], [B32 B33]]
    
    Actually, let's think of it as:
    A is 2x4, B is 4x8.
    Split A into 2 blocks of 2 columns: A = [A_left | A_right] where A_left, A_right are 2x2
    Split B into 4 blocks of 2 rows: B = [B0; B1; B2; B3] where B0, B1, B2, B3 are 2x8
    
    C = A * B = A_left * [B0; B1] + A_right * [B2; B3]
    
    This doesn't quite work for Strassen. Let me reconsider.
    
    Better approach: 
    A (2x4) can be seen as a 2x2 matrix where each entry is a 1x2 vector.
    A = [[a00, a01], [a10, a11]] where aij is a 1x2 row vector.
    
    B (4x8) can be seen as a 2x2 matrix where each entry is a 2x4 block.
    B = [[B00, B01], [B10, B11]] where Bij is a 2x4 matrix.
    
    Then C = A * B is a 2x2 matrix of 1x4 blocks:
    C = [[C00, C01], [C10, C11]] where Cij is a 1x4 row vector.
    
    C00 = a00*B00 + a01*B10
    C01 = a00*B01 + a01*B11
    C10 = a10*B00 + a11*B10
    C11 = a10*B01 + a11*B11
    
    This is a 2x2 matrix multiplication over the ring R where elements of R are 1x2 * 2x4 = 1x4 products.
    The ring multiplication is: (u, v) * (X, Y) where u,v are 1x2 and X,Y are 2x4 blocks... 
    Actually, the "elements" are the 1x4 vectors, and multiplication is not defined in the usual sense.
    
    Let me think differently. The tensor for 2x4 * 4x8 has rank at most 7 * rank(1x2 * 2x4).
    The rank of 1x2 * 2x4 (i.e., multiplying a 1x2 row vector by a 2x4 matrix to get a 1x4 row vector) is 4.
    So rank <= 7 * 4 = 28.
    
    Construction:
    Use Strassen's 7 multiplications for the 2x2 "outer" structure.
    For each Strassen multiplier, we need to compute products of the form (linear combo of a-entries) * (linear combo of B-blocks).
    
    Let's define:
    a00 = row 0, col 0-1 of A (1x2)
    a01 = row 0, col 2-3 of A (1x2)  
    a10 = row 1, col 0-1 of A (1x2)
    a11 = row 1, col 2-3 of A (1x2)
    
    B00 = rows 0-1, cols 0-3 of B (2x4)
    B01 = rows 0-1, cols 4-7 of B (2x4)
    B10 = rows 2-3, cols 0-3 of B (2x4)
    B11 = rows 2-3, cols 4-7 of B (2x4)
    
    Strassen multipliers:
    M1 = (a00 + a11) * (B00 + B11)
    M2 = (a10 + a11) * B00
    M3 = a00 * (B10 - B11)
    M4 = a11 * (B01 - B00)
    M5 = (a00 + a10) * B11
    M6 = (a10 - a00) * (B00 + B01)
    M7 = (a01 - a11) * (B10 + B11)
    
    Then:
    C00 = M1 + M3 - M4 + M6
    C01 = M1 + M2 - M4 + M7  (wait, need to check)
    
    Let me verify with standard Strassen:
    C00 = M1 - M4 + M6 + M2
    C01 = M4 + M5
    C10 = M1 + M3 - M5 - M7
    C11 = M2 + M4 - M3 + M7
    
    Each Mi is a 1x4 vector (product of 1x2 and 2x4).
    To decompose each Mi into rank-4 terms, we use:
    If u = [u0, u1] (1x2) and X is 2x4, then u*X = u0*X_row0 + u1*X_row1.
    This requires 4 scalar multiplications (one per column of the result).
    
    So each Mi contributes 4 terms to the tensor rank.
    Total rank = 7 * 4 = 28.
    
    Let me build this explicitly.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (32, 28)  
    # W has shape (p*n, rank) = (16, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index conventions:
    # A[i,j] is at U[i*m + j]
    # B[j,k] is at V[j*p + k]
    # C[i,k] is at W[k*n + i] (trace convention: W[k*n + i, r])
    
    # Strassen coefficients for A-side (2x2 blocks):
    # M1: a00 + a11
    # M2: a10 + a11
    # M3: a00
    # M4: a11
    # M5: a00 + a10
    # M6: a10 - a00
    # M7: a01 - a11
    
    # For each multiplier s (0-6), and each column t (0-3) of the 1x4 result:
    # The term is: (sum of A coefficients) * (sum of B coefficients) * output
    
    # A indices: a00 = [0,1], a01 = [2,3], a10 = [4,5], a11 = [6,7] in A (2x4)
    # B blocks:
    # B00 = B[0:2, 0:4] -> V indices [0:8]
    # B01 = B[0:2, 4:8] -> V indices [8:16]
    # B10 = B[2:4, 0:4] -> V indices [16:24]
    # B11 = B[2:4, 4:8] -> V indices [24:32]
    
    # For each Strassen multiplier and each of the 4 columns:
    col = 0
    for s in range(7):
        for t in range(4):  # t = 0,1,2,3 corresponds to the 4 columns of the 1x4 result
            # Determine which A entries contribute and with what coefficient
            # Determine which B entries contribute and with what coefficient
            
            if s == 0:  # M1 = (a00 + a11) * (B00 + B11)
                # A: a00 and a11 both with coeff 1
                # B: B00 and B11 both with coeff 1
                # For column t of the result, we need B00[:, t] and B11[:, t]
                # a00 = A[0,0], A[0,1]; a11 = A[1,2], A[1,3]
                U[0][col] += 1   # a00, col 0
                U[1][col] += 1   # a00, col 1
                U[6][col] += 1   # a11, col 0 (row 1, col 2)
                U[7][col] += 1   # a11, col 1 (row 1, col 3)
                
                # B00[:, t] = B[0,t], B[1,t]
                # B11[:, t] = B[2, t+4], B[3, t+4]
                V[0 * p + t][col] += 1        # B[0, t]
                V[1 * p + t][col] += 1        # B[1, t]
                V[2 * p + (t + 4)][col] += 1  # B[2, t+4]
                V[3 * p + (t + 4)][col] += 1  # B[3, t+4]
                
                # Output: C[0, t] += M1[:, t], C[1, t+4] += M1[:, t]
                # Wait, C is 2x8. C00 is cols 0-3, C01 is cols 4-7, C10 is cols 0-3, C11 is cols 4-7
                # M1 contributes to C00 and C11? Let me recheck Strassen.
                # C00 = M1 - M4 + M6 + M2
                # C11 = M2 + M4 - M3 + M7
                # So M1 contributes to C00 (row 0, cols 0-3)
                W[t * n + 0][col] += 1  # C[0, t]
                
            elif s == 1:  # M2 = (a10 + a11) * B00
                # A: a10 and a11 with coeff 1
                # B: B00 with coeff 1
                U[4][col] += 1   # a10, col 0
                U[5][col] += 1   # a10, col 1
                U[6][col] += 1   # a11, col 0
                U[7][col] += 1   # a11, col 1
                
                V[0 * p + t][col] += 1  # B[0, t]
                V[1 * p + t][col] += 1  # B[1, t]
                
                # M2 contributes to C00 and C11
                W[t * n + 0][col] += 1  # C[0, t]
                W[(t + 4) * n + 1][col] += 1  # C[1, t+4]
                
            elif s == 2:  # M3 = a00 * (B10 - B11)
                # A: a00 with coeff 1
                # B: B10 with coeff 1, B11 with coeff -1
                U[0][col] += 1   # a00, col 0
                U[1][col] += 1   # a00, col 1
                
                V[2 * p + t][col] += 1        # B[2, t]
                V[3 * p + t][col] += 1        # B[3, t]
                V[2 * p + (t + 4)][col] -= 1  # B[2, t+4]
                V[3 * p + (t + 4)][col] -= 1  # B[3, t+4]
                
                # M3 contributes to C10 and C11
                W[t * n + 1][col] += 1  # C[1, t]
                W[(t + 4) * n + 1][col] -= 1  # C[1, t+4]
                
            elif s == 3:  # M4 = a11 * (B01 - B00)
                # A: a11 with coeff 1
                # B: B01 with coeff 1, B00 with coeff -1
                U[6][col] += 1   # a11, col 0
                U[7][col] += 1   # a11, col 1
                
                V[0 * p + (t + 4)][col] += 1  # B[0, t+4]
                V[1 * p + (t + 4)][col] += 1  # B[1, t+4]
                V[0 * p + t][col] -= 1        # B[0, t]
                V[1 * p + t][col] -= 1        # B[1, t]
                
                # M4 contributes to C00, C01, C11
                W[t * n + 0][col] -= 1  # C[0, t]
                W[(t + 4) * n + 0][col] += 1  # C[0, t+4]
                W[(t + 4) * n + 1][col] += 1  # C[1, t+4]
                
            elif s == 4:  # M5 = (a00 + a10) * B11
                # A: a00 and a10 with coeff 1
                # B: B11 with coeff 1
                U[0][col] += 1   # a00, col 0
                U[1][col] += 1   # a00, col 1
                U[4][col] += 1   # a10, col 0
                U[5][col] += 1   # a10, col 1
                
                V[2 * p + (t + 4)][col] += 1  # B[2, t+4]
                V[3 * p + (t + 4)][col] += 1  # B[3, t+4]
                
                # M5 contributes to C01 and C10
                W[(t + 4) * n + 0][col] += 1  # C[0, t+4]
                W[t * n + 1][col] -= 1  # C[1, t]
                
            elif s == 5:  # M6 = (a10 - a00) * (B00 + B01)
                # A: a10 with coeff 1, a00 with coeff -1
                # B: B00 and B01 with coeff 1
                U[4][col] += 1   # a10, col 0
                U[5][col] += 1   # a10, col 1
                U[0][col] -= 1   # a00, col 0
                U[1][col] -= 1   # a00, col 1
                
                V[0 * p + t][col] += 1        # B[0, t]
                V[1 * p + t][col] += 1        # B[1, t]
                V[0 * p + (t + 4)][col] += 1  # B[0, t+4]
                V[1 * p + (t + 4)][col] += 1  # B[1, t+4]
                
                # M6 contributes to C00
                W[t * n + 0][col] += 1  # C[0, t]
                
            elif s == 6:  # M7 = (a01 - a11) * (B10 + B11)
                # A: a01 with coeff 1, a11 with coeff -1
                # B: B10 and B11 with coeff 1
                U[2][col] += 1   # a01, col 0
                U[3][col] += 1   # a01, col 1
                U[6][col] -= 1   # a11, col 0
                U[7][col] -= 1   # a11, col 1
                
                V[2 * p + t][col] += 1        # B[2, t]
                V[3 * p + t][col] += 1        # B[3, t]
                V[2 * p + (t + 4)][col] += 1  # B[2, t+4]
                V[3 * p + (t + 4)][col] += 1  # B[3, t+4]
                
                # M7 contributes to C01 and C11
                W[(t + 4) * n + 0][col] += 1  # C[0, t+4]
                W[(t + 4) * n + 1][col] += 1  # C[1, t+4]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

