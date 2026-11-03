import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition:
    # A is 2x3, B is 3x3
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x3, B2 is 1x3
    # C = A*B = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x3 -> 2x3. This can be decomposed further.
    # Let B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x1.
    # A1*B1 = A1*B1a + A1*B1b
    # A1*B1a: 2x2 * 2x2 -> 2x2. Rank 7 (Strassen).
    # A1*B1b: 2x2 * 2x1 -> 2x1. Rank 2.
    # A2*B2: 2x1 * 1x3 -> 2x3. Rank 2.
    
    # Total rank: 7 + 2 + 2 = 11? Let's verify.
    
    # Indices:
    # A: (i, j) i in {0,1}, j in {0,1,2}
    # B: (j, k) j in {0,1,2}, k in {0,1,2}
    # C: (i, k) i in {0,1}, k in {0,1,2}
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This is standard Strassen on the 2x2 block.
    # Strassen terms for C_11, C_12, C_21, C_22 (using 1-based for clarity)
    # C11 = A11B11 + A21B12
    # C12 = A11B12 + A21B22
    # C21 = A12B11 + A22B12
    # C22 = A12B12 + A22B22
    
    # Wait, my previous indices:
    # A1 = [[a00, a01], [a10, a11]] (rows 0,1; cols 0,1)
    # B1a = [[b00, b01], [b10, b11]] (rows 0,1; cols 0,1)
    # Product P = A1 * B1a
    # P00 = a00*b00 + a01*b10
    # P01 = a00*b01 + a01*b11
    # P10 = a10*b00 + a11*b10
    # P11 = a10*b01 + a11*b11
    
    # Strassen for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b11-b00)
    # M4 = a11(b00-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A1 * B1b where B1b is 2x1: [[b02], [b12]]
    # Q00 = a00*b02 + a01*b12
    # Q10 = a10*b02 + a11*b12
    # This is 2x2 * 2x1. Rank 2.
    # Can we share with Block 1?
    # The terms in Block 1 involve b00, b01, b10, b11.
    # The terms in Block 2 involve b02, b12.
    # No overlap in B entries. So ranks add.
    # Rank 2 construction:
    # T1 = (a00 + a01) * b02 -> contributes to Q00? No.
    # Let's just do schoolbook for this small part? No, rank 2 is minimal.
    # T1 = a00 * b02
    # T2 = a01 * b12
    # Q00 = T1 + T2
    # But Q10 = a10*b02 + a11*b12.
    # We need 2 more terms?
    # T3 = a10 * b02
    # T4 = a11 * b12
    # Total rank 4 for this block?
    # Wait. 2x2 * 2x1.
    # Tensor rank of (2,2,1) is 2.
    # T1 = (a00 + a10) * b02
    # T2 = (a01 + a11) * b12
    # Then Q00 = ?
    # We need to separate a00 from a10.
    # Actually, 2x2 * 2x1 can be done in rank 2?
    # C = A * v.
    # A is 2x2. v is 2x1.
    # C0 = a00*v0 + a01*v1
    # C1 = a10*v0 + a11*v1
    # This is a linear map from 4-dim space (a00,a01,a10,a11) to 2-dim space? No.
    # It's a bilinear map from (2x2) x (2x1) to (2x1).
    # The tensor is in R^{2x2} \otimes R^{2x1} \otimes R^{2x1}.
    # Rank is 2.
    # Construction:
    # T1 = (a00 + a10) * b02
    # T2 = (a01 + a11) * b12
    # Then C0 = a00*b02 + a01*b12.
    # From T1, we get (a00+a10)*b02. We need to subtract a10*b02.
    # This requires another term.
    # It seems rank is higher.
    # Let's check rank of matrix multiplication (2,2,1).
    # It's the same as (2,1,1) + (2,1,1)? No.
    # It's 2 independent products of 2x1 * 1x1?
    # C0 = a00*b02 + a01*b12
    # C1 = a10*b02 + a11*b12
    # These are two separate equations.
    # They share b02 and b12.
    # If we use schoolbook, rank is 4.
    # Can we do better?
    # (a00+b02) ... no.
    # I suspect rank is 4 for this block.
    # Let's assume rank 4 for now.
    
    # Block 3: A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # A2 is 2x1: [a02, a12]
    # B2 is 1x3: [b20, b21, b22]
    # R00 = a02*b20
    # R01 = a02*b21
    # R02 = a02*b22
    # R10 = a12*b20
    # R11 = a12*b21
    # R12 = a12*b22
    # Rank of (2,1,3).
    # This is 2x1 * 1x3.
    # Rank is 2.
    # T1 = a02 * (b20 + b21 + b22)
    # T2 = a12 * (b20 + b21 + b22)
    # Then R00 = T1 - a02*(b21+b22). Need more terms.
    # Actually, rank of (n, 1, p) is n*p? No.
    # (1,1,1) is rank 1.
    # (2,1,1) is rank 2.
    # (2,1,2) is rank ?
    # C0 = a0*b0 + a0*b1 ? No, a0 is scalar.
    # C = [a0, a1] * [b0, b1, b2]^T? No.
    # A is column vector, B is row vector.
    # C = A * B (outer product).
    # Outer product rank is 1.
    # Wait, A is 2x1, B is 1x3.
    # C is 2x3.
    # C[i][k] = A[i][0] * B[0][k].
    # This is exactly an outer product.
    # The tensor is rank 1!
    # T = A[0]*B[0] + A[1]*B[0] ... no.
    # The bilinear form is sum_k C[i][k] * ...
    # The tensor T in R^2 \otimes R^1 \otimes R^3.
    # T_{i, 0, k} = delta_{i,0} delta_{k,0} ... no.
    # T_{i, j, k} = A_{i,j} B_{j,k}.
    # Here j=2 is fixed.
    # T_{i, 2, k} = A_{i,2} B_{2,k}.
    # This is a rank 1 tensor!
    # U = [1, 1]^T (for A02, A12) -> No, U has shape (n*m, rank).
    # We want to represent the term A[0:2, 2] * B[2, 0:3].
    # This term contributes to C[0:2, 0:3].
    # Let's use 1 term.
    # U[2][0] = 1 (A02), U[5][0] = 1 (A12).
    # V[6][0] = 1 (B20), V[7][0] = 1 (B21), V[8][0] = 1 (B22).
    # This product is (A02+A12)(B20+B21+B22).
    # This gives all 6 terms plus cross terms.
    # We need to subtract cross terms.
    # So rank is not 1.
    # Rank of (2,1,3) matrix multiplication.
    # It is known to be 2? Or 3?
    # Let's use schoolbook for safety. Rank 6.
    # Or maybe 4?
    
    # Let's stick to the verified Rank 17 from the previous run, but fix the errors.
    # The previous run had Rank 17.
    # My analysis suggests:
    # Block 1 (2x2 * 2x2): Rank 7.
    # Block 2 (2x2 * 2x1): Rank 4?
    # Block 3 (2x1 * 1x3): Rank 6?
    # Total 17.
    # This matches the previous score.
    
    # Let's implement the Rank 17 construction carefully.
    
    U = [[0]*17 for _ in range(6)]
    V = [[0]*17 for _ in range(9)]
    W = [[0]*17 for _ in range(6)]
    
    col = 0
    
    # --- Block 1: Strassen for A[0:2,0:2] * B[0:2,0:2] ---
    # Indices:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1 # C00, C11
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1 # C10
    col += 1
    
    # M3 = A00(B11-B00)
    U[0][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1 # C01
    col += 1
    
    # M4 = A11(B00-B11)
    U[4][col] = 1
    V[0][col] = 1; V[5][col] = -1
    W[0][col] = 1; W[2][col] = 1 # C00, C10
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1 # C00, C01
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1 # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[5][col] = 1
    W[0][col] = 1 # C00
    col += 1
    
    # --- Block 2: Schoolbook for A[0:2,0:2] * B[0:2,2] ---
    # A00*B02, A01*B12, A10*B02, A11*B12
    # C02, C12
    
    # T8: A00 * B02
    U[0][col] = 1
    V[2][col] = 1
    W[4][col] = 1 # C02 (k=2, i=0 -> index 4)
    col += 1
    
    # T9: A01 * B12
    U[1][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    col += 1
    
    # T10: A10 * B02
    U[3][col] = 1
    V[2][col] = 1
    W[5][col] = 1 # C12 (k=2, i=1 -> index 5)
    col += 1
    
    # T11: A11 * B12
    U[4][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # --- Block 3: Schoolbook for A[0:2,2] * B[2,0:3] ---
    # A02*B20, A02*B21, A02*B22
    # A12*B20, A12*B21, A12*B22
    # C00, C01, C02
    # C10, C11, C12
    
    # T12: A02 * B20
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1 # C00
    col += 1
    
    # T13: A02 * B21
    U[2][col] = 1
    V[7][col] = 1
    W[1][col] = 1 # C01
    col += 1
    
    # T14: A02 * B22
    U[2][col] = 1
    V[8][col] = 1
    W[4][col] = 1 # C02
    col += 1
    
    # T15: A12 * B20
    U[5][col] = 1
    V[6][col] = 1
    W[2][col] = 1 # C10
    col += 1
    
    # T16: A12 * B21
    U[5][col] = 1
    V[7][col] = 1
    W[3][col] = 1 # C11
    col += 1
    
    # T17: A12 * B22
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1 # C12
    col += 1
    
    return {"U": U, "V": V, "W": W}
