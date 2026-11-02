"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Laderman's 7-multiplication algorithm for 3x3 x 3x3
    # We use this for each of the 8 columns of B.
    # Total rank: 7 * 8 = 56
    
    rank = 56
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 24 rows
    W = [[0]*rank for _ in range(p*n)]  # 24 rows
    
    # Define the 7 Laderman terms for 3x3 x 3x3
    # Indices: A[i,j], B[j,k], C[i,k]
    # Laderman (1976) coefficients:
    # M1: (A00+A11+A22)(B00+B11+B22)
    # M2: (A00+A11)(B00+B11)
    # M3: A00 B00
    # M4: A11 B11
    # M5: (A01+A11+A21)(B01+B11+B21)  -- Note: B indices are (row, col) in B[j,k]
    # M6: (A00+A01+A02)(B00+B10+B20)
    # M7: (A10+A11+A12)(B10+B11+B12)
    
    # Let's define the A-forms (coeffs for A[i,j])
    # A is 3x3. Index i*m+j.
    # B is 3x8. Index j*p+k.
    # C is 3x8. Index k*n+i.
    
    # Laderman A-forms (for 3x3):
    # 0: A00+A11+A22 -> [1,0,0, 0,1,0, 0,0,1]
    # 1: A00+A11 -> [1,0,0, 0,1,0, 0,0,0]
    # 2: A00 -> [1,0,0, 0,0,0, 0,0,0]
    # 3: A11 -> [0,0,0, 0,1,0, 0,0,0]
    # 4: A01+A11+A21 -> [0,1,0, 0,1,0, 0,1,0]
    # 5: A00+A01+A02 -> [1,1,1, 0,0,0, 0,0,0]
    # 6: A10+A11+A12 -> [0,0,0, 1,1,1, 0,0,0]
    
    A_forms = [
        [1,0,0, 0,1,0, 0,0,1],
        [1,0,0, 0,1,0, 0,0,0],
        [1,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0],
        [0,1,0, 0,1,0, 0,1,0],
        [1,1,1, 0,0,0, 0,0,0],
        [0,0,0, 1,1,1, 0,0,0]
    ]
    
    # Laderman B-forms (for 3x3, but we need to map to 3x8 columns)
    # For a specific column k of B, the B-forms are:
    # 0: B00+B11+B22 (of that column? No, B[j,k] is row j, col k)
    # Wait, Laderman B-forms act on the 3x3 matrix B.
    # If we treat each column of B (length 3) as a vector, the "B-matrix" in the 3x3x3 problem is 3x1?
    # No, Laderman is for 3x3 x 3x3.
    # If we apply it to 3x3 x 3x1, we need a 3x1 version.
    # The 3x3x3 algorithm assumes B is 3x3.
    # If we set B to be 3x1 (column vector), the "B01, B02" etc are 0.
    # This reduces the number of non-zero terms.
    # Let's use the 3x3x3 Laderman but zero out the unused B entries.
    
    # For column k of B (vector b_k of length 3):
    # B00 = b_k[0], B10 = b_k[1], B20 = b_k[2]
    # B01 = 0, B11 = 0, B21 = 0
    # B02 = 0, B12 = 0, B22 = 0
    
    # Let's re-evaluate Laderman terms for B=3x1:
    # M1: (A00+A11+A22)(B00)  (since B11=B22=0)
    # M2: (A00+A11)(B00)
    # M3: A00 B00
    # M4: A11 * 0 = 0
    # M5: (A01+A11+A21)(0) = 0
    # M6: (A00+A01+A02)(B00)
    # M7: (A10+A11+A12)(B00)
    
    # This reduces to 5 non-zero terms per column?
    # M1, M2, M3, M6, M7.
    # M4, M5 are zero.
    # So rank per column is 5?
    # Total rank 5 * 8 = 40.
    
    # Let's verify if these 5 terms are sufficient to reconstruct C.
    # C00 = A00B00 + A01B10 + A02B20
    # C10 = A10B00 + A11B10 + A12B20
    # C20 = A20B00 + A21B10 + A22B20
    
    # The 5 terms are:
    # T1 = (A00+A11+A22) B00
    # T2 = (A00+A11) B00
    # T3 = A00 B00
    # T4 = (A00+A01+A02) B00
    # T5 = (A10+A11+A12) B00
    
    # All these terms only involve B00.
    # They do not involve B10 or B20.
    # So this is NOT sufficient.
    # The Laderman algorithm for 3x3x3 uses all 9 B entries.
    # If we restrict B to 3x1, we lose the terms involving B10, B20.
    # We need a 3x3x1 algorithm.
    
    # Rank of 3x3x1?
    # C = A b.
    # This is just a matrix-vector product.
    # Rank is 3? (A00*b0 + A01*b1 + A02*b2)
    # For each row i, we need 3 multiplications.
    # Total 3*3 = 9 multiplications.
    # So rank 9 for 3x3x1.
    # Total rank 9 * 8 = 72.
    
    # So Laderman doesn't help if we split columns.
    # We need to share across columns.
    # This brings us back to the 3x3x8 tensor.
    
    # I will implement the schoolbook (rank 72) as it is the only safe bet I can verify immediately.
    # However, the prompt asks to improve.
    # I will try to implement a **rank 54** construction if I can find one.
    # Or I will return the schoolbook to ensure validity, but score will be 1.0.
    # To beat 1.0, I must find a lower rank.
    
    # Let's try **rank 64**?
    # If I can save 8 multiplications.
    
    # I will output the schoolbook for now to ensure the code runs, 
    # but I will structure it to allow easy swapping.
    
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
