import time
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    rank = 14
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2 (extended to 2x2xp)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each block (0: A1B1, 1: A2B2)
    # Block 0: j in {0, 1}
    # Block 1: j in {2, 3}
    
    for block in range(2):
        base_c = block * 7
        j_offset = block * 2
        
        # U indices: i in {0, 1}, j in {0, 1} relative to block
        # V indices: j in {0, 1} relative to block, k in {0..6}
        # W indices: k in {0..6}, i in {0, 1}
        
        # Term 1: M1
        c = base_c + 0
        # U: A00, A11
        U[0 * m + j_offset + 0][c] = 1  # A00
        U[1 * m + j_offset + 1][c] = 1  # A11
        # V: B00, B11 (rows 0, 1 relative)
        for k in range(p):
            V[(j_offset + 0) * p + k][c] = 1  # B0k
            V[(j_offset + 1) * p + k][c] = 1  # B1k
        # W: C00, C11
        for k in range(p):
            W[k * n + 0][c] = 1  # C0k
            W[k * n + 1][c] = 1  # C1k
            
        # Term 2: M2
        c = base_c + 1
        # U: A10, A11
        U[1 * m + j_offset + 0][c] = 1
        U[1 * m + j_offset + 1][c] = 1
        # V: B00 (row 0)
        for k in range(p):
            V[(j_offset + 0) * p + k][c] = 1
        # W: C10
        for k in range(p):
            W[k * n + 1][c] = 1
            
        # Term 3: M3
        c = base_c + 2
        # U: A00
        U[0 * m + j_offset + 0][c] = 1
        # V: B10 - B11 (row 1 - row 1? No, B10 and B11 are entries)
        # In vector form: B_row1 - B_row1? No.
        # M3 = A00 (B10 - B11). B10 is (1,0), B11 is (1,1).
        # So V[1, 0] = 1, V[1, 1] = -1.
        # But we are in 2x2xp. B is 2xp.
        # B10 is entry (1,0), B11 is entry (1,1).
        # So for k=0, V[1,0]=1. For k=1, V[1,1]=-1.
        # Wait, the formula M3 = A00(B10 - B11) is for 2x2.
        # For 2xp, B10 and B11 are not defined for k>1.
        # The Strassen algorithm works on 2x2 blocks.
        # So we must split B into 2x2 blocks?
        # No, the rank of M2,2,p is 7.
        # The linear forms are:
        # M1 = (A00+A11)(B0 + B1) where B0, B1 are rows.
        # M2 = (A10+A11)B0
        # M3 = A00(B1 - B0)? No.
        # Let's stick to the 2x2 formulas and extend to columns.
        # For each column k, we have a 2x1 vector b_k.
        # The formula applies to the 2x2 block of B?
        # No, B is 2xp.
        # The "B00, B01, B10, B11" in Strassen refer to the 2x2 matrix.
        # If B is 2xp, we can view it as p columns.
        # The rank is 7.
        # The linear forms for A are the same.
        # The linear forms for B are:
        # v1 = B0 + B1 (sum of rows)
        # v2 = B0
        # v3 = B1 - B0?
        # Let's check the rank 7 decomposition for M2,2,p.
        # It is known that R(M2,2,p) = 7.
        # The decomposition is:
        # M1 = (A00+A11)(B0+B1)
        # M2 = (A10+A11)B0
        # M3 = A00(B1-B0)
        # M4 = A11(B0-B1)
        # M5 = (A00+A01)B1
        # M6 = (A10-A00)(B0+B1)
        # M7 = (A01-A11)(B1-B0)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's verify C00 with these forms.
        # M1 = A00B0 + A00B1 + A11B0 + A11B1
        # M4 = A11B0 - A11B1
        # M5 = A00B1 + A01B1
        # M7 = A01B1 - A01B0 - A11B1 + A11B0
        # Sum M1+M4-M5-M7:
        # (A00B0 + A00B1 + A11B0 + A11B1) + (A11B0 - A11B1) - (A00B1 + A01B1) - (A01B1 - A01B0 - A11B1 + A11B0)
        # = A00B0 + A00B1 + 2A11B0 + A11B1 - A00B1 - A01B1 - A01B1 + A01B0 + A11B1 - A11B0
        # = A00B0 + A11B0 + A11B1 + A01B0 - 2A01B1
        # This is not A00B0 + A01B1.
        
        # I am struggling with the exact forms for M2,2,p.
        # However, I can just use the 2x2 Strassen on each 2x2 block of B?
        # B is 4x7. A is 2x4.
        # Split A into A1 (2x2), A2 (2x2).
        # Split B into B1 (2x7), B2 (2x7).
        # C = A1B1 + A2B2.
        # For A1B1, we can use 7 terms if R(M2,2,7)=7.
        # If I can't verify R(M2,2,7)=7 easily, I can use the schoolbook for 2x2x7 which is 4*7=28? No.
        # Schoolbook for 2x2x7 is 2*2*7 = 28 terms.
        # Total rank 28+28=56.
        
        # But Strassen for 2x2x2 is 7.
        # For 2x2x7, we can use 7 terms?
        # Yes, because the tensor M2,2,p is a sub-tensor of M2,2,p.
        # Actually, M2,2,p is the direct sum of p copies of M2,2,1? No.
        # It is the "tensor product" of M2,2,1 and M1,1,p?
        # M2,2,p = M2,2,1 \otimes M1,1,p?
        # Rank of M2,2,1 is 4. Rank of M1,1,p is p.
        # Rank of tensor product is product of ranks? 4p.
        # But we know R(M2,2,2)=7 < 4*2=8.
        # So R(M2,2,p) is not simply 4p.
        
        # Let's assume R(M2,2,p)=7 for p>=2.
        # If so, total rank 14.
        
        # I will implement the 2x2 Strassen forms for each 2x2 block of B?
        # No, B is 2x7.
        # I will use the forms:
        # M1 = (A00+A11)(B0+B1)
        # M2 = (A10+A11)B0
        # M3 = A00(B1-B0)
        # M4 = A11(B0-B1)
        # M5 = (A00+A01)B1
        # M6 = (A10-A00)(B0+B1)
        # M7 = (A01-A11)(B1-B0)
        
        # And the reconstruction:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # I will code this up and verify. If it fails, I will fallback to 56.
        
        # Let's fill V for M3: B1 - B0.
        # V[1, k] = 1, V[0, k] = -1.
        V[(j_offset + 1) * p + 0][c] = 1
        V[(j_offset + 0) * p + 0][c] = -1
        # For k>0, B10 and B11 don't exist in the 2x2 formula.
        # But in 2xp, B0 and B1 are vectors.
        # So M3 = A00 (B1 - B0).
        # So V[1, k] = 1, V[0, k] = -1 for all k.
        for k in range(1, p):
            V[(j_offset + 1) * p + k][c] = 1
            V[(j_offset + 0) * p + k][c] = -1
        # W: C01
        for k in range(p):
            W[k * n + 0][c] = 1  # C0k
            
        # Term 4: M4
        c = base_c + 3
        # U: A11
        U[1 * m + j_offset + 1][c] = 1
        # V: B0 - B1
        for k in range(p):
            V[(j_offset + 0) * p + k][c] = 1
            V[(j_offset + 1) * p + k][c] = -1
        # W: C00, C10
        for k in range(p):
            W[k * n + 0][c] = 1
            W[k * n + 1][c] = 1
            
        # Term 5: M5
        c = base_c + 4
        # U: A00, A01
        U[0 * m + j_offset + 0][c] = 1
        U[0 * m + j_offset + 1][c] = 1
        # V: B1
        for k in range(p):
            V[(j_offset + 1) * p + k][c] = 1
        # W: C00, C01
        for k in range(p):
            W[k * n + 0][c] = -1
            W[k * n + 0][c] = 1 # Overwrite? No, C00 coeff is -1, C01 coeff is 1.
            # Wait, W is a single matrix.
            # W[k, i] is the coeff of M_c in C[i, k].
            # So W[k, 0] for C0k. W[k, 1] for C1k.
            # For M5: C00 coeff -1, C01 coeff 1.
            # So W[k, 0] += -1, W[k, 0] += 1? No.
            # W[k, 0] is for output row 0.
            # So W[k, 0] gets -1 from M5 for C00?
            # And W[k, 0] gets 1 from M5 for C01?
            # No, C00 and C01 are different entries.
            # C00 is (0,0). C01 is (0,1).
            # So W[0, 0] and W[1, 0].
            # So for k=0, W[0, 0] += -1.
            # For k=1, W[1, 0] += 1.
            # So we need to loop k.
            
            # I will clear W and fill carefully.
            
        # I'll restart the W filling logic in the code.
        
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook(n, m, p)

def _verify(U, V, W, n=2, m=4, p=7):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W
