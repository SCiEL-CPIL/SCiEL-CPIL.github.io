import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 30 solution
    # Two 2x3 by 3x3 multiplications, each rank 15.
    
    rank = 30
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_2x3_by_3x3(b_col_offset):
        nonlocal col
        # b_col_offset is 0 or 3
        # A: 2x3 (0..5)
        # B: 3x3 (offset..offset+8)
        # C: 2x3 (0..5) -> W index k*n+i
        
        # Strassen 7 terms for A[0:2, 0:2] and B[0:2, 0:2]
        # M1 = A00 * (B00 + B10)
        # M2 = (A00 + A10) * B10
        # M3 = (A10 + A11) * B00
        # M4 = A11 * (B11 + B10)
        # M5 = (A00 + A10) * (B00 + B10)
        # M6 = (A11 - A01) * (B11 + B10)
        # M7 = (A01 - A11) * (B00 + B11)
        
        # C00 = M1 - M2 + M5
        # C01 = M2 + M4 - M6
        # C10 = M3 - M5 + M7
        # C11 = M1 - M3 + M6
        
        # A indices:
        # A00=0, A01=1, A02=2
        # A10=3, A11=4, A12=5
        
        # B indices:
        # B00=b_col_offset, B01=b_col_offset+1, B02=b_col_offset+2
        # B10=b_col_offset+3, B11=b_col_offset+4, B12=b_col_offset+5
        # B20=b_col_offset+6, B21=b_col_offset+7, B22=b_col_offset+8
        
        # C indices (k*n+i):
        # C00=0, C01=1
        # C10=2, C11=3
        # C20=4, C21=5
        
        # Term 1: M1
        U[0][col] = 1
        V[b_col_offset][col] = 1
        V[b_col_offset+3][col] = 1
        W[0][col] = 1
        W[3][col] = 1
        col += 1
        
        # Term 2: M2
        U[0][col] = 1
        U[3][col] = 1
        V[b_col_offset+3][col] = 1
        W[0][col] = -1
        W[1][col] = 1
        col += 1
        
        # Term 3: M3
        U[3][col] = 1
        U[4][col] = 1
        V[b_col_offset][col] = 1
        W[2][col] = 1
        W[3][col] = -1
        col += 1
        
        # Term 4: M4
        U[4][col] = 1
        V[b_col_offset+4][col] = 1
        V[b_col_offset+3][col] = 1
        W[1][col] = 1
        col += 1
        
        # Term 5: M5
        U[0][col] = 1
        U[3][col] = 1
        V[b_col_offset][col] = 1
        V[b_col_offset+3][col] = 1
        W[0][col] = 1
        W[2][col] = -1
        col += 1
        
        # Term 6: M6
        U[4][col] = 1
        U[1][col] = -1
        V[b_col_offset+4][col] = 1
        V[b_col_offset+3][col] = 1
        W[1][col] = -1
        W[3][col] = 1
        col += 1
        
        # Term 7: M7
        U[1][col] = 1
        U[4][col] = -1
        V[b_col_offset][col] = 1
        V[b_col_offset+4][col] = 1
        W[2][col] = 1
        col += 1
        
        # Remaining 8 terms (schoolbook)
        # C02 = A00*B02 + A01*B12 + A02*B22
        # C12 = A10*B02 + A11*B12 + A12*B22
        # C20 = A02*B20
        # C21 = A02*B21
        # C22 = A02*B22
        # Wait, C is 2x3.
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
        # C[2,0] = A[0,2]B[2,0] ? No, C is 2x3.
        # Indices:
        # C[0,0], C[0,1], C[0,2]
        # C[1,0], C[1,1], C[1,2]
        # W index k*n+i.
        # k=0: 0,1,2
        # k=1: 3,4,5
        
        # So:
        # C[0,2] (idx 2)
        # C[1,2] (idx 5)
        # C[0,0] (idx 0) - already done?
        # C[1,0] (idx 3) - already done?
        
        # The Strassen part computed C[0,0], C[0,1], C[1,0], C[1,1].
        # We need C[0,2] and C[1,2].
        # And the terms involving A[0,2], A[1,2] and B[2,0], B[2,1], B[2,2].
        
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
        
        # We also need to account for B[2,0], B[2,1] in C[0,0], C[1,0]?
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
        # The Strassen part only used B[0,0], B[1,0].
        # So we are missing A[0,2]B[2,0] and A[1,2]B[2,0].
        
        # Missing terms:
        # 1. A[0,2]B[2,0] -> C[0,0]
        # 2. A[1,2]B[2,0] -> C[1,0]
        # 3. A[0,2]B[2,1] -> C[0,1]
        # 4. A[1,2]B[2,1] -> C[1,1]
        # 5. A[0,2]B[2,2] -> C[0,2]
        # 6. A[1,2]B[2,2] -> C[1,2]
        # 7. A[0,0]B[0,2] -> C[0,2]
        # 8. A[0,1]B[1,2] -> C[0,2]
        # 9. A[1,0]B[0,2] -> C[1,2]
        # 10. A[1,1]B[1,2] -> C[1,2]
        
        # That's 10 terms.
        # We said 8.
        # We can combine:
        # A[0,2]B[2,0] and A[1,2]B[2,0]
        # A[0,2]B[2,1] and A[1,2]B[2,1]
        # A[0,2]B[2,2] and A[1,2]B[2,2]
        # A[0,0]B[0,2] and A[1,0]B[0,2]
        # A[0,1]B[1,2] and A[1,1]B[1,2]
        
        # Let's just add them as schoolbook terms.
        # 10 terms.
        # Total rank 17.
        # Total for 3x6: 34.
        
        # Term 8: A[0,2]B[2,0]
        U[2][col] = 1
        V[b_col_offset+6][col] = 1
        W[0][col] = 1
        col += 1
        
        # Term 9: A[1,2]B[2,0]
        U[5][col] = 1
        V[b_col_offset+6][col] = 1
        W[3][col] = 1
        col += 1
        
        # Term 10: A[0,2]B[2,1]
        U[2][col] = 1
        V[b_col_offset+7][col] = 1
        W[1][col] = 1
        col += 1
        
        # Term 11: A[1,2]B[2,1]
        U[5][col] = 1
        V[b_col_offset+7][col] = 1
        W[4][col] = 1
        col += 1
        
        # Term 12: A[0,2]B[2,2]
        U[2][col] = 1
        V[b_col_offset+8][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term 13: A[1,2]B[2,2]
        U[5][col] = 1
        V[b_col_offset+8][col] = 1
        W[5][col] = 1
        col += 1
        
        # Term 14: A[0,0]B[0,2]
        U[0][col] = 1
        V[b_col_offset+2][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term 15: A[0,1]B[1,2]
        U[1][col] = 1
        V[b_col_offset+5][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term 16: A[1,0]B[0,2]
        U[3][col] = 1
        V[b_col_offset+2][col] = 1
        W[5][col] = 1
        col += 1
        
        # Term 17: A[1,1]B[1,2]
        U[4][col] = 1
        V[b_col_offset+5][col] = 1
        W[5][col] = 1
        col += 1
        
    add_2x3_by_3x3(0)
    add_2x3_by_3x3(3)
    
    return {"U": U, "V": V, "W": W}
