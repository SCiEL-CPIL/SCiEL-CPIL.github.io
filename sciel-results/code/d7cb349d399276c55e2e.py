import time

DIMENSIONS = [2, 2, 14]

def construct():
    # We use a block-wise Strassen decomposition for the 2x2 by 2x14 multiplication.
    # The 2x14 matrix B can be viewed as 7 blocks of size 2x2.
    # We apply Strassen's algorithm to each 2x2 block independently.
    # Strassen's algorithm uses 7 multiplications per block.
    # Total rank = 7 blocks * 7 Strassen mults = 49.
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen's 7 products for a 2x2 block:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For each block s (0 to 6), and each Strassen product t (0 to 6),
    # we define the coefficients for U, V, W.
    
    # U indices: A[i,j] -> i*m + j = i*2 + j
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    # V indices: B[j,k] -> j*p + k = j*14 + k
    # For block s, local k is 0 or 1, global k is 2s or 2s+1
    # B[0, 2s] -> 0*14 + 2s = 2s
    # B[0, 2s+1] -> 0*14 + (2s+1) = 2s+1
    # B[1, 2s] -> 1*14 + 2s = 14 + 2s
    # B[1, 2s+1] -> 1*14 + (2s+1) = 14 + 2s + 1
    
    # W indices: C[k,i] -> k*n + i = k*2 + i
    # For block s, local l is 0 or 1, global k is 2s or 2s+1
    # C[0, 2s] -> (2s)*2 + 0 = 4s
    # C[1, 2s] -> (2s)*2 + 1 = 4s+1
    # C[0, 2s+1] -> (2s+1)*2 + 0 = 4s+2
    # C[1, 2s+1] -> (2s+1)*2 + 1 = 4s+3
    
    # Strassen products and their contributions:
    # M1 = (A00 + A11) * (B00 + B11)
    #   U: A00=1, A11=1
    #   V: B00=1, B11=1
    #   W: C00+=M1, C01+=M1, C10+=M1, C11+=M1 (actually C00+=M1-M2-M4+M7, C01+=M3+M5, C10+=M2+M4, C11+=M1-M3-M5+M6)
    #   Let me re-derive the Strassen formulas properly.
    
    # Standard Strassen:
    # C00 = M1 - M2 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M6
    
    # So the W matrix entries for each block:
    # M1: C00+=1, C01+=0, C10+=0, C11+=1
    # M2: C00+=-1, C01+=0, C10+=1, C11+=0
    # M3: C00+=0, C01+=1, C10+=0, C11+=-1
    # M4: C00+=-1, C01+=0, C10+=1, C11+=0
    # M5: C00+=0, C01+=1, C10+=0, C11+=-1
    # M6: C00+=0, C01+=0, C10+=0, C11+=1
    # M7: C00+=1, C01+=0, C10+=0, C11+=0
    
    # Let me define the patterns for each Strassen product t:
    # t=0: M1 = (A00+A11)*(B00+B11)
    #   U: {0:1, 3:1}
    #   V: {B00:1, B11:1}
    #   W: {C00:1, C11:1}
    
    # t=1: M2 = (A10+A11)*B00
    #   U: {2:1, 3:1}
    #   V: {B00:1}
    #   W: {C00:-1, C10:1}
    
    # t=2: M3 = A00*(B01-B11)
    #   U: {0:1}
    #   V: {B01:1, B11:-1}
    #   W: {C01:1, C11:-1}
    
    # t=3: M4 = A11*(B10-B00)
    #   U: {3:1}
    #   V: {B10:1, B00:-1}
    #   W: {C00:-1, C10:1}
    
    # t=4: M5 = (A00+A01)*B11
    #   U: {0:1, 1:1}
    #   V: {B11:1}
    #   W: {C01:1, C11:-1}
    
    # t=5: M6 = (A10-A00)*(B00+B01)
    #   U: {2:1, 0:-1}
    #   V: {B00:1, B01:1}
    #   W: {C11:1}
    
    # t=6: M7 = (A01-A11)*(B10+B11)
    #   U: {1:1, 3:-1}
    #   V: {B10:1, B11:1}
    #   W: {C00:1}
    
    # Now build the matrices:
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            if t == 0:
                U[0][c] = 1; U[3][c] = 1
                V[2*s][c] = 1; V[14+2*s+1][c] = 1
                W[4*s][c] = 1; W[4*s+3][c] = 1
            elif t == 1:
                U[2][c] = 1; U[3][c] = 1
                V[2*s][c] = 1
                W[4*s][c] = -1; W[4*s+1][c] = 1
            elif t == 2:
                U[0][c] = 1
                V[2*s+1][c] = 1; V[14+2*s+1][c] = -1
                W[4*s+2][c] = 1; W[4*s+3][c] = -1
            elif t == 3:
                U[3][c] = 1
                V[14+2*s][c] = 1; V[2*s][c] = -1
                W[4*s][c] = -1; W[4*s+1][c] = 1
            elif t == 4:
                U[0][c] = 1; U[1][c] = 1
                V[14+2*s+1][c] = 1
                W[4*s+2][c] = 1; W[4*s+3][c] = -1
            elif t == 5:
                U[2][c] = 1; U[0][c] = -1
                V[2*s][c] = 1; V[2*s+1][c] = 1
                W[4*s+3][c] = 1
            elif t == 6:
                U[1][c] = 1; U[3][c] = -1
                V[14+2*s][c] = 1; V[14+2*s+1][c] = 1
                W[4*s][c] = 1
    
    return {"U": U, "V": V, "W": W}
