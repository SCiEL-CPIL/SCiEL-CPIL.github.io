import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    # Split p=5 into p1=2, p2=3
    # Split m=3 into m1=2, m2=1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2) -> Strassen Rank 7
    # Block 2: A[0:2, 0:2] * B[0:2, 2:5] (2x2x3) -> Schoolbook Rank 12
    # Block 3: A[0:2, 2:3] * B[2:3, 0:5] (2x1x5) -> Schoolbook Rank 10
    
    rank = 7 + 12 + 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Block 1: 2x2x2 Strassen ---
    # A indices: i in 0,1; j in 0,1
    # B indices: j in 0,1; k in 0,1
    # C indices: i in 0,1; k in 0,1
    
    # M1 = (a00+a11)(b00+b11)
    U[0*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*2+0][col] = 1; V[1*2+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1 # c00 += M1, c11 += M1
    col += 1
    
    # M2 = (a10+a11)b00
    U[1*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*2+0][col] = 1
    W[1*2+0][col] = 1 # c10 += M2
    col += 1
    
    # M3 = a00(b01-b11)
    U[0*2+0][col] = 1
    V[0*2+1][col] = 1; V[1*2+1][col] = -1
    W[0*2+1][col] = 1 # c01 += M3
    col += 1
    
    # M4 = a11(b10-b00)
    U[1*2+1][col] = 1
    V[1*2+0][col] = 1; V[0*2+0][col] = -1
    W[0*2+0][col] = 1; W[1*2+0][col] = 1 # c00 += M4, c10 += M4
    col += 1
    
    # M5 = (a00+a01)b11
    U[0*2+0][col] = 1; U[0*2+1][col] = 1
    V[1*2+1][col] = 1
    W[0*2+0][col] = -1; W[0*2+1][col] = 1 # c00 -= M5, c01 += M5
    col += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[1*2+0][col] = 1; U[0*2+0][col] = -1
    V[0*2+0][col] = 1; V[0*2+1][col] = 1
    W[1*2+1][col] = 1 # c11 += M6? No, c11 = M1 - M2 + M3 + M7. 
    # Wait, standard Strassen:
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    # My M6 above is not in the final sum? 
    # Standard Strassen uses 7 products.
    # M6 is used in c11? No.
    # Let's re-verify Strassen.
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    # M6 is NOT used in the final sum? 
    # Ah, M6 is used? No, M6 is not in the formulas above.
    # Wait, M6 is defined but not used? 
    # No, M6 is used in some variants. 
    # Actually, M6 is not needed for the 4 outputs.
    # So M6 is redundant? 
    # No, the 7 products are M1..M7. 
    # If M6 is not used, rank is 6? No, rank is 7.
    # I must have the formula wrong.
    # c11 = M1 - M2 + M3 + M7. 
    # Where is M6? 
    # Maybe c11 uses M6? 
    # Let's assume standard Strassen works. I will just include M6 with 0 in W? 
    # No, that wastes rank.
    # I will skip M6 if it's not used. 
    # But rank 7 is the bound. 
    # I will use the 7 products and ensure they are used.
    
    # Let's restart Block 1 with a simpler valid rank 4 schoolbook for 2x2x2? 
    # Rank 4 is better than 7? No, 7 is lower. 
    # I will use Rank 4 for 2x2x2 to be safe and simple. 
    # Rank 4 + 12 + 10 = 26. 
    # 26 < 30. 
    # Let's do that. It's safer.
    
    # Reset
    rank = 4 + 12 + 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 1: 2x2x2 Schoolbook (Rank 4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
                
    # Block 2: 2x2x3 Schoolbook (Rank 12)
    # A[0:2, 0:2] * B[0:2, 2:5]
    # i in 0,1; j in 0,1; k in 2,3,4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
                
    # Block 3: 2x1x5 Schoolbook (Rank 10)
    # A[0:2, 2:3] * B[2:3, 0:5]
    # i in 0,1; j in 2; k in 0,1,2,3,4
    for i in range(2):
        for j in range(2, 3):
            for k in range(5):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# (Code above)
# EVOLVE-BLOCK-END
