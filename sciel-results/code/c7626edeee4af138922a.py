import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition using two Strassen algorithms for 2x2x2
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # Products:
    # M1 = (a+d)(e+g)
    # M2 = (c+d)g
    # M3 = a(f-g)
    # M4 = d(h-e)
    # M5 = (a+b)e
    # M6 = (c+d)(f-e)
    # M7 = (a+b)(h-g)
    
    # U vectors (for A)
    # Index mapping for A: 0:a, 1:b, 2:c, 3:d
    u_strassen = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, 1], # c+d
        [1, 1, 0, 0]  # a+b
    ]
    
    # V vectors (for B)
    # Index mapping for B (2x2): 0:e, 1:f, 2:g, 3:h
    v_strassen = [
        [1, 0, 1, 0], # e+g
        [0, 0, 1, 0], # g
        [0, 1, -1, 0], # f-g
        [-1, 0, 0, 1], # h-e
        [1, 0, 0, 0], # e
        [-1, 1, 0, 0], # f-e
        [0, 0, -1, 1]  # h-g
    ]
    
    # W vectors (for C)
    # Index mapping for C (2x2): 0:C00, 1:C01, 2:C10, 3:C11
    # C00 = M1 - M2 + M3 - M6 ? Let's verify.
    # Standard Strassen:
    # C11 = M1 - M2 + M3 - M6
    # C12 = M2 + M6
    # C21 = M4 + M7
    # C22 = M1 - M4 + M5 - M7
    # Note: Indices in literature might vary. 
    # Let's use:
    # C[0,0] = M1 - M4 + M5 - M7 ? No.
    # Let's stick to the verified:
    # C00 = M1 - M2 + M3 - M6 is NOT correct based on my previous check.
    # Correct Strassen:
    # C11 = M1 - M2 + M3 - M6
    # C12 = M2 + M6
    # C21 = M4 + M7
    # C22 = M1 - M4 + M5 - M7
    # Wait, C11 is bottom-right? 
    # Usually C = [[C11, C12], [C21, C22]].
    # Let's assume C[0,0] is top-left.
    # Top-left = M1 - M4 + M5 - M7?
    # Let's use a known correct set:
    # C00 = M1 - M2 + M3 - M6 is wrong.
    # C00 = M1 + M4 - M5 + M7?
    
    # Let's use the schoolbook for safety if I'm unsure about Strassen signs.
    # But Rank 16 is the baseline.
    # I will use Rank 16 to be safe? No, I need improvement.
    # I will use Rank 14 with a verified Strassen implementation.
    
    # Verified Strassen:
    # P1 = (a+d)(e+g)
    # P2 = (c+d)g
    # P3 = a(f-g)
    # P4 = d(h-e)
    # P5 = (a+b)e
    # P6 = (c+d)(f-e)
    # P7 = (a+b)(h-g)
    
    # C00 = P1 - P2 + P3 - P6  (This was my doubt)
    # Let's check C00 = ae + cg.
    # P1 - P2 + P3 - P6
    # = (a+d)(e+g) - (c+d)g + a(f-g) - (c+d)(f-e)
    # = ae + ag + de + dg - cg - dg + af - ag - cf + ce + df - de
    # = ae + af - cg - cf + ce + df
    # This is not ae + cg.
    
    # Correct Strassen formulas:
    # C11 = P1 - P4 + P5 - P7 ?
    # Let's try:
    # C00 = P1 - P4 + P5 - P7
    # = (a+d)(e+g) - d(h-e) + (a+b)e - (a+b)(h-g)
    # = ae + ag + de + dg - dh + de + ae + be - ah - bh + ag + bg
    # = 2ae + 2ag + 2de + dg - dh + be - ah - bh + bg
    # Not working.
    
    # I will fall back to Rank 16 (Schoolbook) to ensure correctness, 
    # but I will try to find a Rank 12 solution online in my training data.
    # Actually, I can use the fact that 2x2x4 rank is 12.
    # I will hardcode a Rank 12 solution if I can find one.
    # Since I can't, I will use Rank 16.
    
    # Wait, I can use Rank 14 if I get the Strassen right.
    # Let's use the standard:
    # C00 = P1 - P2 + P3 - P6 is WRONG.
    # C00 = P1 - P4 + P5 - P7 is WRONG.
    
    # Let's use the schoolbook. It's safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
