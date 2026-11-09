def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: 0..5 (2x3)
    # B indices: 0..14 (3x5)
    # C indices: 0..9 (5x2, W[k*n+i] -> k in 0..4, i in 0..1 -> idx = k*2+i)
    
    # A1: 0,1,2,3 (a11,a12,a21,a22)
    # A2: 4,5 (a13,a23)
    
    # B1: 0..9 (b11..b15, b21..b25)
    # B2: 10..14 (b31..b35)
    
    # C1: 0..7 (c11..c15, c21..c25) -> W indices
    # C2: 8..9 (c16? No, C is 2x5. c11..c15, c21..c25)
    # W index: k*n+i. k=0..4, i=0..1.
    # c11: k=0,i=0 -> 0
    # c12: k=1,i=0 -> 2
    # c13: k=2,i=0 -> 4
    # c14: k=3,i=0 -> 6
    # c15: k=4,i=0 -> 8
    # c21: k=0,i=1 -> 1
    # c22: k=1,i=1 -> 3
    # c23: k=2,i=1 -> 5
    # c24: k=3,i=1 -> 7
    # c25: k=4,i=1 -> 9
    
    # Part 1: A1 B1 (2x2 by 2x5) Rank 18
    # Split B1 into B1a (cols 0,1), B1b (cols 2,3), B1c (col 4)
    # B1a: b11,b12,b21,b22 (V indices 0,1,5,6)
    # B1b: b13,b14,b23,b24 (V indices 2,3,7,8)
    # B1c: b15,b25 (V indices 4,9)
    
    # Strassen for B1a
    # U coeffs (A1)
    # V coeffs (B1a)
    # W coeffs (C1a: c11,c12,c21,c22 -> W 0,2,1,3)
    
    # I will use schoolbook for 2x2x2 to be safe if Strassen is tricky, 
    # but that gives rank 8+8+4=20 for A1B1. Total 30.
    # I will try Strassen.
    
    # Let's use the verified Strassen coefficients.
    # M1..M7
    # U:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [-1, 0, 1, 0]
    # M7: [0, 1, 0, -1]
    
    # V:
    # M1: [1, 0, 0, 1]
    # M2: [0, 1, 0, 0]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]
    
    # W:
    # M1: [1, 0, 0, -1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 0, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 0]
    # M7: [-1, 1, 0, 0]
    
    # Note: M6 is not in W?
    # If M6 is not in W, it's not used in output.
    # But M6 is a product.
    # If M6 is not used, why compute it?
    # Maybe my W coeffs are wrong.
    # I will use schoolbook for the 2x2 blocks to ensure validity, 
    # and accept rank 30? No.
    # I will use the 28 rank decomposition from the hint.
    # I will assume the hint is correct and implement the 18+10 split.
    # For the 2x2x2, I will use Strassen.
    # I will verify in code.

    return {"U": U, "V": V, "W": W}
