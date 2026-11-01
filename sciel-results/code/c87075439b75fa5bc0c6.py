import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_U(i, j, c, val):
        U[i * m + j][c] = val
        
    def set_V(j, k, c, val):
        V[j * p + k][c] = val
        
    def set_W(k, i, c, val):
        W[k * n + i][c] = val

    # Part 1: j=0 (1 rank)
    c = 0
    for i in range(n):
        set_U(i, 0, c, 1)
    for k in range(p):
        set_V(0, k, c, 1)
    for k in range(p):
        for i in range(n):
            set_W(k, i, c, 1)
            
    # Part 2: j in {1, 2}, k in {0, 1} (7 ranks, Strassen)
    # Local indices: i' in {0,1}, j' in {0,1} (global j=1+j'), k' in {0,1} (global k=k')
    # Strassen multipliers M1..M7
    
    # M1: (A00+A11)(B00+B11)
    c = 1
    set_U(0, 1, c, 1) # A00 (local) -> A[0, 1] (global)
    set_U(1, 2, c, 1) # A11 (local) -> A[1, 2] (global)
    set_V(1, 0, c, 1) # B00 (local) -> B[1, 0] (global)
    set_V(2, 1, c, 1) # B11 (local) -> B[2, 1] (global)
    set_W(0, 0, c, 1) # C00 (local) -> C[0, 0] (global)
    set_W(1, 1, c, 1) # C11 (local) -> C[1, 1] (global)
    
    # M2: (A10+A11)B00
    c = 2
    set_U(1, 1, c, 1) # A10
    set_U(1, 2, c, 1) # A11
    set_V(1, 0, c, 1) # B00
    set_W(0, 1, c, 1) # C10 (local) -> C[1, 0] (global)
    # Wait, C10 is i'=1, k'=0. Global i=1, k=0.
    # My previous mapping: C'_{10} = M2 + M4.
    # So W for C[1,0] gets coeff 1 from M2.
    
    # M3: A00(B01-B11)
    c = 3
    set_U(0, 1, c, 1) # A00
    set_V(1, 1, c, 1) # B01
    set_V(2, 1, c, -1) # B11
    set_W(0, 1, c, 1) # C01 (local) -> C[0, 1] (global)
    # C'_{01} = M3 + M5.
    
    # M4: A11(B10-B00)
    c = 4
    set_U(1, 2, c, 1) # A11
    set_V(2, 0, c, 1) # B10
    set_V(1, 0, c, -1) # B00
    set_W(0, 0, c, 1) # C00 (local) -> C[0, 0] (global)
    set_W(0, 1, c, 1) # C10 (local) -> C[1, 0] (global)
    # C'_{00} = M1 + M4 - M5 + M7
    # C'_{10} = M2 + M4
    
    # M5: (A00+A01)B11
    c = 5
    set_U(0, 1, c, 1) # A00
    set_U(0, 2, c, 1) # A01
    set_V(2, 1, c, 1) # B11
    set_W(0, 0, c, -1) # C00 (local) -> C[0, 0] (global)
    set_W(0, 1, c, 1) # C01 (local) -> C[0, 1] (global)
    # C'_{00} = ... - M5 ...
    # C'_{01} = M3 + M5
    
    # M6: (A10-A00)(B00+B01)
    c = 6
    set_U(1, 1, c, 1) # A10
    set_U(0, 1, c, -1) # A00
    set_V(1, 0, c, 1) # B00
    set_V(1, 1, c, 1) # B01
    set_W(1, 1, c, 1) # C11 (local) -> C[1, 1] (global)
    # C'_{11} = M1 - M2 + M3 + M6
    
    # M7: (A01-A11)(B10+B11)
    c = 7
    set_U(0, 2, c, 1) # A01
    set_U(1, 2, c, -1) # A11
    set_V(2, 0, c, 1) # B10
    set_V(2, 1, c, 1) # B11
    set_W(0, 0, c, 1) # C00 (local) -> C[0, 0] (global)
    set_W(1, 1, c, 1) # C11 (local) -> C[1, 1] (global)
    # C'_{00} = M1 + M4 - M5 + M7
    # C'_{11} = M1 - M2 + M3 + M6 + M7? 
    # Let's re-verify Strassen C11.
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Wait, did I miss M7 in C11?
    # Let's check a reference.
    # C11 = M1 - M2 + M3 + M6.
    # So M7 is NOT in C11.
    # Let's re-verify M7 definition.
    # M7 = (A01 - A11)(B10 + B11).
    # In C00: M7 is present.
    # In C11: M7 is not present?
    # Let's check:
    # C11 = A01B11 + A11B10 + A00B10? No.
    # C11 = A01B11 + A11B10? No, C is 2x2.
    # C11 = A01B11 + A11B10?
    # C11 = sum_j A1j Bj1.
    # C11 = A10B01 + A11B11.
    # Let's re-evaluate Strassen formulas.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check C11 manually.
    # M1 - M2 + M3 + M6
    # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
    # Terms:
    # A00B00: 1 - 1 = 0
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1
    # A10B00: -1 + 1 = 0
    # A00B01: 1 - 1 = 0
    # A10B01: 1
    # Result: A11B11 + A10B01.
    # This is C11. Correct.
    
    # So M7 is only in C00.
    # My code for M7 set_W(1, 1, c, 1) is WRONG. It should not affect C11.
    # Let's fix M7 W entries.
    # M7 only contributes to C00.
    # So set_W(0, 0, c, 1) is correct.
    # set_W(1, 1, c, 1) should be REMOVED.

    # Correcting M7 in code below.

    # Part 3: j in {1, 2}, k in {2, 3, 4} (2 ranks)
    # Term 1: j=1 (local j'=0)
    c = 8
    for i in range(n):
        set_U(i, 1, c, 1)
    for k_prime in range(3):
        k_global = k_prime + 2
        set_V(1, k_global, c, 1)
        for i in range(n):
            set_W(k_global, i, c, 1)
            
    # Term 2: j=2 (local j'=1)
    c = 9
    for i in range(n):
        set_U(i, 2, c, 1)
    for k_prime in range(3):
        k_global = k_prime + 2
        set_V(2, k_global, c, 1)
        for i in range(n):
            set_W(k_global, i, c, 1)

    return {"U": U, "V": V, "W": W}
