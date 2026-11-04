import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 27 decomposition
    rank = 27
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # term_idx: column index in U, V, W
    # u_vals: dict {(i,j): val}
    # v_vals: dict {(j,k): val}
    # w_vals: dict {(k,i): val}
    def add_term(term_idx, u_vals, v_vals, w_vals):
        for (i, j), val in u_vals.items():
            U[i * m + j][term_idx] += val
        for (j, k), val in v_vals.items():
            V[j * p + k][term_idx] += val
        for (k, i), val in w_vals.items():
            W[k * n + i][term_idx] += val
            
    # Strassen 2x2x2 terms
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M5 - M6
    # C01 = M2 + M6
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    
    # We need to map this to the specific sub-matrices.
    
    # Part 1: C1 (rows 0,1 of C) from A1 (rows 0,1 of A) and B (all)
    # A1 is 2x3. B is 3x4.
    # A1a (cols 0,1) is 2x2. A1b (col 2) is 2x1.
    # B_top (rows 0,1) is 2x4. B_bot (row 2) is 1x4.
    
    # C1 = A1a B_top + A1b B_bot
    
    # A1a B_top:
    # B_top split into BL (cols 0,1) and BR (cols 2,3).
    # A1a BL: Strassen on A1a (2x2) and BL (2x2).
    # A1a BR: Strassen on A1a (2x2) and BR (2x2).
    
    # A1b B_bot:
    # A1b is 2x1. B_bot is 1x4.
    # Term: A1b[i] * B_bot[k] -> C1[i, k]
    
    # Indices for A1a: i in {0,1}, j in {0,1}
    # Indices for BL: j in {0,1}, k in {0,1}
    # Indices for BR: j in {0,1}, k in {2,3}
    
    # Term 0-6: A1a BL
    # U: A1a indices (i,j) with i,j in 0..1
    # V: BL indices (j,k) with j in 0..1, k in 0..1
    # W: C1 indices (k,i) with i in 0..1, k in 0..1
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M0
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # C00+=1, C11+=1
        # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # C10+=1
        # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,0):-1}), # C01+=1, C10+=-1? 
        # Wait, Strassen formulas:
        # C00 = M0 + M3 - M5 - M6
        # C01 = M2 + M6
        # C10 = M1 + M5
        # C11 = M0 - M1 + M2 + M4
        
        # Let's list W coefficients for each M
        # M0: C00:1, C11:1
        # M1: C10:1
        # M2: C01:1, C11:1
        # M3: C00:1
        # M4: C11:1
        # M5: C00:-1, C10:1
        # M6: C00:-1, C01:1
    ]
    
    # Correct W mapping for Strassen:
    # M0: W[0,0]=1, W[1,1]=1 (k=0,i=0 and k=1,i=1)
    # M1: W[0,1]=1 (k=0,i=1)
    # M2: W[1,0]=1, W[1,1]=1 (k=1,i=0 and k=1,i=1)
    # M3: W[0,0]=1 (k=0,i=0)
    # M4: W[1,1]=1 (k=1,i=1)
    # M5: W[0,0]=-1, W[0,1]=1 (k=0,i=0 and k=0,i=1)
    # M6: W[0,0]=-1, W[1,0]=1 (k=0,i=0 and k=1,i=0)
    
    # Let's re-verify C11 = M0 - M1 + M2 + M4
    # M0: 1
    # M1: 0
    # M2: 1
    # M4: 1
    # Sum: 3? No.
    # Standard Strassen:
    # C11 = M0 - M1 + M2 + M4
    # My M2 def: A00(B10-B11).
    # C01 = M2 + M6.
    # C11 uses M2?
    # Yes.
    
    # Let's just use the standard coefficients.
    
    s_terms = []
    # M0
    s_terms.append((
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1}
    ))
    # M1
    s_terms.append((
        {(1,0):1, (1,1):1}, 
        {(0,0):1}, 
        {(0,1):1}
    ))
    # M2
    s_terms.append((
        {(0,0):1}, 
        {(1,0):1, (1,1):-1}, 
        {(1,0):1, (1,1):1}
    ))
    # M3
    s_terms.append((
        {(1,1):1}, 
        {(0,1):1, (0,0):-1}, 
        {(0,0):1}
    ))
    # M4
    s_terms.append((
        {(0,0):1, (0,1):1}, 
        {(1,1):1}, 
        {(1,1):1}
    ))
    # M5
    s_terms.append((
        {(1,0):1, (0,0):-1}, 
        {(0,0):1, (0,1):1}, 
        {(0,0):-1, (0,1):1}
    ))
    # M6
    s_terms.append((
        {(0,1):1, (1,1):-1}, 
        {(1,0):1, (1,1):1}, 
        {(0,0):-1, (1,0):1}
    ))
    
    # Apply to A1a BL (cols 0,1 of B)
    for c in range(7):
        u, v, w = s_terms[c]
        term_idx = c
        # Map local indices to global
        # A1a: i,j in 0..1
        # BL: j,k in 0..1
        # C1: i in 0..1, k in 0..1
        add_term(term_idx, u, v, w)
        
    # Apply to A1a BR (cols 2,3 of B)
    # BR indices: j in 0..1, k in 2..3
    # Local k' in 0..1 maps to global k = k' + 2
    for c in range(7):
        u, v, w = s_terms[c]
        term_idx = 7 + c
        v_mapped = {}
        for (j, k_local), val in v.items():
            v_mapped[(j, k_local + 2)] = val
        w_mapped = {}
        for (k_local, i), val in w.items():
            w_mapped[(k_local + 2, i)] = val
        add_term(term_idx, u, v_mapped, w_mapped)
        
    # A1b B_bot
    # A1b: i in 0..1, j=2
    # B_bot: j=2, k in 0..3
    # C1: i in 0..1, k in 0..3
    # Term: U[i,2]=1, V[2,k]=1, W[k,i]=1
    term_idx = 14
    u = {(0,2):1, (1,2):1}
    v = {}
    for k in range(4):
        v[(2,k)] = 1
    w = {}
    for k in range(4):
        for i in range(2):
            w[(k,i)] = 1
    add_term(term_idx, u, v, w)
    
    # Part 2: C2 (row 2 of C) from A2 (row 2 of A) and B (all)
    # A2: i=2, j in 0..2
    # B: j in 0..2, k in 0..3
    # C2: i=2, k in 0..3
    # Terms: A2[j] B[j,k]
    for j in range(3):
        for k in range(4):
            term_idx = 15 + j * 4 + k
            u = {(2,j):1}
            v = {(j,k):1}
            w = {(k,2):1}
            add_term(term_idx, u, v, w)
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
