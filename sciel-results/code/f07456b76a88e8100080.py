import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 5, 5, 5

    U_rows = []
    V_rows = []
    W_rows = []

    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        U_rows.append(i * m + j)
        V_rows.append(j * p + k)
        W_rows.append(k * n + i)

    # Helper to add a Strassen term for a 2x2x2 subproblem
    # The subproblem is defined by block indices (bi, bj, bk)
    # and local indices (li, lj, lk) in 0..1
    # Global i = 2*bi + li, etc.
    # The 7 terms of Strassen for 2x2x2:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We will generate the 7 rank-1 terms for a given 2x2x2 instance
    def add_strassen_2x2(bi, bj, bk):
        # bi, bj, bk are the block indices (0 or 1) in the 4x4 core
        # Local indices li, lj, lk are 0 or 1
        
        # M1
        # (A00+A11)(B00+B11)
        # A00: li=0, lj=0
        # A11: li=1, lj=1
        # B00: lj=0, lk=0
        # B11: lj=1, lk=1
        # Terms:
        # 1. A00 B00: i=0, j=0, k=0
        # 2. A00 B11: i=0, j=1, k=1
        # 3. A11 B00: i=1, j=0, k=0
        # 4. A11 B11: i=1, j=1, k=1
        # Wait, Strassen M1 is ONE rank-1 term in the 2x2x2 decomposition?
        # No, M1 is a product of two 2x2 matrices.
        # The decomposition of M1 into rank-1 terms is not trivial.
        # The rank-1 terms of the 2x2x2 tensor are the 7 vectors.
        
        # Let's use the known 7 rank-1 decompositions for 2x2x2.
        # They are defined by the 7 M terms.
        # Each M term corresponds to ONE column in U,V,W for the 2x2x2 case.
        
        # For M1:
        # U: A00=1, A11=1, others 0.
        # V: B00=1, B11=1, others 0.
        # W: C00=1, C11=1, others 0? No.
        # The W matrix for 2x2x2 Strassen:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # So for M1 (column 1):
        # U[0,0]=1, U[1,1]=1
        # V[0,0]=1, V[1,1]=1
        # W[0,0]=1 (C00), W[1,1]=1 (C11)
        
        # We need to map these local indices to global indices.
        
        # M1
        u = {(0,0):1, (1,1):1}
        v = {(0,0):1, (1,1):1}
        w = {(0,0):1, (1,1):1}
        _add_term(bi, bj, bk, u, v, w)
        
        # M2
        u = {(1,0):1, (1,1):1}
        v = {(0,0):1}
        w = {(1,0):1, (1,1):1} # C10, C11?
        # C10 = M2 + M4. C11 = M1 - M2 + ...
        # So M2 contributes to C10 and C11.
        w = {(1,0):1, (1,1):-1}
        _add_term(bi, bj, bk, u, v, w)

        # M3
        u = {(0,0):1}
        v = {(0,1):1, (1,1):-1}
        w = {(0,1):1, (1,1):1} # C01, C11
        _add_term(bi, bj, bk, u, v, w)

        # M4
        u = {(1,1):1}
        v = {(1,0):1, (0,0):-1}
        w = {(0,0):1, (1,0):1} # C00, C10
        _add_term(bi, bj, bk, u, v, w)

        # M5
        u = {(0,0):1, (0,1):1}
        v = {(1,1):1}
        w = {(0,0):-1, (0,1):1} # C00, C01
        _add_term(bi, bj, bk, u, v, w)

        # M6
        u = {(1,0):1, (0,0):-1}
        v = {(0,0):1, (0,1):1}
        w = {(1,0):1} # C10 only?
        # C10 = M2 + M4. C11 = ...
        # M6 is not in C10 or C11?
        # Wait, Strassen:
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # M6 is not used in the final C?
        # Ah, M6 = (A10-A00)(B00+B01).
        # It is an intermediate.
        # The 7 rank-1 terms correspond to the 7 products M1..M7.
        # The W matrix entries are the coefficients of M_t in C_ik.
        
        # Let's re-verify W coefficients.
        # C00: M1 + M4 - M5 - M7
        # C01: M3 + M5
        # C10: M2 + M4
        # C11: M1 - M2 + M3 + M7
        
        # M1: W[0,0]=1, W[1,1]=1
        # M2: W[1,0]=1, W[1,1]=-1
        # M3: W[0,1]=1, W[1,1]=1
        # M4: W[0,0]=1, W[1,0]=1
        # M5: W[0,0]=-1, W[0,1]=1
        # M6: W is all 0?
        # If M6 is not in any C_ik, then it is not needed?
        # No, Strassen uses 7 products.
        # Maybe I have the formulas wrong.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # M6 is indeed not used in the final sum?
        # No, M6 is used in some variants?
        # Actually, M6 is often defined as (A21-A11)(B11+B12).
        # And it is NOT in the final C.
        # This implies rank is 7, but one term is unused?
        # No, the rank is 7 because we need 7 products.
        # If M6 is not used, we could drop it?
        # If M6 is not used, rank is 6?
        # No, 2x2x2 rank is 7.
        # I must have the formula for M6 wrong or its usage.
        # Actually, M6 is used in C21? No, C21 = M2 + M4.
        # Maybe M6 is used in a different indexing.
        
        # Let's assume the standard 7 terms are needed.
        # I will implement the 7 terms with their W coefficients.
        
    # This is getting too complex for a quick fix.
    # I will return the schoolbook rank 125 to ensure validity, 
    # but the prompt asks for a rewrite.
    
    # Given the complexity, I will implement the Frame (61) + Schoolbook Core (64) = 125.
    # This is the same as the seed.
    
    # To get a score > 1.0, I MUST find a valid lower rank.
    # The safest bet is the Rank 110 solution (Frame 61 + Strassen Core 49).
    
    # I will implement the Rank 110 solution.

    # Frame
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue
                add_schoolbook(i, j, k)

    # Core: 7 Strassen products, each 7 rank-1 terms.
    # I need the exact U,V,W for the 2x2x2 Strassen terms.
    
    return {"U": U, "V": "W"} # Placeholder

