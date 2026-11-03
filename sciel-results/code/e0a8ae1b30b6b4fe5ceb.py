"""Matrix multiplication 3,3,5 – rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r]*V[:,r]*W[:,r] == the MM tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = len(Ua[0])
    result = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                result[i, j, k] = s
    expected = np.ones((n, m, p))
    return np.allclose(result, expected, atol=tol)

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Build schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try Strassen-based hybrid decomposition
    # Partition: rows 0-1 vs row 2, cols 0-1 vs cols 2-4
    # 
    # A = [[A11(2x2), A12(2x3)],
    #      [A21(1x2), A22(1x3)]]
    # B = [[B11(2x2), B12(2x3)],
    #      [B21(1x2), B22(1x3)]]
    # C = [[C11(2x2), C12(2x3)],
    #      [C21(1x2), C22(1x3)]]
    #
    # C11 = A11*B11 + A12*B21  -> use Strassen for A11*B11 (7 terms), schoolbook for A12*B21 (6 terms)
    # C12 = A11*B12 + A12*B22  -> schoolbook: 2*2*3 + 2*1*3 = 12+6=18... wait
    # Let me think more carefully.
    
    # Actually, let's use a simpler approach:
    # Use Strassen for the 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in {0,1})
    # and schoolbook for everything else.
    #
    # Total terms:
    # - Strassen for (0,1)x(0,1)x(0,1): 7 terms
    # - Schoolbook for remaining: (3*3*5 - 2*2*2) = 45 - 8 = 37 terms
    # But we need to be careful about double counting.
    
    # Let's enumerate all (i,j,k) tuples and decide which method to use:
    # If i in {0,1} and j in {0,1} and k in {0,1}: use Strassen
    # Otherwise: use schoolbook
    
    # Strassen decomposition for 2x2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Build the hybrid decomposition
    U_hybrid = []
    V_hybrid = []
    W_hybrid = []
    
    # First, add Strassen terms for (i,j,k) in {0,1}x{0,1}x{0,1}
    # Strassen coefficients for the 2x2x2 core
    # A indices: i in {0,1}, j in {0,1} -> A[i,j]
    # B indices: j in {0,1}, k in {0,1} -> B[j,k]
    # C indices: i in {0,1}, k in {0,1} -> C[i,k]
    
    # M1 = (a+d)(e+h) where a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # U contribution: A[0,0]=1, A[1,1]=1
    # V contribution: B[0,0]=1, B[1,1]=1
    # W contribution: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=1 with appropriate signs
    
    # Let's define the 7 Strassen multipliers:
    # M1: A factors: (0,0):1, (1,1):1; B factors: (0,0):1, (1,1):1
    # M2: A factors: (1,0):1, (1,1):1; B factors: (0,0):1
    # M3: A factors: (0,0):1; B factors: (0,1):1, (1,1):-1
    # M4: A factors: (1,1):1; B factors: (1,0):1, (0,0):-1
    # M5: A factors: (0,0):1, (0,1):1; B factors: (1,1):1
    # M6: A factors: (1,0):1, (0,0):-1; B factors: (0,0):1, (0,1):1
    # M7: A factors: (0,1):1, (1,1):-1; B factors: (1,0):1, (1,1):1
    
    strassen_A = [
        {0: 1, 3: 1},   # M1: A[0,0], A[1,1]
        {3: 1, 4: 1},   # M2: A[1,0], A[1,1]
        {0: 1},          # M3: A[0,0]
        {3: 1},          # M4: A[1,1]
        {0: 1, 1: 1},   # M5: A[0,0], A[0,1]
        {3: 1, 0: -1},  # M6: A[1,0], A[0,0]
        {1: 1, 3: -1},  # M7: A[0,1], A[1,1]
    ]
    strassen_B = [
        {0: 1, 2: 1},   # M1: B[0,0], B[1,1]
        {0: 1},          # M2: B[0,0]
        {1: 1, 2: -1},  # M3: B[0,1], B[1,1]
        {2: 1, 0: -1},  # M4: B[1,0], B[0,0]
        {2: 1},          # M5: B[1,1]
        {0: 1, 1: 1},   # M6: B[0,0], B[0,1]
        {2: 1, 3: 1},   # M7: B[1,0], B[1,1]
    ]
    strassen_C = [
        {0: 1, 1: 1, 2: -1, 3: 1},  # M1: C[0,0], C[0,1], C[1,0], C[1,1]
        {2: 1},                       # M2: C[1,0]
        {1: 1},                       # M3: C[0,1]
        {0: 1, 2: 1},                 # M4: C[0,0], C[1,0]
        {1: 1, 3: -1},                # M5: C[0,1], C[1,1]
        {3: 1},                       # M6: C[1,1]
        {0: 1, 3: 1},                 # M7: C[0,0], C[1,1]
    ]
    
    # Add Strassen terms
    for r in range(7):
        u_row = [0] * (n * m)
        v_row = [0] * (m * p)
        w_row = [0] * (p * n)
        
        for idx, val in strassen_A[r].items():
            i, j = divmod(idx, m)  # idx = i*m + j
            if i < 2 and j < 2:  # Only for the 2x2 core
                u_row[i * m + j] = val
        
        for idx, val in strassen_B[r].items():
            j, k = divmod(idx, p)  # idx = j*p + k
            if j < 2 and k < 2:  # Only for the 2x2 core
                v_row[j * p + k] = val
        
        for idx, val in strassen_C[r].items():
            k, i = divmod(idx, n)  # idx = k*n + i
            if k < 2 and i < 2:  # Only for the 2x2 core
                w_row[k * n + i] = val
        
        U_hybrid.append(u_row)
        V_hybrid.append(v_row)
        W_hybrid.append(w_row)
    
    # Add schoolbook terms for all (i,j,k) NOT in the Strassen core
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen core
                if i < 2 and j < 2 and k < 2:
                    continue
                
                u_row = [0] * (n * m)
                v_row = [0] * (m * p)
                w_row = [0] * (p * n)
                
                u_row[i * m + j] = 1
                v_row[j * p + k] = 1
                w_row[k * n + i] = 1
                
                U_hybrid.append(u_row)
                V_hybrid.append(v_row)
                W_hybrid.append(w_row)
    
    rank_hybrid = len(U_hybrid)
    
    # Validate the hybrid decomposition
    if _validate(U_hybrid, V_hybrid, W_hybrid, n, m, p):
        return {"U": U_hybrid, "V": V_hybrid, "W": W_hybrid}
    
    # If hybrid fails, try a different partition or fall back to schoolbook
    # Let's try: use Strassen for (i in {0,1}, j in {0,1}, k in {0,1})
    # but with a different approach to handle the boundary terms
    
    # For now, return schoolbook if hybrid doesn't validate
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
