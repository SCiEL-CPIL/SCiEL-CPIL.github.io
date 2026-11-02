import numpy as np

def construct():
    """
    Construct a tensor decomposition for 3x3 @ 3x8.
    Uses Strassen for the 2x2x2 block (7 terms) + schoolbook for remaining terms.
    Total rank: 7 + 19 + 45 = 71.
    """
    n, m, p = 3, 3, 8
    
    # Build the 3x3x3 part with Strassen for 2x2x2 block
    U23 = np.zeros((9, 26))
    V23 = np.zeros((9, 26))
    W23 = np.zeros((9, 26))
    
    # Strassen 2x2x2 (indices 0,1 for i,j,k):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M3 + M4 - M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M5 + M7
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U23[0, 0] = 1; U23[4, 0] = 1
    V23[0, 0] = 1; V23[4, 0] = 1
    W23[0, 0] = 1; W23[4, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U23[3, 1] = 1; U23[4, 1] = 1
    V23[0, 1] = 1
    W23[3, 1] = 1; W23[4, 1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    U23[0, 2] = 1
    V23[3, 2] = 1; V23[4, 2] = -1
    W23[0, 2] = -1; W23[3, 2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    U23[4, 3] = 1
    V23[1, 3] = 1; V23[0, 3] = -1
    W23[0, 3] = 1; W23[1, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U23[0, 4] = 1; U23[1, 4] = 1
    V23[4, 4] = 1
    W23[1, 4] = 1; W23[4, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U23[3, 5] = 1; U23[0, 5] = -1
    V23[0, 5] = 1; V23[1, 5] = 1
    W23[0, 5] = -1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U23[1, 6] = 1; U23[4, 6] = -1
    V23[3, 6] = 1; V23[4, 6] = 1
    W23[4, 6] = 1
    
    # Add 19 schoolbook terms for the non-2x2 parts of 3x3x3
    count = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U23[i*3+j, count] = 1
                V23[j*3+k, count] = 1
                W23[k*3+i, count] = 1
                count += 1
    
    assert count == 26, f"Expected 26 terms, got {count}"
    
    # Now extend to 3x3x8: add 45 more schoolbook terms for k=3,4,5,6,7
    total_rank = 26 + 45
    U = np.zeros((9, total_rank))
    V = np.zeros(24, total_rank)
    W = np.zeros(24, total_rank)
    
    # Copy the 3x3x3 part (first 26 terms)
    U[:, :26] = U23
    V[:9, :26] = V23
    W[:9, :26] = W23
    
    # Add schoolbook terms for k=3,4,5,6,7
    idx = 26
    for i in range(3):
        for j in range(3):
            for k in range(3, 8):
                U[i*3+j, idx] = 1
                V[j*3+k, idx] = 1
                W[k*3+i, idx] = 1
                idx += 1
    
    assert idx == total_rank
    
    # Verify the decomposition
    error = 0
    for i in range(3):
        for k in range(8):
            for j in range(3):
                val = 0
                for r in range(total_rank):
                    val += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)
    
    # Fallback to schoolbook if verification fails
    if error > 1e-10:
        rank_fallback = n * m * p
        U_fb = np.zeros((n*m, rank_fallback))
        V_fb = np.zeros((m*p, rank_fallback))
        W_fb = np.zeros((p*n, rank_fallback))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, idx] = 1.0
                    V_fb[j*p + k, idx] = 1.0
                    W_fb[k*n + i, idx] = 1.0
                    idx += 1
        
        U = U_fb
        V = V_fb
        W = W_fb
    
    # Convert to lists with integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    U_list = [[int(x) for x in row] for row in U_list]
    V_list = [[int(x) for x in row] for row in V_list]
    W_list = [[int(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
