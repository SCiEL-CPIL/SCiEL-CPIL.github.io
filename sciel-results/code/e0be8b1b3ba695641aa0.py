import time
import numpy as np
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the known rank-39 decomposition
    rank_fallback = 5 * 7 + 1 * 4
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U_fallback[0][r] = 1; U_fallback[3][r] = 1
        V_fallback[v_e][r] = 1; V_fallback[v_h][r] = 1
        W_fallback[w_c00][r] = 1; W_fallback[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U_fallback[2][r] = 1; U_fallback[3][r] = 1
        V_fallback[v_e][r] = 1
        W_fallback[w_c10][r] = 1; W_fallback[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U_fallback[0][r] = 1
        V_fallback[v_f][r] = 1; V_fallback[v_h][r] = -1
        W_fallback[w_c01][r] = 1; W_fallback[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U_fallback[3][r] = 1
        V_fallback[v_g][r] = 1; V_fallback[v_e][r] = -1
        W_fallback[w_c00][r] = 1; W_fallback[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U_fallback[0][r] = 1; U_fallback[1][r] = 1
        V_fallback[v_h][r] = 1
        W_fallback[w_c00][r] = -1; W_fallback[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U_fallback[2][r] = 1; U_fallback[0][r] = -1
        V_fallback[v_e][r] = 1; V_fallback[v_f][r] = 1
        W_fallback[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U_fallback[1][r] = 1; U_fallback[3][r] = -1
        V_fallback[v_g][r] = 1; V_fallback[v_h][r] = 1
        W_fallback[w_c00][r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U_fallback[0][r] = 1
    V_fallback[v_e2][r] = 1
    W_fallback[w_c02][r] = 1
    r += 1
    
    U_fallback[1][r] = 1
    V_fallback[v_g2][r] = 1
    W_fallback[w_c02][r] = 1
    r += 1
    
    U_fallback[2][r] = 1
    V_fallback[v_e2][r] = 1
    W_fallback[w_c12][r] = 1
    r += 1
    
    U_fallback[3][r] = 1
    V_fallback[v_g2][r] = 1
    W_fallback[w_c12][r] = 1
    r += 1
    
    # Now try to find a lower-rank decomposition using a structured approach
    # Key insight: The 2x2 matrix multiplication has rank 7 (not 8 schoolbook)
    # For 2x2 * 2x11, we can potentially do better than 5*7 + 4 = 39
    
    # Let's try a more systematic approach:
    # For each pair of columns in B, use the rank-7 decomposition
    # But allow sharing between adjacent pairs
    
    # Actually, let's try to find rank 38 or lower by being smarter
    
    # Strategy: Use the fact that for 2x2 multiplication, we can use:
    # (a+b)(e+f) - a*e - b*f gives us c*d type terms with 3 multiplications
    # instead of 4
    
    # Let me try a different decomposition for 2x2 * 2x2 that uses 7 terms:
    # This is the standard Strassen-like decomposition
    
    # For the full 2x2 * 2x11 problem, let's try rank 38:
    # Use 5 blocks of 7 = 35, plus 3 for the last column = 38
    
    # Wait, the last column needs 4 terms (schoolbook) or can we do better?
    # For a single column, we need:
    # result[i,10] = sum_j A[i,j] * B[j,10]
    # That's 4 multiplications minimum for general case
    
    # So minimum is 5*7 + 4 = 39? Unless we can share across blocks
    
    # Let me try to see if we can do 5*7 + 3 = 38 by sharing the last column
    # with the 5th block somehow
    
    # Actually, let's just return the known rank-39 decomposition
    # Finding rank < 39 requires more sophisticated algebraic techniques
    
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
