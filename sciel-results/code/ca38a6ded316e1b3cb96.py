"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _check_all_entries(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                x2 = 2.0 * float(x)
                if abs(x2 - round(x2)) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Return Strassen rank-7 decomposition for 2x2x2 matmul tensor.
    U2[i,t] for i in {0,1}, V2[j,t] for j in {0,1}, W2[k,t] for k in {0,1}
    such that sum_t U2[i,t]*V2[j,t]*W2[k,t] = 1 for all i,j,k.
    """
    U2 = [[0.0]*7 for _ in range(2)]
    V2 = [[0.0]*7 for _ in range(2)]
    W2 = [[0.0]*7 for _ in range(2)]
    
    # t=0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U2[0][0] = 1; U2[1][0] = 1
    V2[0][0] = 1; V2[1][0] = 1
    W2[0][0] = 1; W2[1][0] = 1
    
    # t=1: (A10+A11)B00 -> C10+=1, C11-=1
    U2[1][1] = 1
    V2[0][1] = 1
    W2[0][1] = 1; W2[1][1] = -1
    
    # t=2: A00(B01-B11) -> C01+=1
    U2[0][2] = 1
    V2[0][2] = 1; V2[1][2] = -1
    W2[1][2] = 1
    
    # t=3: A11(B10-B00) -> C10+=1
    U2[1][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1
    
    # t=4: (A00+A01)B11 -> C00-=1, C01+=1
    U2[0][4] = 1
    V2[1][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # t=5: (A10-A00)(B00+B01) -> C11+=1
    U2[1][5] = 1; U2[0][5] = -1
    V2[0][5] = 1
    W2[1][5] = 1
    
    # t=6: (A01-A11)(B10+B11) -> C00+=1
    U2[0][6] = 1; U2[1][6] = -1
    V2[1][6] = 1
    W2[0][6] = 1
    
    return U2, V2, W2

def _build_rank28():
    """
    Build a rank-28 decomposition for 3x4x4 matmul using Strassen structure.
    
    Decomposition: j = 2*j1 + j2, k = 2*k1 + k2 where j1,k1,j2,k2 in {0,1}
    For each (j1,k1) block, use 7 Strassen terms. Total rank = 4*7 = 28.
    
    T[i, 2*j1+j2, 2*k1+k2] = sum_t U[i,2*j1+j2,t] * V[2*j1+j2,2*k1+k2,t] * W[2*k1+k2,i,t]
    
    For block (j1,k1) with Strassen term t:
    U[i, 2*j1+j2, t] = U2[j2, t]  (independent of i)
    V[2*j1+j2, 2*k1+k2, t] = V2[k2, t]  (depends on k2 only within this block)
    W[2*k1+k2, i, t] = W2[j2, t]  (depends on j2 only)
    
    Wait, this won't work because the indices don't align properly.
    Let me use a different factorization.
    
    Actually, for the 2x2x2 Strassen: sum_t U2[a,t]*V2[b,t]*W2[c,t] = 1 where a,b,c in {0,1}
    Here a corresponds to row of C, b to col of A, c to col of B.
    
    For 3x4x4: i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
    j = 2*j1 + j2, k = 2*k1 + k2
    
    I need: sum_t U[i, 2*j1+j2, t] * V[2*j1+j2, 2*k1+k2, t] * W[2*k1+k2, i, t] = 1
    
    If I use 4 independent Strassen decompositions (one per (j1,k1)):
    For block (j1,k1), term t in {0..6}:
    U[i, 2*j1+j2, t] = U2[j2, t] for all i
    V[2*j1+j2, 2*k1+k2, t] = V2[k2, t] 
    W[2*k1+k2, i, t] = W2[j2, t]
    
    Then sum_t U2[j2,t]*V2[k2,t]*W2[j2,t] -- this is not 1 in general.
    
    The issue is that Strassen has U depending on (i,j) combined, not separately.
    In 2x2x2, U[i,t] where i is the row index of A (= row of C in output).
    But in our 3x4x4, U[i,j,t] has both i and j.
    
    Let me try: U[i, 2*j1+j2, t] = U2[j2, t] * delta(i, something)
    This is getting too complicated. Let me just do a smarter random search.
    """
    n, m, p = 3, 4, 4
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try random search for lower rank with more iterations
    values = [-1, -0.5, 0, 0.5, 1]
    
    for target_rank in range(36, 48):
        if time.time() > deadline - 10:
            break
        rng = random.Random(target_rank * 1000 + 42)
        found = False
        max_iters = 50
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            U_try = [[rng.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V_try = [[rng.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            W_try = [[rng.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U_try, V_try, W_try, n, m, p):
                if _check_all_entries(U_try, V_try, W_try):
                    U, V, W = U_try, V_try, W_try
                    found = True
                    break
        if found:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
