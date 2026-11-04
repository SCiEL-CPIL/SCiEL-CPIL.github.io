import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2 matrix multiplication."""
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (A10+A11)(B00) -> C10, -C11
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # M3 = A00(B01-B11) -> C01, C11
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    # M4 = A11(B10-B00) -> C00, C10
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    # M5 = (A00+A01)B11 -> -C00, C01
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # M6 = (A10-A00)(B00+B01) -> C11
    terms.append(([0,0,1,-1], [1,1,0,0], [0,0,0,1]))
    # M7 = (A01-A11)(B10+B11) -> C00
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _build_decomposition(n, m, p):
    """Build rank-39 decomposition for 2x3x7 matrix multiplication."""
    rank = 39
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _strassen_2x2x2()
    
    # Process k pairs (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        # Part 1: j in {0, 1} -> 2x2x2 Strassen
        for (u_vals, v_vals, w_vals) in strassen_terms:
            # A indices: i*m + j. i in {0,1}, j in {0,1}
            a_idx = [0, 1, m, m+1]
            # B indices: j*p + k. j in {0,1}, k in {k_start, k_start+1}
            b_idx = [k_start, k_start+1, p+k_start, p+k_start+1]
            # C indices: k*n + i. k in {k_start, k_start+1}, i in {0,1}
            c_idx = [k_start*n, k_start*n+1, (k_start+1)*n, (k_start+1)*n+1]
            
            for li, val in enumerate(u_vals):
                if val != 0:
                    U[a_idx[li]][col] += val
            for li, val in enumerate(v_vals):
                if val != 0:
                    V[b_idx[li]][col] += val
            for li, val in enumerate(w_vals):
                if val != 0:
                    W[c_idx[li]][col] += val
            col += 1
            
        # Part 2: j = 2 -> 2x1x2 (Rank 4)
        # C[k, i] += A[i, 2] * B[2, k]
        # Terms:
        # T1: (A[0,2]+A[1,2]) * B[2, k_start] -> C[k_start, 0], C[k_start, 1]
        U[0*m+2][col] += 1
        U[1*m+2][col] += 1
        V[2*p+k_start][col] += 1
        W[k_start*n+0][col] += 1
        W[k_start*n+1][col] += 1
        col += 1
        
        # T2: A[0,2] * B[2, k_start] -> C[k_start, 0]
        U[0*m+2][col] += 1
        V[2*p+k_start][col] += 1
        W[k_start*n+0][col] += 1
        col += 1
        
        # T3: (A[0,2]+A[1,2]) * B[2, k_start+1] -> C[k_start+1, 0], C[k_start+1, 1]
        U[0*m+2][col] += 1
        U[1*m+2][col] += 1
        V[2*p+k_start+1][col] += 1
        W[(k_start+1)*n+0][col] += 1
        W[(k_start+1)*n+1][col] += 1
        col += 1
        
        # T4: A[0,2] * B[2, k_start+1] -> C[k_start+1, 0]
        U[0*m+2][col] += 1
        V[2*p+k_start+1][col] += 1
        W[(k_start+1)*n+0][col] += 1
        col += 1
        
    # Process k = 6
    # Part 1: j in {0, 1} -> 2x2x1 (Rank 4)
    # C[6, i] += A[i, 0] * B[0, 6] + A[i, 1] * B[1, 6]
    # T1: (A[0,0]+A[1,0]) * B[0, 6] -> C[6, 0], C[6, 1]
    U[0*m+0][col] += 1
    U[1*m+0][col] += 1
    V[0*p+6][col] += 1
    W[6*n+0][col] += 1
    W[6*n+1][col] += 1
    col += 1
    
    # T2: A[0,0] * B[0, 6] -> C[6, 0]
    U[0*m+0][col] += 1
    V[0*p+6][col] += 1
    W[6*n+0][col] += 1
    col += 1
    
    # T3: (A[0,1]+A[1,1]) * B[1, 6] -> C[6, 0], C[6, 1]
    U[0*m+1][col] += 1
    U[1*m+1][col] += 1
    V[1*p+6][col] += 1
    W[6*n+0][col] += 1
    W[6*n+1][col] += 1
    col += 1
    
    # T4: A[0,1] * B[1, 6] -> C[6, 0]
    U[0*m+1][col] += 1
    V[1*p+6][col] += 1
    W[6*n+0][col] += 1
    col += 1
    
    # Part 2: j = 2 -> 2x1x1 (Rank 2)
    # C[6, i] += A[i, 2] * B[2, 6]
    # T5: (A[0,2]+A[1,2]) * B[2, 6] -> C[6, 0], C[6, 1]
    U[0*m+2][col] += 1
    U[1*m+2][col] += 1
    V[2*p+6][col] += 1
    W[6*n+0][col] += 1
    W[6*n+1][col] += 1
    col += 1
    
    # T6: A[0,2] * B[2, 6] -> C[6, 0]
    U[0*m+2][col] += 1
    V[2*p+6][col] += 1
    W[6*n+0][col] += 1
    col += 1
    
    if col != rank:
        raise ValueError(f"Expected rank {rank}, got {col}")
        
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    sol = _build_decomposition(n, m, p)
    
    if not _verify(sol["U"], sol["V"], sol["W"], n, m, p):
        # Fallback: full schoolbook rank 42
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return sol

# EVOLVE-BLOCK-END
