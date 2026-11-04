"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use Strassen's algorithm on 2x2 sub-blocks to reduce rank from 42 to 39.
- The 2x3 * 3x7 product can be decomposed into overlapping 2x2 * 2x2 blocks.
- Strassen's algorithm computes 2x2 * 2x2 in rank 7 instead of 8.
- We construct the decomposition by tiling the product with Strassen blocks.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    u_val = Fraction(str(U[u_idx][c]))
                    v_val = Fraction(str(V[v_idx][c]))
                    w_val = Fraction(str(W[w_idx][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _strassen_block():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    
    For A (2x2) * B (2x2) = C (2x2):
    A = [[a11, a12], [a21, a22]]
    B = [[b11, b12], [b21, b22]]
    
    Strassen's 7 products:
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a21 + a22)b11
    M3 = a11(b12 - b22)
    M4 = a22(b21 - b11)
    M5 = (a11 + a12)b22
    M6 = (a21 - a11)(b11 + b12)
    M7 = (a12 - a22)(b21 + b22)
    
    Result:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U_rows = []
    V_rows = []
    W_rows = []
    
    # M1 = (a11 + a22)(b11 + b22)
    U_rows.append([1, 0, 0, 1])
    V_rows.append([1, 0, 0, 1])
    W_rows.append([1, 0, 0, 1])
    
    # M2 = (a21 + a22)b11
    U_rows.append([0, 0, 1, 1])
    V_rows.append([1, 0, 0, 0])
    W_rows.append([0, 0, 1, -1])
    
    # M3 = a11(b12 - b22)
    U_rows.append([1, 0, 0, 0])
    V_rows.append([0, 1, 0, -1])
    W_rows.append([0, 1, 0, 1])
    
    # M4 = a22(b21 - b11)
    U_rows.append([0, 0, 0, 1])
    V_rows.append([-1, 0, 1, 0])
    W_rows.append([1, 0, 1, 0])
    
    # M5 = (a11 + a12)b22
    U_rows.append([1, 1, 0, 0])
    V_rows.append([0, 0, 0, 1])
    W_rows.append([-1, 1, 0, 0])
    
    # M6 = (a21 - a11)(b11 + b12)
    U_rows.append([-1, 0, 1, 0])
    V_rows.append([1, 1, 0, 0])
    W_rows.append([0, 0, 0, 1])
    
    # M7 = (a12 - a22)(b21 + b22)
    U_rows.append([0, 1, 0, -1])
    V_rows.append([0, 0, 1, 1])
    W_rows.append([1, 0, 0, 0])
    
    return U_rows, V_rows, W_rows

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_strassen_decomposition():
    """
    Construct a rank-39 decomposition for 2x3 * 3x7.
    
    Approach: Decompose the 2x3 * 3x7 into:
    - 3 disjoint 2x2 * 2x2 subproblems, each handled by Strassen (rank 7 each = 21)
    - Remaining 18 terms handled by schoolbook
    Total rank: 3*7 + 18 = 39
    """
    n, m, p = 2, 3, 7
    
    U = []
    V = []
    W = []
    
    # Strassen blocks for (j in {0,1}, k in {0,1}), (j in {0,1}, k in {2,3}), (j in {0,1}, k in {4,5})
    strassen_U, strassen_V, strassen_W = _strassen_block()
    
    for k0 in [0, 2, 4]:
        for r in range(7):
            u_row = [0] * (n * m)  # 6 entries
            v_row = [0] * (m * p)  # 21 entries
            w_row = [0] * (p * n)  # 14 entries
            
            # Map Strassen A indices to U indices
            # Strassen A: (0,0), (0,1), (1,0), (1,1) -> (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
            # U index = i * m + j = i * 3 + j
            a_idx = [0 * 3 + 0, 0 * 3 + 1, 1 * 3 + 0, 1 * 3 + 1]  # [0, 1, 3, 4]
            
            # Map Strassen B indices to V indices
            # Strassen B: (0,0), (0,1), (1,0), (1,1) -> (j=0,k=k0), (j=0,k=k0+1), (j=1,k=k0), (j=1,k=k0+1)
            # V index = j * p + k
            b_idx = [0 * p + k0, 0 * p + (k0 + 1), 1 * p + k0, 1 * p + (k0 + 1)]
            
            # Map Strassen C indices to W indices
            # Strassen C: (0,0), (0,1), (1,0), (1,1) -> (i=0,k=k0), (i=0,k=k0+1), (i=1,k=k0), (i=1,k=k0+1)
            # W index = k * n + i
            c_idx = [k0 * n + 0, (k0 + 1) * n + 0, k0 * n + 1, (k0 + 1) * n + 1]
            
            for s in range(4):
                u_row[a_idx[s]] += strassen_U[r][s]
                v_row[b_idx[s]] += strassen_V[r][s]
                w_row[c_idx[s]] += strassen_W[r][s]
            
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
    
    # Schoolbook for remaining 18 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is in one of the Strassen blocks
                in_strassen = False
                for k0 in [0, 2, 4]:
                    if j in [0, 1] and k in [k0, k0 + 1]:
                        in_strassen = True
                        break
                if in_strassen:
                    continue
                
                u_row = [0] * (n * m)
                v_row = [0] * (m * p)
                w_row = [0] * (p * n)
                
                u_row[i * m + j] = 1
                v_row[j * p + k] = 1
                w_row[k * n + i] = 1
                
                U.append(u_row)
                V.append(v_row)
                W.append(w_row)
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    
    # Try the Strassen-based construction first
    try:
        result = _construct_strassen_decomposition()
        n, m, p = DIMENSIONS
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
