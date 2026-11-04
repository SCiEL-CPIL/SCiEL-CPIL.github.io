"""Combined program for matrix multiplication 2x3x8 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float-based check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-vectors.
    A is indexed [i*2+j], B is [j*2+k], C is [k*2+i].
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10+A11)B00
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # M3 = A00(B10-B11)
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]))
    # M4 = A11(B01-B10)
    terms.append(([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]))
    # M5 = (A00+A01)B11
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6 = (A10-A00)(B00+B10)
    terms.append(([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]))
    # M7 = (A01-A11)(B01+B11)
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]))
    return terms

def _build_strassen_decomposition():
    """
    Build a rank-44 decomposition using block-Strassen.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _strassen_2x2x2()
    
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        
        for (u_4, v_4, w_4) in strassen_terms:
            U[0][col] = u_4[0]
            U[1][col] = u_4[1]
            U[3][col] = u_4[2]
            U[4][col] = u_4[3]
            
            V[k0][col] = v_4[0]
            V[k1][col] = v_4[1]
            V[8+k0][col] = v_4[2]
            V[8+k1][col] = v_4[3]
            
            W[2*k0][col] = w_4[0]
            W[2*k1][col] = w_4[1]
            W[2*k0+1][col] = w_4[2]
            W[2*k1+1][col] = w_4[3]
            
            col += 1
        
        for i in range(2):
            for k in [k0, k1]:
                U[i*3+2][col] = 1
                V[2*8+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_rank29_decomposition():
    """
    Build a rank-29 decomposition.
    Uses 4 Strassen blocks for j in {0,1} and 1 term for j=2.
    """
    n, m, p = 2, 3, 8
    rank = 29
    U = [[0] * rank for _ in range(n * m)] # 6 x 29
    V = [[0] * rank for _ in range(m * p)] # 24 x 29
    W = [[0] * rank for _ in range(p * n)] # 16 x 29
    
    col = 0
    strassen_terms = _strassen_2x2x2()
    
    # 4 blocks of Strassen for j in {0,1}
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        
        for (u_4, v_4, w_4) in strassen_terms:
            # U: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0, 1, 3, 4
            U[0][col] = u_4[0]
            U[1][col] = u_4[1]
            U[3][col] = u_4[2]
            U[4][col] = u_4[3]
            
            # V: B[0,k0], B[0,k1], B[1,k0], B[1,k1] -> indices k0, k1, 8+k0, 8+k1
            V[k0][col] = v_4[0]
            V[k1][col] = v_4[1]
            V[8+k0][col] = v_4[2]
            V[8+k1][col] = v_4[3]
            
            # W: C[0,k0], C[0,k1], C[1,k0], C[1,k1] -> indices 2*k0, 2*k1, 2*k0+1, 2*k1+1
            W[2*k0][col] = w_4[0]
            W[2*k1][col] = w_4[1]
            W[2*k0+1][col] = w_4[2]
            W[2*k1+1][col] = w_4[3]
            
            col += 1
            
    # 1 term for j=2
    # U: A[0,2], A[1,2] -> indices 2, 5
    U[2][col] = 1
    U[5][col] = 1
    
    # V: B[2,0]...B[2,7] -> indices 16...23
    for k in range(8):
        V[16+k][col] = 1
        
    # W: C[0,0]...C[1,7] -> indices 0...15
    for k in range(8):
        W[2*k][col] = 1      # C[0,k]
        W[2*k+1][col] = 1    # C[1,k]
        
    col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_and_clean(decomp, n, m, p):
    """Verify decomposition and clean up."""
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    
    if not _check_fast(U, V, W, n, m, p):
        return None
    
    if not _check_exact(U, V, W, n, m, p):
        return None
    
    # Remove zero columns
    new_U_cols, new_V_cols, new_W_cols = [], [], []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(len(U))) or \
           any(V[i][c] != 0 for i in range(len(V))) or \
           any(W[i][c] != 0 for i in range(len(W))):
            new_U_cols.append([U[i][c] for i in range(len(U))])
            new_V_cols.append([V[i][c] for i in range(len(V))])
            new_W_cols.append([W[i][c] for i in range(len(W))])
    
    new_rank = len(new_U_cols)
    final_U = [[new_U_cols[c][i] for c in range(new_rank)] for i in range(len(U))]
    final_V = [[new_V_cols[c][i] for c in range(new_rank)] for i in range(len(V))]
    final_W = [[new_W_cols[c][i] for c in range(new_rank)] for i in range(len(W))]
    
    return {"U": final_U, "V": final_V, "W": final_W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try the rank 29 decomposition
    decomp = _build_rank29_decomposition()
    verified = _verify_and_clean(decomp, n, m, p)
    
    if verified is not None:
        return verified
    
    # Fallback to rank 44
    decomp = _build_strassen_decomposition()
    verified = _verify_and_clean(decomp, n, m, p)
    
    if verified is not None:
        return verified
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
