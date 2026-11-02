import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_true_tensor(n, m, p):
    """Build the true tensor for matrix multiplication."""
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _build_rank32():
    """Build the known rank-32 solution using Strassen for 2x2x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # Standard Strassen for 2x2x2
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0*p+k, 1*p+k
        # W indices: k*n+0, k*n+1
        
        # M1 = (A00+A11)(B00+B11)
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    # 9 columns in B.
    # Split into 4 blocks of 2 columns and 1 block of 1 column.
    # Columns: 0,1 | 2,3 | 4,5 | 6,7 | 8
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
