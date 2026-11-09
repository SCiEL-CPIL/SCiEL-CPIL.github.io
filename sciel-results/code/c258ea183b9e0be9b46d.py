import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for t in range(rank):
                                s += U[a*m+b][t] * V[c*p+d][t] * W[k*n+i][t]
                            expected = (1.0 if (a==i and c==b and d==k) else 0.0)
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W, rank

def _strassen_block(U, V, W, n, m, p, block_start_col, term_offset):
    """
    Add a 2x2x2 Strassen block for A[:, 0:2] and B[0:2, block_start_col:block_start_col+2]
    """
    # A indices: a in {0,1}, b in {0,1}
    # B indices: c in {0,1}, d in {block_start_col, block_start_col+1}
    # C indices: i in {0,1}, k in {block_start_col, block_start_col+1}
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: a=0,b=0; a=1,b=1
    # V: c=0,d=0; c=1,d=1
    # W: k=0,i=0 (+1); k=1,i=1 (-1)
    t = term_offset
    U[0*m + 0][t] += 1
    U[1*m + 1][t] += 1
    V[0*p + block_start_col][t] += 1
    V[1*p + (block_start_col+1)][t] += 1
    W[block_start_col*n + 0][t] += 1
    W[(block_start_col+1)*n + 1][t] += -1
    
    # M2 = (A10 + A11)B00
    # U: a=1,b=0; a=1,b=1
    # V: c=0,d=0
    # W: k=0,i=1 (+1)
    t = term_offset + 1
    U[1*m + 0][t] += 1
    U[1*m + 1][t] += 1
    V[0*p + block_start_col][t] += 1
    W[block_start_col*n + 1][t] += 1
    
    # M3 = A00(B01 - B11)
    # U: a=0,b=0
    # V: c=0,d=1; c=1,d=1
    # W: k=1,i=0 (+1)
    t = term_offset + 2
    U[0*m + 0][t] += 1
    V[0*p + (block_start_col+1)][t] += 1
    V[1*p + (block_start_col+1)][t] += -1
    W[(block_start_col+1)*n + 0][t] += 1
    
    # M4 = A11(B10 - B00)
    # U: a=1,b=1
    # V: c=1,d=0; c=0,d=0
    # W: k=0,i=0 (+1); k=0,i=1 (+1)
    t = term_offset + 3
    U[1*m + 1][t] += 1
    V[1*p + block_start_col][t] += 1
    V[0*p + block_start_col][t] += -1
    W[block_start_col*n + 0][t] += 1
    W[block_start_col*n + 1][t] += 1
    
    # M5 = (A00 + A01)B11
    # U: a=0,b=0; a=0,b=1
    # V: c=1,d=1
    # W: k=0,i=0 (-1); k=1,i=0 (+1)
    t = term_offset + 4
    U[0*m + 0][t] += 1
    U[0*m + 1][t] += 1
    V[1*p + (block_start_col+1)][t] += 1
    W[block_start_col*n + 0][t] += -1
    W[(block_start_col+1)*n + 0][t] += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: a=1,b=0; a=0,b=0
    # V: c=0,d=0; c=0,d=1
    # W: k=1,i=1 (+1)
    t = term_offset + 5
    U[1*m + 0][t] += 1
    U[0*m + 0][t] += -1
    V[0*p + block_start_col][t] += 1
    V[0*p + (block_start_col+1)][t] += 1
    W[(block_start_col+1)*n + 1][t] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: a=0,b=1; a=1,b=1
    # V: c=1,d=0; c=1,d=1
    # W: k=0,i=0 (+1); k=1,i=1 (+1)
    t = term_offset + 6
    U[0*m + 1][t] += 1
    U[1*m + 1][t] += -1
    V[1*p + block_start_col][t] += 1
    V[1*p + (block_start_col+1)][t] += 1
    W[block_start_col*n + 0][t] += 1
    W[(block_start_col+1)*n + 1][t] += 1

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try rank 23
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 3 Strassen blocks for columns 0-1, 2-3, 4-5
    # Block 0: cols 0,1
    _strassen_block(U, V, W, n, m, p, 0, 0)
    # Block 1: cols 2,3
    _strassen_block(U, V, W, n, m, p, 2, 7)
    # Block 2: cols 4,5
    _strassen_block(U, V, W, n, m, p, 4, 14)
    
    # Remainder: A[:, 2] * B[2, :]
    # A indices: a in {0,1}, b=2
    # B indices: c=2, d in {0..5}
    # C indices: i in {0,1}, k in {0..5}
    
    # Term 21: a=0, i=0
    t = 21
    U[0*m + 2][t] += 1
    for d in range(p):
        V[2*p + d][t] += 1
    for k in range(p):
        W[k*n + 0][t] += 1
        W[k*n + 1][t] += 0 # Explicitly 0
        
    # Term 22: a=1, i=1
    t = 22
    U[1*m + 2][t] += 1
    for d in range(p):
        V[2*p + d][t] += 1
    for k in range(p):
        W[k*n + 0][t] += 0
        W[k*n + 1][t] += 1

    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
