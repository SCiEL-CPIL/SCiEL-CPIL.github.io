"""Rewritten program for 3x5x6 matrix multiplication rank search."""
DIMENSIONS = [3, 5, 6]

import sys
import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid matrix multiplication decomposition.
    
    Condition: for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    
    The 7 Strassen products compute C = AB where A, B are 2x2.
    C[0,0] = M1 + M4 - M5 - M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M6 + M7
    
    where:
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1]-B[1,1])
    M4 = A[1,1] * (B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1]) * B[1,1]
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    """
    # For 2x2x2: n=2, m=2, p=2
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*m+j, c] = U[i*2+j, c]
    # V[j*p+k, c] = V[j*2+k, c]
    # W[k*n+i, c] = W[k*2+i, c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1 (k=0,i=0)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C[0,0]
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C[1,0] (k=1,i=1)
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C[0,1] (k=1,i=0)
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1  # C[1,0]
    W[0][3] = 1  # C[0,0]
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1  # C[0,1]
    W[0][4] = -1 # C[0,0]
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1  # C[0,1]
    
    # M7 = (A11-A01)*(B10+B11)
    U[3][6] = 1; U[1][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1 # C[0,0]
    W[2][6] = 1  # C[1,0]
    
    return {"U": U, "V": V, "W": W}

def _winograd_3x3x3():
    """Return Winograd F(2,2) or related decomposition for 3x3x3.
    
    Actually, let's use a simpler approach. For 3x3x3, the known best is 23.
    But constructing it explicitly is complex. Let me try a different approach:
    use the fact that 3x5x6 can be embedded from smaller blocks.
    
    Instead, let me try a direct randomized search with better initialization.
    """
    return None

def _try_random_rank(r, n, m, p, deadline, seed=None):
    """Try to find a rank-r decomposition via random search."""
    if seed is not None:
        random.seed(seed)
    
    # Use small rationals for exactness
    def rand_rational():
        num = random.randint(-3, 3)
        den = random.choice([1, 2, 3, 4])
        return Fraction(num, den)
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        for i in range(n*m):
            for c in range(r):
                U[i][c] = rand_rational()
        for j in range(m*p):
            for c in range(r):
                V[j][c] = rand_rational()
        for k in range(p*n):
            for c in range(r):
                W[k][c] = rand_rational()
        
        if _verify_decomposition(U, V, W, n, m, p, r, tol=1e-12):
            return {"U": U, "V": V, "W": W}
    
    return None

def _structured_search(n, m, p, deadline):
    """Try structured approaches to find lower-rank decompositions."""
    
    # Approach 1: Try to use Strassen-like structure
    # For 3x5x6, we can think of it as blocks
    # 3x5 = [3x2 | 3x3], 5x6 = [2x6; 3x6]
    # This gives a block decomposition
    
    # Approach 2: Randomized with structured initialization
    # Start from schoolbook and try to merge columns
    
    rank0 = n * m * p
    best = _schoolbook(n, m, p)
    best_rank = rank0
    
    # Try ranks from a reasonable upper bound downward
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Upper bound: 90 (schoolbook)
    # Try some key ranks
    
    candidate_ranks = list(range(50, 30, -1)) + list(range(30, max(n*m, m*p, p*n)-1, -1))
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        result = _try_random_rank(r, n, m, p, deadline, seed=r)
        if result is not None:
            best = result
            best_rank = r
    
    return best, best_rank

def _polish_decomposition(U, V, W, n, m, p, rank, max_iter=1000, deadline=None):
    """Try to polish a near-exact decomposition to exact using local search."""
    if deadline is None:
        deadline = time.time() + 60.0
    
    # Convert to float for optimization
    import numpy as np
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    
    def loss():
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_f[i*m+j], V_f[j*p+k] * W_f[k*n+i])
                    total += (s - 1.0)**2
        return total
    
    current_loss = loss()
    if current_loss < 1e-12:
        return {"U": U, "V": V, "W": W}
    
    # Simple local search: perturb one entry at a time
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        best_delta = 0.0
        best_mat, best_row, best_col, best_val = None, -1, -1, 0.0
        
        # Try perturbing U
        for i in range(n*m):
            for c in range(rank):
                for delta in [-1, 1, -2, 2, -0.5, 0.5]:
                    old = U_f[i*m+j if False else i][c]
                    U_f[i][c] = old + delta
                    new_loss = loss()
                    U_f[i][c] = old
                    if new_loss < current_loss - 1e-15:
                        best_delta = new_loss - current_loss
                        best_mat = 'U'
                        best_row = i
                        best_col = c
                        best_val = delta
        
        if best_mat is None:
            break
        
        # Apply the best perturbation
        if best_mat == 'U':
            U_f[best_row][best_col] += best_val
        elif best_mat == 'V':
            V_f[best_row][best_col] += best_val
        else:
            W_f[best_row][best_col] += best_val
        
        current_loss = loss()
        if current_loss < 1e-12:
            # Convert back to fractions
            U_out = [[Fraction(int(round(U_f[i][c]*4)))/4 for c in range(rank)] for i in range(n*m)]
            V_out = [[Fraction(int(round(V_f[j][c]*4)))/4 for c in range(rank)] for j in range(m*p)]
            W_out = [[Fraction(int(round(W_f[k][c]*4)))/4 for c in range(rank)] for k in range(p*n)]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured search
    if time.time() < deadline - 30:
        result, rank = _structured_search(n, m, p, deadline - 30)
        if result is not None and rank < best_rank:
            best = result
            best_rank = rank
    
    # Try to polish if we have a near-miss
    # For now, return the best found
    
    return best

# EVOLVE-BLOCK-END
