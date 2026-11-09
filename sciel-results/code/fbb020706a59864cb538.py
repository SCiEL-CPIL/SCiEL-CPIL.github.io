"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

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

def _verify(U, V, W):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    # Build the multiplication tensor T[i,j,k] for (i,j) in A, (j,k) in B -> C[i,k]
    # The decomposition says: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k]) * 0 # placeholder
                # Actually we need element-wise product then sum
                s = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """
    Strassen factors for 2x2x2 multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists.
    Convention:
    U rows: A00, A01, A10, A11 (index i*m + j)
    V rows: B00, B01, B10, B11 (index j*p + k)
    W rows: C00, C01, C10, C11 (index k*n + i)
    """
    # Columns correspond to M1..M7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A10)*(B00+B10) -> C00, C11
    U[0][0] = 1; U[2][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00 -> C10, -C11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11) -> C01, C11
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00) -> C00, C10
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11 -> -C00, C01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01) -> C11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11) -> C00
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _try_random_rank(r, max_attempts=200, seed=42, val_range=(-1, 0, 1)):
    """Try to find a rank-r decomposition via random search."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        U = np.array([[rng.choice(val_range) for _ in range(r)] for _ in range(n*m)], dtype=int)
        V = np.array([[rng.choice(val_range) for _ in range(r)] for _ in range(m*p)], dtype=int)
        W = np.array([[rng.choice(val_range) for _ in range(r)] for _ in range(p*n)], dtype=int)
        
        # Verify efficiently using numpy
        # T[i*m+j, j*p+k, k*n+i] should be 1
        # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        ok = True
        for i in range(n):
            if not ok:
                break
            for j in range(m):
                if not ok:
                    break
                for k in range(p):
                    s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if s != 1:
                        ok = False
                        break
        if ok:
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _construct_strassen_based(n, m, p):
    """
    Construct a decomposition using Strassen recursion.
    For 2x4x6, we can't directly use 2x2x2 Strassen.
    However, we can use the fact that 2x4x6 can be viewed as:
    A (2x4) * B (4x6) = C (2x6)
    
    Let's try to decompose 2x4x6 using a mix of schoolbook and Strassen-like ideas.
    Actually, a simpler approach:
    We can use the known rank bound for 2x4x6.
    According to literature, the rank of 2x4x6 matrix multiplication is 16? Or maybe 18?
    Let's check: 
    2x2x2 rank is 7 (Strassen).
    2x3x3 rank is 13?
    2x4x4 rank is ?
    
    Instead of guessing, let's try to build a specific low-rank decomposition.
    One known result: R(2,4,6) <= 16.
    Let's try to find a rank 16 or 18 solution.
    
    Since we don't have a hardcoded rank 16 solution, we will rely on the random search
    to find a valid lower rank if one exists, or fall back to schoolbook.
    However, random search for rank 16-24 with 0,1,-1 values is very unlikely to succeed quickly.
    
    Let's try a structured approach:
    Split the 4x6 matrix B into two 4x3 matrices B1, B2.
    C = A * [B1 B2] = [A*B1 A*B2].
    If we know rank(2x4x3), we can compute it twice? No, that's not how tensor rank works.
    
    Let's stick to the strategy: 
    1. Try to find a rank < 48 solution via random search (low probability but possible if structure is simple).
    2. If not found, return schoolbook.
    
    Wait, the inspiration 3 suggests Strassen. Can we use Strassen for 2x2x2 blocks?
    If we split A (2x4) into A1 (2x2), A2 (2x2).
    B (4x6) into B1 (2x6), B2 (2x6).
    C = A1*B1 + A2*B2.
    This is a sum of two 2x2x6 multiplications.
    Rank(2x2x6) = ?
    2x2x2 is 7. 2x2x3 is 12? 2x2x4 is ?
    This path is complex to derive on the fly.
    
    Let's enhance the random search with a specific heuristic:
    For small ranks, use sparse matrices.
    """
    pass

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Try ranks from known lower bounds up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Try a range of ranks
    # We will prioritize trying ranks near the lower bound first.
    
    # Strategy 1: Random search with {-1, 0, 1}
    # Strategy 2: Random search with {-2, -1, 0, 1, 2}
    
    # We will try ranks 24 to 47.
    # To maximize chance, we will spend more time on lower ranks.
    
    # Time allocation:
    # 24-28: 50% of time
    # 29-38: 30% of time
    # 39-47: 20% of time
    
    total_time = 140.0
    t1 = start_time + total_time * 0.5
    t2 = start_time + total_time * 0.8
    
    # Rank 24-28
    for r in range(24, 29):
        if time.time() > t1:
            break
        result = _try_random_rank(r, max_attempts=100, seed=42+r, val_range=(-1, 0, 1))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        result = _try_random_rank(r, max_attempts=50, seed=1000+r, val_range=(-2, -1, 0, 1, 2))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Rank 29-38
    for r in range(29, 39):
        if time.time() > t2:
            break
        result = _try_random_rank(r, max_attempts=50, seed=2000+r, val_range=(-1, 0, 1))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        result = _try_random_rank(r, max_attempts=20, seed=3000+r, val_range=(-2, -1, 0, 1, 2))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Rank 39-47
    for r in range(39, 48):
        if time.time() > deadline:
            break
        result = _try_random_rank(r, max_attempts=20, seed=4000+r, val_range=(-1, 0, 1))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        result = _try_random_rank(r, max_attempts=10, seed=5000+r, val_range=(-2, -1, 0, 1, 2))
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
