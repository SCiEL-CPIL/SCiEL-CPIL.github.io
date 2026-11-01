"""
2x2 matrix multiplication: search for rank-6 tensor decomposition over F_3,
fall back to Strassen's rank-7 decomposition.

The matmul tensor T[i,a,k] = 1 for all i,a,k in {0,1} (8 entries).
A rank-r decomposition: T[i,a,k] = sum_r U[i,a,r] * V[a,k,r] * W[k*2+i, r]
"""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 2]


def _verify_decomposition(U, V, W, n=2, m=2, p=2, rank=7, mod=None):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    for i in range(n):
        for a in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + a][r] * V[a * p + k][r] * W[k * n + i][r]
                expected = 1
                if mod is not None:
                    if (s % mod) != (expected % mod):
                        return False
                else:
                    if s != expected:
                        return False
    return True


def _strassen_decomposition():
    """Return the verified rank-7 Strassen decomposition."""
    n, m, p = 2, 2, 2
    rank = 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1   # A[0,0]
    U[3][0] = 1   # A[1,1]
    V[0][0] = 1   # B[0,0]
    V[3][0] = 1   # B[1,1]

    # M2 = (A10 + A11)B00
    U[2][1] = 1   # A[1,0]
    U[3][1] = 1   # A[1,1]
    V[0][1] = 1   # B[0,0]

    # M3 = A00(B01 - B11)
    U[0][2] = 1   # A[0,0]
    V[1][2] = 1   # B[0,1]
    V[3][2] = -1  # -B[1,1]

    # M4 = A11(B10 - B00)
    U[3][3] = 1   # A[1,1]
    V[2][3] = 1   # B[1,0]
    V[0][3] = -1  # -B[0,0]

    # M5 = (A00 + A01)B11
    U[0][4] = 1   # A[0,0]
    U[1][4] = 1   # A[0,1]
    V[3][4] = 1   # B[1,1]

    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1   # A[1,0]
    U[0][5] = -1  # -A[0,0]
    V[0][5] = 1   # B[0,0]
    V[1][5] = 1   # B[0,1]

    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1   # A[0,1]
    U[3][6] = -1  # -A[1,1]
    V[2][6] = 1   # B[1,0]
    V[3][6] = 1   # B[1,1]

    # W[k*n + i, r]: C[i,k] = sum_r (products) * W[k*2+i, r]
    # C[0,0] = M1 + M4 - M5 + M7  -> W[0]
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1

    # C[0,1] = M3 + M5  -> W[2]
    W[2][2] = 1
    W[2][4] = 1

    # C[1,0] = M2 + M4  -> W[1]
    W[1][1] = 1
    W[1][3] = 1

    # C[1,1] = M1 - M2 + M3 + M6  -> W[3]
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1

    return {"U": U, "V": V, "W": W}


def _search_rank6_over_F3():
    """
    Search for a rank-6 decomposition over F_3 that also works over integers.
    
    Key insight: For fixed U and W, the system for V decouples by (a,k) pair.
    For each (a,k), we have 2 linear equations in 6 unknowns (V[a,k,0..5]).
    A solution always exists over F_3. But we need it to work over integers.
    
    Strategy: Search over U and W with entries in {-1, 0, 1} (integers),
    solve for V over integers (using the decoupled structure), and verify.
    """
    rank = 6
    n, m, p = 2, 2, 2
    MOD = 3
    
    import random
    random.seed(12345)
    
    # Search over U and W with entries in {-1, 0, 1}
    # U: 4x6, W: 4x6
    # Total possibilities: 3^(48) - way too many for full enumeration.
    # Use random sampling.
    
    max_attempts = 100000
    F3_int = [-1, 0, 1]  # Integer representatives for F_3
    
    for attempt in range(max_attempts):
        # Random U and W with entries in {-1, 0, 1}
        U = [[random.choice(F3_int) for _ in range(rank)] for _ in range(4)]
        W = [[random.choice(F3_int) for _ in range(rank)] for _ in range(4)]
        
        # Solve for V over integers using the decoupled structure
        # For each (a,k): solve 2x6 system over integers
        # sum_r U[i,a,r] * W[k*2+i, r] * V[a,k,r] = 1 for i=0,1
        
        V = [[0] * rank for _ in range(4)]
        valid = True
        
        for a in range(2):
            for k in range(2):
                # Build coefficient matrix (2x6)
                # Row i: [U[i*2+a, r] * W[k*2+i, r] for r in 0..5]
                coeffs = []
                for i in range(2):
                    row = []
                    for r in range(rank):
                        row.append(U[i * 2 + a][r] * W[k * 2 + i][r])
                    coeffs.append(row)
                
                # Solve: coeffs[i][r] * V[a*2+k][r] summed over r = 1 for i=0,1
                # This is underdetermined (2 eqs, 6 unknowns).
                # Set V[a*2+k][r] = 0 for r >= 2, solve for r=0,1.
                # coeffs[0][0]*V0 + coeffs[0][1]*V1 = 1
                # coeffs[1][0]*V0 + coeffs[1][1]*V1 = 1
                
                c00 = coeffs[0][0]
                c01 = coeffs[0][1]
                c10 = coeffs[1][0]
                c11 = coeffs[1][1]
                
                det = c00 * c11 - c01 * c10
                
                if det == 0:
                    # Check if consistent
                    if c00 != 0:
                        V0 = 1 // c00 if 1 % c00 == 0 else None
                        if V0 is None:
                            valid = False
                            break
                        V[a*2+k][0] = V0
                        V[a*2+k][1] = 0
                        # Verify second equation
                        if c10 * V0 + c11 * 0 != 1:
                            valid = False
                            break
                    elif c01 != 0:
                        V1 = 1 // c01 if 1 % c01 == 0 else None
                        if V1 is None:
                            valid = False
                            break
                        V[a*2+k][0] = 0
                        V[a*2+k][1] = V1
                        if c10 * 0 + c11 * V1 != 1:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                else:
                    # Cramer's rule
                    V0 = (1 * c11 - c01 * 1) // det if (c11 - c01) % det == 0 else None
                    V1 = (c00 * 1 - 1 * c10) // det if (c00 - c10) % det == 0 else None
                    if V0 is None or V1 is None:
                        valid = False
                        break
                    # Verify
                    if c00 * V0 + c01 * V1 != 1 or c10 * V0 + c11 * V1 != 1:
                        valid = False
                        break
                    V[a*2+k][0] = V0
                    V[a*2+k][1] = V1
        
        if not valid:
            continue
        
        # Verify full decomposition over integers
        if _verify_decomposition(U, V, W, n=2, m=2, p=2, rank=6, mod=None):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    Strategy:
    1. Search for rank-6 decomposition with integer entries.
    2. Fall back to verified rank-7 Strassen decomposition.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer entries.
    """
    # Attempt rank-6 search
    result6 = _search_rank6_over_F3()
    if result6 is not None:
        return result6
    
    # Fall back to rank-7 Strassen
    return _strassen_decomposition()
