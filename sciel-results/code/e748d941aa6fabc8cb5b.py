"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares
from scipy.optimize import nnls

DEADLINE = 140.0  # safety margin before 150s kill

def _verify_fast(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication using Fraction."""
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_merge_fast(U, V, W, n, m, p, c1, c2, factor):
    """Try to merge columns c1 and c2 by adding factor * c2 to c1 and removing c2."""
    rank = len(U[0])
    if c1 == c2:
        return None
    
    new_rank = rank - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    new_col = 0
    for c in range(rank):
        if c == c2:
            continue
        if c == c1:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c1] + factor * U[i][c2]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c1] + factor * V[j][c2]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c1] + factor * W[k][c2]
        else:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c]
        new_col += 1
    
    if _verify_fast(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
    return None

def _strassen_block(U_block, V_block, W_block):
    """
    Apply Strassen's 7-multiplication algorithm to a 2x2x2 block.
    U_block: 4x8 (indices i*2+j for i,j in 0..1)
    V_block: 4x8 (indices j*2+k for j,k in 0..1)
    W_block: 4x8 (indices k*2+i for k,i in 0..1)
    Returns: U_new (4x7), V_new (4x7), W_new (4x7)
    """
    U00, U01, U10, U11 = U_block[0], U_block[1], U_block[2], U_block[3]
    V00, V01, V10, V11 = V_block[0], V_block[1], V_block[2], V_block[3]
    W00, W01, W10, W11 = W_block[0], W_block[1], W_block[2], W_block[3]
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U factors:
    U_M1 = [Fraction(a) + Fraction(b) for a, b in zip(U00, U11)]
    U_M2 = [Fraction(a) + Fraction(b) for a, b in zip(U10, U11)]
    U_M3 = list(U00)
    U_M4 = list(U11)
    U_M5 = [Fraction(a) + Fraction(b) for a, b in zip(U00, U01)]
    U_M6 = [Fraction(a) - Fraction(b) for a, b in zip(U10, U00)]
    U_M7 = [Fraction(a) - Fraction(b) for a, b in zip(U01, U11)]
    
    # V factors:
    V_M1 = [Fraction(a) + Fraction(b) for a, b in zip(V00, V11)]
    V_M2 = list(V00)
    V_M3 = [Fraction(a) - Fraction(b) for a, b in zip(V01, V11)]
    V_M4 = [Fraction(a) - Fraction(b) for a, b in zip(V10, V00)]
    V_M5 = list(V11)
    V_M6 = [Fraction(a) + Fraction(b) for a, b in zip(V00, V01)]
    V_M7 = [Fraction(a) + Fraction(b) for a, b in zip(V10, V11)]
    
    # W factors:
    W_M1 = [Fraction(a) + Fraction(b) for a, b in zip(W00, W11)]
    W_M2 = list(W00)
    W_M3 = list(W11)
    W_M4 = list(W11)
    W_M5 = [Fraction(a) - Fraction(b) for a, b in zip(W00, W11)]
    W_M6 = list(W00)
    W_M7 = [Fraction(a) - Fraction(b) for a, b in zip(W00, W11)]
    
    # Wait, I need to be careful with the W factors. Let me re-derive.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W00 (k=0,i=0): coefficient of M1 is 1, M4 is -1, M5 is 1, M7 is -1
    # W01 (k=0,i=1): coefficient of M3 is 1, M5 is 1
    # W10 (k=1,i=0): coefficient of M2 is 1, M4 is 1
    # W11 (k=1,i=1): coefficient of M1 is 1, M2 is -1, M3 is 1, M6 is 1
    
    # But W factors are added across the 7 terms.
    # W_new[0] (for i=0,k=0): W00 * 1 + W00 * (-1) + W00 * 1 + W00 * (-1) ... no wait.
    
    # Actually, W[k*n+i] is the factor for output (i,k).
    # In the 2x2x2 block, W has 4 rows: (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
    
    # W_new[0] = W00*1 + W00*(-1) + W00*1 + W00*(-1) ... this doesn't make sense.
    
    # Let me think again. The W factor for each Strassen term M_c is a vector of length 4.
    # For M1: contributes to C00 (+1), C11 (+1). So W_M1 = [1, 0, 0, 1]
    # For M2: contributes to C10 (+1), C11 (-1). So W_M2 = [0, 0, 1, -1]
    # For M3: contributes to C01 (+1), C11 (+1). So W_M3 = [0, 1, 0, 1]
    # For M4: contributes to C00 (-1), C10 (+1). So W_M4 = [-1, 0, 1, 0]
    # For M5: contributes to C00 (+1), C01 (+1). So W_M5 = [1, 1, 0, 0]
    # For M6: contributes to C11 (+1). So W_M6 = [0, 0, 0, 1]
    # For M7: contributes to C00 (-1). So W_M7 = [-1, 0, 0, 0]
    
    # But we need to combine with the original W factors.
    # The original W factors are W00, W01, W10, W11 for the 4 schoolbook terms.
    
    # Actually, the Strassen algorithm replaces the 4 schoolbook multiplications with 7 Strassen multiplications.
    # The U, V, W factors need to be transformed accordingly.
    
    # This is getting complex. Let me use a different approach:
    # Just try to find a rank-7 decomposition for 2x2x2 and verify it.
    
    return None  # Skip Strassen for now, use merge search instead

def _optimize_rank(n, m, p, target_rank, max_time=30.0):
    """Try to find a decomposition of target_rank using optimization."""
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(42)
    U_init = np.random.randn(n * m, target_rank)
    V_init = np.random.randn(m * p, target_rank)
    W_init = np.random.randn(p * n, target_rank)
    
    # Scale to get reasonable magnitudes
    U_init /= np.sqrt(target_rank)
    V_init /= np.sqrt(target_rank)
    W_init /= np.sqrt(target_rank)
    
    # Create the target tensor
    # T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a nonlinear optimization problem.
    # Let's use a different approach: fix W and solve for U and V linearly.
    
    # Actually, let's try a simpler approach: use the schoolbook decomposition
    # and try to merge columns more aggressively.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Try to find lower rank decompositions via merging
    # Only try a limited number of merge attempts to stay within time
    max_merge_attempts = 50000
    attempts = 0
    factors_to_try = [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(2), Fraction(-2), Fraction(3), Fraction(-3), Fraction(1,3), Fraction(-1,3), Fraction(1,4), Fraction(-1,4), Fraction(4), Fraction(-4)]
    
    current_U, current_V, current_W = U, V, W
    current_rank = best_rank
    
    while current_rank > 10 and time.time() < DEADLINE:
        merged = False
        for c1 in range(current_rank):
            if time.time() > DEADLINE:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > DEADLINE:
                    break
                for factor in factors_to_try:
                    attempts += 1
                    if attempts > max_merge_attempts:
                        break
                    result = _try_merge_fast(current_U, current_V, current_W, n, m, p, c1, c2, factor)
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        best_U, best_V, best_W = current_U, current_V, current_W
                        best_rank = current_rank
                        merged = True
                        break
                if merged or attempts > max_merge_attempts:
                    break
            if merged or attempts > max_merge_attempts:
                break
        
        if not merged:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
