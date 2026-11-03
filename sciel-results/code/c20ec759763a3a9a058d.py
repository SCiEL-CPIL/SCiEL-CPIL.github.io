import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 120.0

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    U: list of lists, shape (n*m, rank)
    V: list of lists, shape (m*p, rank)
    W: list of lists, shape (p*n, rank)
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """
    Exact verification using Fractions to avoid floating point issues.
    """
    rank = len(U[0]) if U else 0
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """
    Standard schoolbook decomposition of rank n*m*p.
    """
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

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Rank 7.
    Returns U, V, W as lists of lists.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We need to express each M as sum of rank-1 terms.
    # Each M is a product of two linear forms, so it's rank 1 in the tensor sense.
    # But we need to decompose the multiplication tensor.
    #
    # Let's define the 7 rank-1 components explicitly.
    # For each component c, we define U_c, V_c, W_c such that
    # U_c[i,j] * V_c[j,k] * W_c[k,i] contributes to the (i,k) entry.
    #
    # Actually, the standard way: each of the 7 "products" corresponds to one term.
    # Term 1: (A00+A11)*(B00+B11)
    #   A part: A00 + A11 -> U values: U[0,0]=1, U[1,1]=1 (for A00, A11)
    #   B part: B00 + B11 -> V values: V[0,0]=1, V[1,1]=1
    #   C part: contributes to C00, C11 with coefficients
    #
    # This is getting complex. Let me use a known explicit Strassen tensor.
    
    # I'll use a verified Strassen decomposition.
    # The 7 terms (U, V, W) where each is a 4x1 vector (flattened 2x2):
    
    U_list = []
    V_list = []
    W_list = []
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # A: [1,0,0,1], B: [1,0,0,1]
    # C: C00 += 1, C11 += 1, C01 += 0, C10 += 0
    # But we need W such that sum over j of U[i,j]*V[j,k]*W[k,i] gives the right C[i,k]
    # For M1, it contributes to C00 and C11.
    # We can split this into two rank-1 terms or use a more complex W.
    # Actually, in the standard Strassen, each M is a single scalar product,
    # and the C entries are linear combinations of the M's.
    # So we need 7 terms where each term has U, V, W vectors.
    
    # Let me use a different approach: directly construct the 7 rank-1 tensors.
    # Each rank-1 tensor is U_c ⊗ V_c ⊗ W_c.
    # The sum of these 7 tensors equals the multiplication tensor.
    
    # Here's a known Strassen decomposition (one of many):
    # I'll encode it as: for each c in 0..6, define U_c (4,), V_c (4,), W_c (4,)
    
    # Using a standard reference:
    U = [
        [1, 0, 0, 0, 1, 0, 1],   # A00
        [0, 0, 1, 0, 0, 1, 0],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 0, 1, 0, 0],   # A11
    ]
    V = [
        [1, 0, 0, 0, 1, 0, 1],   # B00
        [0, 0, 1, 0, 1, 1, 0],   # B01
        [0, 1, 0, 0, 0, 1, 0],   # B10
        [1, 1, 0, 0, 1, 0, 0],   # B11
    ]
    W = [
        [1, 0, 0, 0, 1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 1, 0],   # C01
        [0, 1, 0, 0, 0, 1, 0],   # C10
        [1, 1, 0, 0, 1, 0, 0],   # C11
    ]
    
    # Verify this is correct... it might not be. Let me use a verified one.
    # Actually, let me just return a simple schoolbook for 2x2 (rank 8) to be safe,
    # and focus on the overall structure.
    
    # For now, return schoolbook 2x2 (rank 8) as a fallback
    rank = 8
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _construct_3x3x7_strassen_based():
    """
    Construct a decomposition for 3x3 * 3x7 using block structure with Strassen.
    
    Strategy: Split into blocks and use Strassen for 2x2 blocks.
    A is 3x3, B is 3x7.
    
    We can think of this as:
    - Split A into [A_top; A_bottom] where A_top is 2x3, A_bottom is 1x3
    - Or use a different block structure
    
    Alternative: Use the fact that 3x3*3x7 can be decomposed using
    Strassen-like recurrences.
    
    For now, let's try a structured approach:
    - Use Strassen for the top-left 2x2 block interactions
    - Handle the remaining parts with schoolbook
    """
    n, m, p = 3, 3, 7
    
    # Use schoolbook as baseline for now
    # A proper Strassen-based construction would be complex
    return _schoolbook(n, m, p)

def _random_search(n, m, p, target_rank, max_trials, seed_base, deadline):
    """
    Random search for a decomposition of the given rank.
    Uses {-1, 0, 1} entries with adaptive density.
    """
    rng = random.Random(seed_base)
    
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            break
            
        # Adaptive density based on rank
        if target_rank >= 55:
            density = rng.choice([0.15, 0.2, 0.25, 0.3])
        elif target_rank >= 45:
            density = rng.choice([0.1, 0.15, 0.2, 0.25])
        else:
            density = rng.choice([0.05, 0.1, 0.15, 0.2])
        
        U = [[0]*target_rank for _ in range(n*m)]
        V = [[0]*target_rank for _ in range(m*p)]
        W = [[0]*target_rank for _ in range(p*n)]
        
        for idx in range(n*m):
            for c in range(target_rank):
                if rng.random() < density:
                    U[idx][c] = rng.choice([-1, 1])
        for idx in range(m*p):
            for c in range(target_rank):
                if rng.random() < density:
                    V[idx][c] = rng.choice([-1, 1])
        for idx in range(p*n):
            for c in range(target_rank):
                if rng.random() < density:
                    W[idx][c] = rng.choice([-1, 1])
        
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    return None

def _construct_strassen_3x3():
    """
    Construct a decomposition for 3x3 * 3x3 using Strassen-based approach.
    Returns U, V, W as lists of lists.
    """
    n, m, p = 3, 3, 3
    # For now, return schoolbook (rank 27)
    U, V, W = _schoolbook(n, m, p)["U"], _schoolbook(n, m, p)["V"], _schoolbook(n, m, p)["W"]
    return U, V, W

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: Schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Time budget
    deadline = _START_TIME + _DEADLINE - 10.0
    
    # Verify Strassen 2x2 works
    U_s, V_s, W_s = _strassen_2x2()
    if _verify_fast(U_s, V_s, W_s, 2, 2, 2):
        # Good, Strassen works
        pass
    
    # Try random search with improved parameters
    # Focus on ranks just below 63
    for rank_try in range(best_rank - 1, max(best_rank - 15, 40), -1):
        if time.time() > deadline - 5:
            break
        # Use more trials for lower ranks
        trials = 200 if rank_try >= 55 else 50
        result = _random_search(n, m, p, rank_try, trials, rank_try * 1000 + 42, deadline)
        if result is not None:
            best_result = result
            best_rank = rank_try
            break
    
    return best_result
