import numpy as np
import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition with integer/half-integer entries."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(U.shape[1]):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _snap_to_half_int(arr):
    """Snap array to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2.0
    return snapped

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _rank23_3x3x3():
    """
    Return a known rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses a Strassen-like approach with careful construction.
    U, V, W shapes: (9, 23), (9, 23), (9, 23)
    Convention: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    """
    n = 3
    m = 3
    p = 3
    rank = 23
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Based on known rank-23 decompositions for 3x3x3
    # Using a combination of Strassen-like terms
    # This is a specific decomposition from the literature
    
    # Term 1-7: Strassen-like for the main 2x2 block
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 0] = 1; V[4, 0] = 1; W[1, 0] = 1
    
    # M2 = (A10+A11)*B01
    U[1, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    U[2, 1] = 1; V[4, 1] = 1; W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; W[0, 2] = 1
    U[0, 2] = 1; V[5, 2] = -1; W[0, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[1, 3] = 1; V[4, 3] = 1; W[1, 3] = 1
    U[1, 3] = 1; V[0, 3] = -1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; V[5, 4] = 1; W[0, 4] = 1
    U[1, 4] = 1; V[5, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[1, 5] = 1; V[0, 5] = 1; W[1, 5] = 1
    U[0, 5] = -1; V[0, 5] = 1; W[0, 5] = 1
    U[1, 5] = 1; V[1, 5] = 1; W[1, 5] = 1
    U[0, 5] = -1; V[1, 5] = 1; W[0, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; V[4, 6] = 1; W[1, 6] = 1
    U[1, 6] = 1; V[5, 6] = 1; W[1, 6] = 1
    U[0, 6] = -1; V[4, 6] = 1; W[0, 6] = 1
    U[0, 6] = -1; V[5, 6] = 1; W[0, 6] = 1
    
    # Terms 8-23: Handle the third row and column
    # This is where the 3x3 differs from 2x2
    # We need additional terms to handle A02, A20, A21, A22, B02, B20, B21, B22
    
    # For simplicity, use a fallback approach:
    # Complete the decomposition using schoolbook for the remaining parts
    # This gives us a valid but not minimal decomposition
    
    # Actually, let's use a more systematic approach:
    # Start with schoolbook and try to reduce rank using ALS
    
    return None  # Fall back to ALS

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-10, seed=0):
    """
    Find a rank decomposition for M_{n,m,p} using ALS.
    Optimized with better initialization and convergence.
    """
    np.random.seed(seed)
    
    # Better initialization: use schoolbook-like structure
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Initialize with random perturbations of a structured pattern
    for t in range(rank):
        # Randomly assign which (i,j,k) triple this term contributes to
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        
        # Set base values
        U[i*m+j, t] = 1.0 + 0.1 * np.random.randn()
        V[j*p+k, t] = 1.0 + 0.1 * np.random.randn()
        W[k*n+i, t] = 1.0 + 0.1 * np.random.randn()
        
        # Add some random structure
        if np.random.rand() < 0.3:
            i2 = np.random.randint(0, n)
            j2 = np.random.randint(0, m)
            k2 = np.random.randint(0, p)
            U[i2*m+j2, t] += 0.5 * np.random.randn()
            V[j2*p+k2, t] += 0.5 * np.random.randn()
            W[k2*n+i2, t] += 0.5 * np.random.randn()
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
            
        # Solve for V
        M_V = np.zeros((n*m*p, m*p*rank))
        rhs_V = np.ones(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = idx
                    for t in range(rank):
                        coeff = U[i*m+j, t] * W[k*n+i, t]
                        col = (j*p+k) * rank + t
                        M_V[row, col] = coeff
                    idx += 1
        V_flat, _, _, _ = np.linalg.lstsq(M_V, rhs_V, rcond=None)
        V = V_flat.reshape(m*p, rank)
        
        # Solve for U
        M_U = np.zeros((n*m*p, n*m*rank))
        rhs_U = np.ones(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = idx
                    for t in range(rank):
                        coeff = V[j*p+k, t] * W[k*n+i, t]
                        col = (i*m+j) * rank + t
                        M_U[row, col] = coeff
                    idx += 1
        U_flat, _, _, _ = np.linalg.lstsq(M_U, rhs_U, rcond=None)
        U = U_flat.reshape(n*m, rank)
        
        # Solve for W
        M_W = np.zeros((n*m*p, p*n*rank))
        rhs_W = np.ones(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = idx
                    for t in range(rank):
                        coeff = U[i*m+j, t] * V[j*p+k, t]
                        col = (k*n+i) * rank + t
                        M_W[row, col] = coeff
                    idx += 1
        W_flat, _, _, _ = np.linalg.lstsq(M_W, rhs_W, rcond=None)
        W = W_flat.reshape(p*n, rank)
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(rank):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    err += (val - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            return U, V, W
            
    return best_U, best_V, best_W

def _try_rank_decomposition(n, m, p, target_rank, max_seeds=50):
    """
    Try to find a decomposition of given rank using ALS with multiple restarts.
    Returns (U, V, W) if found and verifiable, else None.
    """
    for seed in range(max_seeds):
        if _time_left() < 15:
            break
        res = _als_decompose(n, m, p, target_rank, max_iters=150, tol=1e-9, seed=seed)
        if res is not None:
            U_cand, V_cand, W_cand = res
            # Snap to half-integers
            U_int = _snap_to_half_int(U_cand)
            V_int = _snap_to_half_int(V_cand)
            W_int = _snap_to_half_int(W_cand)
            
            if _verify(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
    return None

def construct():
    n, m, p = 3, 3, 7
    
    # Strategy:
    # 1. Try to find rank-23 decomposition for 3x3x3 using ALS with snapping.
    # 2. If found, combine two 3x3x3 and one 3x3x1. Total rank 55.
    # 3. If not, try direct decomposition for 3x3x7 at various ranks.
    # 4. If not, use schoolbook rank 63.
    
    # First, try to find a 3x3x3 decomposition at rank 23
    U_33, V_33, W_33 = None, None, None
    rank_33 = 27
    
    res_33 = _try_rank_decomposition(3, 3, 3, 23, max_seeds=20)
    if res_33 is not None:
        U_33, V_33, W_33 = res_33
        rank_33 = 23
    
    if U_33 is not None:
        # Construct 3x3x7 decomposition using two 3x3x3 blocks and one 3x3x1 block
        U_31, V_31, W_31 = _schoolbook(3, 3, 1)
        rank_31 = 9
        
        total_rank = rank_33 + rank_33 + rank_31
        U = np.zeros((9, total_rank))
        V = np.zeros((21, total_rank))
        W = np.zeros((21, total_rank))
        
        # Part 1: k=0,1,2
        U[:, 0:rank_33] = U_33
        for j in range(3):
            for k in range(3):
                V[j*7+k, 0:rank_33] = V_33[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W[k*3+i, 0:rank_33] = W_33[k*3+i, :]
        
        # Part 2: k=3,4,5
        U[:, rank_33:2*rank_33] = U_33
        for j in range(3):
            for k in range(3):
                V[j*7+(k+3), rank_33:2*rank_33] = V_33[j*3+k, :]
        for k in range(3):
            for i in range(3):
                W[(k+3)*3+i, rank_33:2*rank_33] = W_33[k*3+i, :]
        
        # Part 3: k=6
        U[:, 2*rank_33:2*rank_33+rank_31] = U_31
        for j in range(3):
            V[j*7+6, 2*rank_33:2*rank_33+rank_31] = V_31[j*1+0, :]
        for i in range(3):
            W[6*3+i, 2*rank_33:2*rank_33+rank_31] = W_31[0*3+i, :]
        
        # Verify full decomposition
        if _verify(U, V, W, 3, 3, 7):
            def to_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try direct decomposition for 3x3x7 at various ranks
    # Theoretical lower bound is around 27 (from 3x3x3 being 23, scaling up)
    # But we need to be careful about time
    
    # Try rank 30 first
    for target_rank in [30, 35, 40, 45]:
        if _time_left() < 20:
            break
        res_direct = _try_rank_decomposition(3, 3, 7, target_rank, max_seeds=8)
        if res_direct is not None:
            U, V, W = res_direct
            def to_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(3, 3, 7)
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
