"""Matrix multiplication tensor decomposition for 2x3x8."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_START = time.time()
_DEADLINE = _START + 140.0

def _check_time():
    return time.time() > _DEADLINE - 15

def _validate_exact(U, V, W):
    """Check if factor matrices exactly decompose the matmul tensor using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = Fraction(0)
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    return False
    return True

def _validate_numeric(U, V, W, tol=1e-9):
    """Quick numeric check."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U indexed by (i,j) for A, V indexed by (j,k) for B, W indexed by (k,i) for C.
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: A[i,j] coefficients for each product (rows: A00,A01,A10,A11; cols: M1..M7)
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, -1],  # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    # V: B[j,k] coefficients (rows: B00,B01,B10,B11; cols: M1..M7)
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ]
    # W: C[k,i] coefficients (rows: C00,C01,C10,C11; cols: M1..M7)
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]
    return U, V, W

def _build_2x3x8_structural():
    """Build 2x3x8 decomposition by splitting p=8 into 4+4 and using 2x3x4 blocks.
    For each 2x3x4, split m=3 into 1+2:
    2x1x4: schoolbook rank 8
    2x2x4: Strassen-based rank 14 (7 per 2x2x2, two blocks)
    Total per 2x3x4: 8+14=22, total for 2x3x8: 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 6 x 44
    V = [[0]*rank for _ in range(m*p)]  # 24 x 44
    W = [[0]*rank for _ in range(p*n)]  # 16 x 44
    
    col = 0
    for half in range(2):  # two halves of p
        p_offset = half * 4
        # 2x1x4 part: A[i,0] * B[0, p_offset+k]
        for i in range(2):
            for k in range(4):
                U[i*m + 0][col] = 1
                V[0*p + p_offset + k][col] = 1
                W[(p_offset + k)*n + i][col] = 1
                col += 1
        # 2x2x4 part: Strassen on 2x2x2, doubled
        U_s, V_s, W_s = _strassen_2x2x2()
        for sub in range(2):  # two sub-blocks of p=2
            p_sub_offset = p_offset + sub * 2
            for c in range(7):
                for i in range(2):
                    for j in range(1, 3):  # j=1,2 (the 2x2 part)
                        U[i*m + j][col + c] = U_s[i*2 + (j-1)][c]
                for j in range(1, 3):
                    for k in range(2):
                        V[j*p + p_sub_offset + k][col + c] = V_s[(j-1)*2 + k][c]
                for k in range(2):
                    for i in range(2):
                        W[(p_sub_offset + k)*n + i][col + c] = W_s[k*2 + i][c]
            col += 7
    
    return U, V, W

def _als_step(U, V, W, n, m, p, target_r):
    """One ALS iteration, vectorized where possible."""
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            A_sys = np.zeros((p, target_r))
            for k in range(p):
                A_sys[k] = np.array(V[j*p+k]) * np.array(W[k*n+i])
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)
                U_new[u_idx] = [float(x) for x in sol]
            except:
                pass
    
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            A_sys = np.zeros((n, target_r))
            for i in range(n):
                A_sys[i] = np.array(U_new[i*m+j]) * np.array(W[k*n+i])
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                V_new[v_idx] = [float(x) for x in sol]
            except:
                pass
    
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            A_sys = np.zeros((m, target_r))
            for j in range(m):
                A_sys[j] = np.array(U_new[i*m+j]) * np.array(V_new[j*p+k])
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                W_new[w_idx] = [float(x) for x in sol]
            except:
                pass
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, n, m, p):
    """Compute max error of the decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _recover_rationals(U, V, W, max_den=100):
    """Try to recover exact rationals from numerical solution."""
    def to_frac(x):
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        for den in range(2, max_den + 1):
            num = x * den
            if abs(num - round(num)) < 1e-10:
                return Fraction(int(round(num)), den)
        return Fraction(x).limit_denominator(max_den)
    
    U_fr = [[to_frac(float(x)) for x in row] for row in U]
    V_fr = [[to_frac(float(x)) for x in row] for row in V]
    W_fr = [[to_frac(float(x)) for x in row] for row in W]
    return U_fr, V_fr, W_fr

def _search_direct(n, m, p, target_r, max_trials=5, max_iters=200):
    """Search for rank target_r decomposition of n x m x p matmul tensor."""
    best_err = float('inf')
    best_sol = None
    
    for trial in range(max_trials):
        if _check_time():
            break
        
        rng = np.random.default_rng(42 + trial * 137 + target_r)
        
        U = rng.standard_normal((n*m, target_r)) * 0.5
        V = rng.standard_normal((m*p, target_r)) * 0.5
        W = rng.standard_normal((p*n, target_r)) * 0.5
        
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
        
        prev_err = float('inf')
        for it in range(max_iters):
            if _check_time():
                break
            
            U, V, W = _als_step(U, V, W, n, m, p, target_r)
            
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_sol = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
            
            if err < 1e-10:
                break
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
    
    if best_err < 1e-8:
        return best_sol, best_err
    return None, best_err

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Build schoolbook solution as fallback
    schoolbook_rank = n * m * p  # 48
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Strategy 1: Structural decomposition (split m=3 into 1+2, use Strassen)
    # This gives rank 44
    if not _check_time():
        U_struct, V_struct, W_struct = _build_2x3x8_structural()
        # Validate
        if _validate_numeric(U_struct, V_struct, W_struct, tol=1e-6):
            # Try to recover exact rationals
            U_fr, V_fr, W_fr = _recover_rationals(U_struct, V_struct, W_struct)
            if _validate_exact(U_fr, V_fr, W_fr):
                best_rank = 44
                best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
            else:
                # The structural solution should be exact with integers
                U_int = [[int(x) for x in row] for row in U_struct]
                V_int = [[int(x) for x in row] for row in V_struct]
                W_int = [[int(x) for x in row] for row in W_struct]
                if _validate_exact(U_int, V_int, W_int):
                    best_rank = 44
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
    
    # Strategy 2: Try to improve below 44 using ALS
    candidate_ranks = [43, 42, 41, 40, 39, 38, 36, 34, 32, 30]
    
    for target_r in candidate_ranks:
        if _check_time():
            break
        if target_r >= best_rank:
            continue
        
        sol, err = _search_direct(n, m, p, target_r, max_trials=3, max_iters=100)
        if sol is not None:
            U_sol, V_sol, W_sol = sol
            U_fr, V_fr, W_fr = _recover_rationals(U_sol, V_sol, W_sol)
            if _validate_exact(U_fr, V_fr, W_fr):
                if target_r < best_rank:
                    best_rank = target_r
                    best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
                break
    
    return best_result
