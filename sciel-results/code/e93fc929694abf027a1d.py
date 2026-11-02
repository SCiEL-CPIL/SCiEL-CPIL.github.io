"""Combined program: rank search for 3x3x6 matmul via block decomposition and direct search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are within tol of half-integers. Return rounded half-integers if valid."""
    doubled = arr * 2.0
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _search_3x3x2(rank, max_attempts=5000, deadline=None):
    """Search for a rank-R decomposition of 3x3x2 matmul tensor.
    U: 9 x R, V: 6 x R, W: 6 x R
    Returns (U, V, W) as lists, or None.
    """
    n, m, p = 3, 3, 2
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        # Try different density patterns
        density = 0.5 if attempt % 3 == 0 else (0.7 if attempt % 3 == 1 else 0.3)
        U_rand = (np.random.rand(n * m, rank) < density) * np.random.randint(-2, 3, size=(n * m, rank))
        V_rand = (np.random.rand(m * p, rank) < density) * np.random.randint(-2, 3, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build 3 x rank matrix M where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                # Solve M @ w = ones(m)
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                # Check half-integer
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W_sol[k * n + i, :] = w_check
        
        if valid:
            return U_rand.tolist(), V_rand.tolist(), W_sol.tolist()
    
    return None

def _search_3x3x3(rank, max_attempts=3000, deadline=None):
    """Search for a rank-R decomposition of 3x3x3 matmul tensor.
    U: 9 x R, V: 9 x R, W: 9 x R
    Returns (U, V, W) as lists, or None.
    """
    n, m, p = 3, 3, 3
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        density = 0.5 if attempt % 3 == 0 else (0.7 if attempt % 3 == 1 else 0.3)
        U_rand = (np.random.rand(n * m, rank) < density) * np.random.randint(-2, 3, size=(n * m, rank))
        V_rand = (np.random.rand(m * p, rank) < density) * np.random.randint(-2, 3, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W_sol[k * n + i, :] = w_check
        
        if valid:
            return U_rand.tolist(), V_rand.tolist(), W_sol.tolist()
    
    return None

def _search_3x3x6_direct(rank, max_attempts=1000, deadline=None):
    """Search for a rank-R decomposition of 3x3x6 matmul tensor directly.
    U: 9 x R, V: 18 x R, W: 18 x R
    Returns (U, V, W) as lists, or None.
    """
    n, m, p = 3, 3, 6
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        density = 0.4 if attempt % 3 == 0 else (0.6 if attempt % 3 == 1 else 0.3)
        U_rand = (np.random.rand(n * m, rank) < density) * np.random.randint(-2, 3, size=(n * m, rank))
        V_rand = (np.random.rand(m * p, rank) < density) * np.random.randint(-2, 3, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W_sol[k * n + i, :] = w_check
        
        if valid:
            # Full verification needed for direct search
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_sol.tolist()
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the n x m x p matmul tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Uses vectorized computation for speed.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # T = U^T @ V * W^T ... actually use einsum
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Build expected tensor
    expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i * m + j, j * p + k, k * n + i] = 1.0
    
    if not np.allclose(T, expected, atol=1e-9):
        return False
    return True

def _build_block_decomposition(n, m, p, p_split, core_rank, cores):
    """Build a block decomposition from core decompositions.
    Splits p into p_split blocks of size p//p_split each.
    """
    p_per_block = p // p_split
    R = p_split * core_rank
    U_total = [[0] * R for _ in range(n * m)]
    V_total = [[0] * R for _ in range(m * p)]
    W_total = [[0] * R for _ in range(p * n)]
    
    for t in range(p_split):
        U_t, V_t, W_t = cores[t]
        offset = core_rank * t
        for i in range(n):
            for j in range(m):
                for r in range(core_rank):
                    U_total[i * m + j][offset + r] = U_t[i * m + j][r]
        for j in range(m):
            for k_local in range(p_per_block):
                for r in range(core_rank):
                    k_global = t * p_per_block + k_local
                    V_total[j * p + k_global][offset + r] = V_t[j * p_per_block + k_local][r]
        for k_local in range(p_per_block):
            for i in range(n):
                for r in range(core_rank):
                    k_global = t * p_per_block + k_local
                    W_total[k_global * n + i][offset + r] = W_t[k_local * n + i][r]
    
    return U_total, V_total, W_total

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    best_result = None
    best_rank = n * m * p  # 54
    
    # Strategy 1: Try 3 parallel 3x3x2 blocks with various core ranks
    for core_rank in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
        total_rank = 3 * core_rank
        if total_rank >= best_rank:
            break
        blocks = []
        for t in range(3):
            if time.time() > deadline:
                blocks = None
                break
            result = _search_3x3x2(core_rank, max_attempts=2000, deadline=deadline)
            if result is None:
                blocks = None
                break
            blocks.append(result)
        
        if blocks is not None and len(blocks) == 3:
            U_total, V_total, W_total = _build_block_decomposition(n, m, p, 3, core_rank, blocks)
            if _verify_decomposition(U_total, V_total, W_total, n, m, p):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {"U": U_total, "V": V_total, "W": W_total}
                break  # Found the lowest core_rank that works
    
    # Strategy 2: Try 2 parallel 3x3x3 blocks
    for core_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        total_rank = 2 * core_rank
        if total_rank >= best_rank:
            break
        cores = []
        for t in range(2):
            if time.time() > deadline:
                cores = None
                break
            result = _search_3x3x3(core_rank, max_attempts=1500, deadline=deadline)
            if result is None:
                cores = None
                break
            cores.append(result)
        
        if cores is not None and len(cores) == 2:
            U_total, V_total, W_total = _build_block_decomposition(n, m, p, 2, core_rank, cores)
            if _verify_decomposition(U_total, V_total, W_total, n, m, p):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {"U": U_total, "V": V_total, "W": W_total}
                break
    
    # Strategy 3: Direct search for 3x3x6 at various ranks
    for rank in [30, 36, 40, 42, 44, 46, 48, 50, 52]:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            break
        result = _search_3x3x6_direct(rank, max_attempts=500, deadline=deadline)
        if result is not None:
            U_list, V_list, W_list = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    # Return best result or schoolbook fallback
    if best_result is not None:
        return best_result
    
    # Fallback: Schoolbook rank 54
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
