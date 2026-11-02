"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for c in range(rank):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _verify_exact(U_frac, V_frac, W_frac, n, m, p):
    """Verify with exact Fraction arithmetic."""
    rank = len(U_frac[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = Fraction(0, 1)
                for c in range(rank):
                    s += U_frac[u_idx][c] * V_frac[v_idx][c] * W_frac[w_idx][c]
                if s != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def _to_list(U, V, W):
    """Convert Fraction matrices to lists of floats (for return)."""
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return U_list, V_list, W_list

def _greedy_reduce(U, V, W, n, m, p, deadline):
    """Repeatedly try to remove columns."""
    rank = len(U[0])
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for col_idx in range(rank - 1, -1, -1):
            if time.time() > deadline:
                break
            U_test = [row[:col_idx] + row[col_idx+1:] for row in U]
            V_test = [row[:col_idx] + row[col_idx+1:] for row in V]
            W_test = [row[:col_idx] + row[col_idx+1:] for row in W]
            
            if _verify_exact(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                rank = len(U[0])
                changed = True
                break
    return U, V, W, rank

def _try_merge_pairs(U, V, W, n, m, p, deadline):
    """Try to merge pairs of columns that sum to zero on the support."""
    rank = len(U[0])
    merged = False
    for a in range(rank - 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, rank):
            if time.time() > deadline:
                break
            
            # Check if removing both a and b still gives correct result
            # after adjusting... actually we need to check if the sum of
            # contributions from a and b is zero everywhere.
            # 
            # For each (i,j,k), check if U[i,j,a]*V[j,k,a]*W[k,i,a] + 
            # U[i,j,b]*V[j,k,b]*W[k,i,b] = 0
            # 
            # But that's not quite right. We need to see if there exists
            # a linear combination that cancels.
            #
            # Simpler: try removing column a and see if the result is still valid
            # with column b absorbing the residual.
            #
            # Actually, let's just try the simple approach: for each pair,
            # check if the "sum tensor" of the two rank-1 terms is zero.
            
            zero_sum = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = (U[u_idx][a] * V[v_idx][a] * W[w_idx][a] +
                               U[u_idx][b] * V[v_idx][b] * W[w_idx][b])
                        if val != 0:
                            zero_sum = False
                            break
                    if not zero_sum:
                        break
                if not zero_sum:
                    break
            if zero_sum:
                # Remove both columns
                U_new = [row[:a] + row[a+1:b] + row[b+1:] for row in U]
                V_new = [row[:a] + row[a+1:b] + row[b+1:] for row in V]
                W_new = [row[:a] + row[a+1:b] + row[b+1:] for row in W]
                U, V, W = U_new, V_new, W_new
                rank = len(U[0])
                merged = True
                break
        if merged:
            break
    return U, V, W, rank, merged

def _random_half_integer_search(start_rank, end_rank, time_budget, n, m, p):
    """Search for a decomposition with rank in [start_rank, end_rank] using half-integers."""
    best = None
    
    for r in range(start_rank, end_rank + 1):
        if time.time() > time_budget:
            break
        for attempt in range(200):
            if time.time() > time_budget:
                break
            
            # Random half-integer initialization
            U = np.random.randint(-4, 5, size=(n*m, r)).astype(float) / 2.0
            V = np.random.randint(-4, 5, size=(m*p, r)).astype(float) / 2.0
            W = np.random.randint(-4, 5, size=(p*n, r)).astype(float) / 2.0
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if best is None or r < best[0]:
                    best = (r, U.copy(), V.copy(), W.copy())
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook in exact fractions
    U, V, W = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try greedy reduction
    U, V, W, best_rank = _greedy_reduce(U, V, W, n, m, p, deadline)
    
    # Try merge pairs
    if time.time() < deadline:
        U, V, W, best_rank, merged = _try_merge_pairs(U, V, W, n, m, p, deadline)
        if merged:
            U, V, W, best_rank = _greedy_reduce(U, V, W, n, m, p, deadline)
    
    # If we improved, great. Otherwise, try random search for lower ranks.
    if best_rank == schoolbook_rank and time.time() < deadline:
        # Try random search from rank 35 to 41
        result = _random_half_integer_search(35, 41, deadline, n, m, p)
        if result is not None:
            r, U_r, V_r, W_r = result
            if r < best_rank:
                best_rank = r
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Convert to list and return
    U_list, V_list, W_list = _to_list(U, V, W)
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
