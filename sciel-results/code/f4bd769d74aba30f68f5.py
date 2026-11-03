"""
Tensor decomposition for 4x4 @ 4x5 matrix multiplication.
Uses a combined Strassen (4x4x4) + schoolbook (4x4x1) approach with rank reduction.
"""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using recursive Strassen: 7*7 = 49 terms."""
    n = 4
    terms_2x2 = _strassen_2x2_terms()
    rank = 49
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            for (bi, bj), cu in term1['U'].items():
                for (li, lj), cv in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i * n + j][idx] += cu * cv
            
            for (bj, bk), cv in term1['V'].items():
                for (lj, lk), cw in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j * n + k][idx] += cv * cw
            
            for (bk, bi), cw in term1['W'].items():
                for (lk, li), cu2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k * n + i][idx] += cw * cu2
    
    return {"U": U, "V": V, "W": W}


def _build_4x4x5_combined():
    """Build 4x4x5 decomposition: Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 4) = 53."""
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    base = _build_4x4x4_strassen()
    rank_base = 49
    rank = rank_base + 4
    
    U = [[0] * rank for _ in range(n * m)]   # 16 x rank
    V = [[0] * rank for _ in range(m * p)]   # 20 x rank
    W = [[0] * rank for _ in range(p * n)]   # 20 x rank
    
    # Copy 4x4x4 part
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    
    for j in range(m):
        for k in range(4):
            for r in range(rank_base):
                V[j * p + k][r] = base["V"][j * 4 + k][r]
    
    for k in range(4):
        for i in range(n):
            for r in range(rank_base):
                W[k * n + i][r] = base["W"][k * 4 + i][r]
    
    # 4x4x1 part: C[i, 4] = sum_j A[i,j] * B[j, 4]
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i * m + j][r] = 1
        V[j * p + 4][r] = 1
        for i in range(n):
            W[4 * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, start_rank, deadline=None):
    """Try to reduce rank by finding linearly dependent terms."""
    rank = start_rank
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    if deadline is not None and time.time() > deadline - 5:
        return None, None, None, best_rank
    
    # Try removing terms that are linear combinations of others
    for _ in range(5):
        if deadline is not None and time.time() > deadline - 5:
            break
        reduced = False
        for r_remove in range(best_rank):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Check if term r_remove is redundant by seeing if removing it still gives correct result
            mask = [x for x in range(best_rank) if x != r_remove]
            if len(mask) == 0:
                continue
            
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            U_reshaped = U_test.reshape(n, m, len(mask))
            V_reshaped = V_test.reshape(m, p, len(mask))
            W_reshaped = W_test.reshape(p, n, len(mask))
            
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            if np.allclose(T, 1.0, atol=1e-6):
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_rank = len(mask)
                reduced = True
                break
        
        if not reduced:
            break
    
    if best_rank < start_rank:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        return U_out, V_out, W_out, best_rank
    
    return None, None, None, best_rank


def _als_refinement(U, V, W, n, m, p, target_rank, max_iter=100, deadline=None):
    """Try ALS-based refinement to find a lower-rank decomposition."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    orig_rank = len(U_arr[0])
    
    # Initialize with random projections of the original
    np.random.seed(42)
    U_new = np.random.randn(n * m, target_rank) * 0.1
    V_new = np.random.randn(m * p, target_rank) * 0.1
    W_new = np.random.randn(p * n, target_rank) * 0.1
    
    # Also try using subsets of original columns
    if orig_rank >= target_rank:
        indices = np.random.choice(orig_rank, target_rank, replace=False)
        U_init = U_arr[:, indices].copy()
        V_init = V_arr[:, indices].copy()
        W_init = W_arr[:, indices].copy()
    else:
        U_init = U_new
        V_init = V_new
        W_init = W_new
    
    best_err = float('inf')
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Update U
        # For each (i,j), U[i*m+j, :] should satisfy sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is complex; use gradient-free approach instead
        
        # Simple approach: try small perturbations
        for coord in range(min(20, n * m * target_rank)):
            if deadline is not None and time.time() > deadline - 10:
                break
            row = coord // target_rank
            col = coord % target_rank
            
            for delta in [-0.1, 0.1]:
                old_val = U_new[row, col]
                U_new[row, col] = old_val + delta
                
                U_reshaped = U_new.reshape(n, m, target_rank)
                V_reshaped = V_new.reshape(m, p, target_rank)
                W_reshaped = W_new.reshape(p, n, target_rank)
                T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                err = np.max(np.abs(T - 1.0))
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                else:
                    U_new[row, col] = old_val
        
        if best_err < 1e-6:
            break
    
    if best_err < 1e-6:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        return U_out, V_out, W_out, target_rank
    
    return None, None, None, target_rank


def _build_schoolbook():
    """Fallback schoolbook decomposition: rank = n*m*p = 80."""
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


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Combined Strassen + schoolbook (rank 53)
    try:
        result = _build_4x4x5_combined()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
                
                # Try to reduce rank
                if time.time() < deadline - 20:
                    U_red, V_red, W_red, red_rank = _try_reduce_rank(
                        result["U"], result["V"], result["W"], n, m, p, rank, deadline=deadline
                    )
                    if U_red is not None and red_rank < best_rank:
                        if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                            if _verify_integer_half_integer(U_red, V_red, W_red):
                                best_result = {"U": U_red, "V": V_red, "W": W_red}
                                best_rank = red_rank
    except Exception:
        pass
    
    # Approach 2: Try ALS refinement for lower rank
    if best_result is not None and time.time() < deadline - 30:
        for target_rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
            if time.time() > deadline - 15:
                break
            try:
                U_new, V_new, W_new, new_rank = _als_refinement(
                    best_result["U"], best_result["V"], best_result["W"],
                    n, m, p, target_rank, max_iter=50, deadline=deadline
                )
                if U_new is not None and new_rank < best_rank:
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                        if _verify_integer_half_integer(U_new, V_new, W_new):
                            best_result = {"U": U_new, "V": V_new, "W": W_new}
                            best_rank = new_rank
                            break
            except Exception:
                continue
    
    # Approach 3: Schoolbook (rank 80) as fallback
    if best_result is None or (time.time() < deadline - 10 and 80 < best_rank):
        try:
            result = _build_schoolbook()
            rank = len(result["U"][0])
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                if best_result is None or rank < best_rank:
                    best_result = result
                    best_rank = rank
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return _build_schoolbook()

# EVOLVE-BLOCK-END
