"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
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

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    terms = []
    # Term 0: M1 = (A00+A11)(B00+B11)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # Term 1: M2 = (A10+A11)B00
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    # Term 2: M3 = A00(B10-B11)
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): -1}
    })
    # Term 3: M4 = A11(B01-B00)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    # Term 4: M5 = (A00+A01)B11
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # Term 5: M6 = (A10-A00)(B00+B01)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # Term 6: M7 = (A01-A11)(B10+B11)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """Build a 4x4x4 decomposition using recursive Strassen (rank 49)."""
    n = 4
    terms_2x2 = _strassen_2x2()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            for (bi, bj), coeff_u in term1['U'].items():
                for (li, lj), coeff_v in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i*4 + j][idx] += coeff_u * coeff_v
            
            for (bj, bk), coeff_v in term1['V'].items():
                for (lj, lk), coeff_w in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j*4 + k][idx] += coeff_v * coeff_w
            
            for (bk, bi), coeff_w in term1['W'].items():
                for (lk, li), coeff_u2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k*4 + i][idx] += coeff_w * coeff_u2
    
    return {"U": U, "V": V, "W": W}

def _build_4x4x5():
    """Build a 4x4x5 decomposition: 4x4x4 (rank 49) + 4x4x1 (rank 4) = rank 53."""
    n, m, p = DIMENSIONS
    
    base = _build_4x4x4_strassen()
    rank_base = 49
    
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m * 4):
        for r in range(rank_base):
            V[j][r] = base["V"][j][r]
    for k in range(4 * n):
        for r in range(rank_base):
            W[k][r] = base["W"][k][r]
    
    for i in range(n):
        for j in range(m):
            r = rank_base + i
            U[i*m + j][r] = 1
            V[j*p + 4][r] = 1
            W[4*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_remove_term(U, V, W, n, m, p, term_idx):
    """
    Try to remove term_idx by expressing its contribution as a linear combination
    of other terms. Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    
    We check if removing term_idx still gives a valid decomposition.
    If not, we try to find coefficients to add to other terms to compensate.
    """
    rank = len(U[0])
    if rank <= 1:
        return False, None, None, None
    
    # First, just check if we can simply remove the term
    U_new = [row[:term_idx] + row[term_idx+1:] for row in U]
    V_new = [row[:term_idx] + row[term_idx+1:] for row in V]
    W_new = [row[:term_idx] + row[term_idx+1:] for row in W]
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        return True, U_new, V_new, W_new
    
    # Try to find a linear combination: for each remaining term r, add alpha_r * term_r
    # to compensate for the removed term
    # The contribution of term_idx is: U[idx][i*m+j] * V[idx][j*p+k] * W[idx][k*n+i]
    # We need: sum_r alpha_r * U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i] = U[idx][i*m+j] * V[idx][j*p+k] * W[idx][k*n+i]
    
    # This is a nonlinear system, so we use a simpler approach:
    # Try pairs of terms to see if one can be expressed as a combination
    
    return False, None, None, None

def _gaussian_elimination_reduce(U, V, W, n, m, p, time_budget=100):
    """
    Perform systematic elimination to reduce rank.
    Strategy: 
    1. Check if any term is linearly dependent on others in a way that allows removal
    2. Use the structure of the problem to find simplifications
    """
    start_time = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    rank = len(U[0])
    removed = set()
    
    # Try removing each term one at a time
    for attempt in range(10):  # Limit iterations
        if time.time() - start_time > time_budget:
            break
        
        progress = False
        for term_idx in range(rank):
            if term_idx in removed:
                continue
            if time.time() - start_time > time_budget:
                break
            
            success, U_new, V_new, W_new = _try_remove_term(U, V, W, n, m, p, term_idx)
            if success:
                U, V, W = U_new, V_new, W_new
                removed.add(term_idx)
                rank = len(U[0])
                progress = True
                break
        
        if not progress:
            break
    
    return U, V, W

def _try_merge_two_terms(U, V, W, n, m, p, t1, t2):
    """
    Try to merge terms t1 and t2 into a single term.
    Two rank-1 tensors can be merged if their product structure allows it.
    """
    rank = len(U[0])
    if t1 == t2 or rank <= 2:
        return False, None, None, None
    
    # The merged term would have:
    # U_merged[i*m+j] = a*U[t1][i*m+j] + b*U[t2][i*m+j]
    # V_merged[j*p+k] = c*V[t1][j*p+k] + d*V[t2][j*p+k]  
    # W_merged[k*n+i] = e*W[t1][k*n+i] + f*W[t2][k*n+i]
    
    # For the product to work, we need specific relationships.
    # A simpler check: if one term's U,V,W vectors are scalar multiples of the other's
    # in a compatible way, they can be merged.
    
    # Check if U[t1] is proportional to U[t2]
    def _is_proportional(vec1, vec2):
        """Check if vec1 = c * vec2 for some scalar c."""
        nonzero = [(i, vec1[i], vec2[i]) for i in range(len(vec1)) if vec2[i] != 0]
        if not nonzero:
            return None
        c = nonzero[0][1] / nonzero[0][2]
        for i, v1, v2 in nonzero:
            if abs(v1 - c * v2) > 1e-9:
                return None
        # Check zeros are consistent
        for i in range(len(vec1)):
            if vec2[i] == 0 and abs(vec1[i]) > 1e-9:
                return None
        return c
    
    c_u = _is_proportional(U[t1], U[t2])
    if c_u is None:
        return False, None, None, None
    
    c_v = _is_proportional(V[t1], V[t2])
    if c_v is None:
        return False, None, None, None
    
    c_w = _is_proportional(W[t1], W[t2])
    if c_w is None:
        return False, None, None, None
    
    # If all proportional, check if the products are consistent
    # U[t1]*V[t1]*W[t1] = c_u*U[t2] * c_v*V[t2] * c_w*W[t2] = (c_u*c_v*c_w) * U[t2]*V[t2]*W[t2]
    # So term t1 = (c_u*c_v*c_w) * term t2, meaning we can just scale t2
    
    scale = c_u * c_v * c_w
    
    # Remove t1, scale t2 by (1 + scale)
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    # Scale t2
    for i in range(len(U[t2])):
        U_new[i][t2] = U_new[i][t2] + scale * U_new[i][t2]
        V_new[i][t2] = V_new[i][t2] + scale * V_new[i][t2]
        W_new[i][t2] = W_new[i][t2] + scale * W_new[i][t2]
    
    # Remove t1
    U_new = [row[:t1] + row[t1+1:] for row in U_new]
    V_new = [row[:t1] + row[t1+1:] for row in V_new]
    W_new = [row[:t1] + row[t1+1:] for row in W_new]
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        return True, U_new, V_new, W_new
    
    return False, None, None, None

def _reduce_rank(U, V, W, n, m, p, time_budget=100):
    """
    Combined rank reduction: first try simple removals, then try merges.
    """
    start_time = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Phase 1: Try removing individual terms
    U, V, W = _gaussian_elimination_reduce(U, V, W, n, m, p, time_budget=30)
    
    if time.time() - start_time > time_budget:
        return U, V, W
    
    # Phase 2: Try merging pairs of terms
    rank = len(U[0])
    for attempt in range(5):
        if time.time() - start_time > time_budget:
            break
        
        progress = False
        for t1 in range(rank):
            if time.time() - start_time > time_budget:
                break
            for t2 in range(t1+1, rank):
                success, U_new, V_new, W_new = _try_merge_two_terms(U, V, W, n, m, p, t1, t2)
                if success:
                    U, V, W = U_new, V_new, W_new
                    rank = len(U[0])
                    progress = True
                    break
            if progress:
                break
        
        if not progress:
            break
    
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 53)
    try:
        result = _build_4x4x5()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            # Try to reduce rank
            U_reduced, V_reduced, W_reduced = _reduce_rank(
                result["U"], result["V"], result["W"], n, m, p, 
                time_budget=120
            )
            
            # Verify reduced version
            if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
                # Convert to proper format
                result = {
                    "U": [[float(x) for x in row] for row in U_reduced],
                    "V": [[float(x) for x in row] for row in V_reduced],
                    "W": [[float(x) for x in row] for row in W_reduced]
                }
                return result
        # If reduction failed or verification failed, return original
        return {
            "U": [[float(x) for x in row] for row in result["U"]],
            "V": [[float(x) for x in row] for row in result["V"]],
            "W": [[float(x) for x in row] for row in result["W"]]
        }
    except Exception:
        pass
    
    # Fallback to schoolbook
    schoolbook = _build_schoolbook()
    return {
        "U": [[float(x) for x in row] for row in schoolbook["U"]],
        "V": [[float(x) for x in row] for row in schoolbook["V"]],
        "W": [[float(x) for x in row] for row in schoolbook["W"]]
    }
# EVOLVE-BLOCK-END
