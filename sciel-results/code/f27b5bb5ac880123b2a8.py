"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.
Combined approach: structured constraint-satisfaction search with exact
rational/half-integer verification, plus a final polish pass that attempts
to reduce rank by zeroing redundant columns and re-validating.

Key insight: For each rank component c, U[i*m+j,c]*V[j*p+k,c] must be
independent of j. This means for fixed (i,k,c), the triple
(U[i*m+0,c], U[i*m+1,c], U[i*m+2,c]) and (V[0*p+k,c], V[1*p+k,c], V[2*p+k,c])
must satisfy a0*v0 = a1*v1 = a2*v2.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 135.0  # safety margin before 150s kill


def _build_schoolbook():
    n, m, p = DIMENSIONS
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


def _validate_exact(U, V, W, n, m, p):
    """Check that U[i*m+j,c] * V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c.
    Uses exact integer/half-integer arithmetic."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    # Multiply by 2 to work with integers (half-integers become integers)
    U2 = np.round(2 * U).astype(int)
    V2 = np.round(2 * V).astype(int)
    W2 = np.round(2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :]
                rhs = 4 * W2[k * n + i, :]
                if not np.array_equal(lhs, rhs):
                    return False
    return True


def _to_list(M):
    """Convert numpy array to list of lists of int or float (half-integer)."""
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result


def _try_constrained_search(r, n, m, p, seed=0, max_iters=20000):
    """Constrained search: build U and V such that the consistency constraint
    U[i*m+j,c]*V[j*p+k,c] = W[k*n+i,c] is automatically satisfied.
    
    Approach: Pick V values first, then determine valid U values.
    For fixed c, k, and V values (v0, v1, v2), for each i we need
    U[i*m+0,c]*v0 = U[i*m+1,c]*v1 = U[i*m+2,c]*v2 = w_{ikc}
    """
    rng = np.random.default_rng(seed)
    
    # Small values for V entries (doubled)
    v_vals = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6]
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            return None
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        valid = True
        for c in range(r):
            # Pick random V column
            for j in range(m):
                for k in range(p):
                    V[j * p + k, c] = v_vals[rng.integers(len(v_vals))] / 2.0
            
            # For each (i,k), determine if there exists a consistent U row
            for i in range(n):
                for k in range(p):
                    v0 = V[0 * p + k, c]
                    v1 = V[1 * p + k, c]
                    v2 = V[2 * p + k, c]
                    
                    # Need a0*v0 = a1*v1 = a2*v2 = w for some half-integer a0,a1,a2
                    found = False
                    for w2 in range(-20, 21):  # w = w2/4
                        w = w2 / 4.0
                        a0 = a1 = a2 = 0.0
                        ok = True
                        for j, vj in enumerate((v0, v1, v2)):
                            if vj == 0:
                                if w != 0:
                                    ok = False
                                    break
                                a_vals = [0.0, 0.5, -0.5, 1.0, -1.0]
                                aj = a_vals[rng.integers(len(a_vals))]
                            else:
                                aj = w / vj
                                aj2 = aj * 2
                                if abs(aj2 - round(aj2)) > 1e-9:
                                    ok = False
                                    break
                                aj = round(aj2) / 2.0
                            if j == 0:
                                a0 = aj
                            elif j == 1:
                                a1 = aj
                            else:
                                a2 = aj
                        if ok:
                            U[i * m + 0, c] = a0
                            U[i * m + 1, c] = a1
                            U[i * m + 2, c] = a2
                            W[k * n + i, c] = w
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _try_pattern_search(r, n, m, p, seed=0, max_iters=15000):
    """Pattern-based search: enumerate valid (a0,a1,a2) x (v0,v1,v2) patterns
    and assign them randomly to each (i,k,c) tuple."""
    rng = np.random.default_rng(seed)
    
    # Precompute valid patterns
    vals = [0, 1, -1, 2, -2, 3, -3, 4, -4]  # doubled values
    patterns = []
    seen = set()
    for a0 in vals:
        for a1 in vals:
            for a2 in vals:
                for v0 in vals:
                    p0 = a0 * v0
                    for v1 in vals:
                        p1 = a1 * v1
                        if p1 != p0:
                            continue
                        for v2 in vals:
                            p2 = a2 * v2
                            if p2 != p0:
                                continue
                            key = (a0, a1, a2, v0, v1, v2)
                            if key not in seen:
                                seen.add(key)
                                patterns.append((a0 / 2.0, a1 / 2.0, a2 / 2.0,
                                                 v0 / 2.0, v1 / 2.0, v2 / 2.0,
                                                 p0 / 4.0))
    
    nontrivial = [pat for pat in patterns if pat[6] != 0]
    if len(nontrivial) == 0:
        return None
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            return None
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        valid = True
        for c in range(r):
            for i in range(n):
                for k in range(p):
                    if rng.random() < 0.3:
                        a0 = a1 = a2 = v0 = v1 = v2 = 0.0
                        w = 0.0
                    else:
                        pat = nontrivial[rng.integers(len(nontrivial))]
                        a0, a1, a2, v0, v1, v2, w = pat
                    
                    U[i * m + 0, c] = a0
                    U[i * m + 1, c] = a1
                    U[i * m + 2, c] = a2
                    V[0 * p + k, c] = v0
                    V[1 * p + k, c] = v1
                    V[2 * p + k, c] = v2
                    W[k * n + i, c] = w
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _polish_reduce_rank(result, n, m, p, max_rounds=5):
    """Try to reduce rank by zeroing out columns that are linearly redundant.
    For each column c, check if setting it to zero still allows a valid
    decomposition (i.e., the tensor is still covered). If so, remove it."""
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    r = U.shape[1]
    
    # Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    # and check if each (i,j,k) is covered by at least one non-zero column.
    
    for _ in range(max_rounds):
        if time.time() > DEADLINE - 3:
            break
        removed = False
        for c in range(r):
            if time.time() > DEADLINE - 3:
                break
            # Try zeroing column c
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            U_new[:, c] = 0
            V_new[:, c] = 0
            W_new[:, c] = 0
            
            # Check if the decomposition still works
            # We need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] = 1
            # Actually we need the tensor to match exactly.
            # Since we're removing a column, we need to check if the remaining
            # columns still produce the correct tensor.
            
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Check if the product sum is still correct
                        # Original: sum over all c of U[i*m+j,c]*V[j*p+k,c]
                        # New: sum over c != new_c
                        # This is complex; instead just validate directly
                        pass
            
            # Simpler: just validate the new matrices
            if _validate_exact(U_new, V_new, W_new, n, m, p):
                # Remove column c
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                r -= 1
                removed = True
                break
        if not removed:
            break
    
    if r < result["U"][0].__len__():
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    return result


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p  # 72

    # Lower bound: max(n*p, m*p, n*m) = max(24, 24, 9) = 24
    lower = max(n * p, m * p, n * m)
    
    # Try ranks from lower bound upward
    for r in range(lower, best_rank):
        if time.time() > DEADLINE - 5:
            break
        
        seed_base = r * 7919  # prime multiplier for varied seeds
        
        # Strategy 1: constrained search (V-first)
        res = _try_constrained_search(r, n, m, p, seed=seed_base, max_iters=10000)
        if res is not None:
            best = res
            best_rank = r
            # Polish: try to reduce rank further
            polished = _polish_reduce_rank(best, n, m, p)
            if polished["U"][0].__len__() < best_rank:
                best = polished
                best_rank = polished["U"][0].__len__()
            break
        
        # Strategy 2: pattern search
        res = _try_pattern_search(r, n, m, p, seed=seed_base + 1, max_iters=10000)
        if res is not None:
            best = res
            best_rank = r
            # Polish: try to reduce rank further
            polished = _polish_reduce_rank(best, n, m, p)
            if polished["U"][0].__len__() < best_rank:
                best = polished
                best_rank = polished["U"][0].__len__()
            break
    
    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
