"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition via overlapping Strassen."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
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

def _overlapping_strassen():
    """
    Rank-28 decomposition using 4 overlapping 2x2 Strassen cores.
    Each core covers a 2x2 sub-block of the 3x3 matrices.
    Cores: top-left [0:2,0:2], top-right [0:2,1:3],
           bottom-left [1:3,0:2], bottom-right [1:3,1:3]
    """
    n = m = p = 3
    total_terms = 28
    U = [[0.0] * total_terms for _ in range(n * m)]
    V = [[0.0] * total_terms for _ in range(m * p)]
    W = [[0.0] * total_terms for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen 7-term formula for C[i0:i0+2, j0:j0+2] from A[i0:i0+2, k0:k0+2] * B[k0:k0+2, j0:j0+2]
    # Terms are: (U_factors, V_factors, W_factors)
    # U factors use local coords (i', k') -> global (i0+i', k0+k')
    # V factors use local coords (k', j') -> global (k0+k', j0+j')
    # W factors use local coords (j', i') -> global (j0+j', i0+i')
    
    def strassen_core(i0, j0, k0, base_term):
        """Add 7 Strassen terms for the 2x2 block starting at (i0,j0) in C, (i0,k0) in A, (k0,j0) in B."""
        # M1 = A00 * B00 -> C00 += M1, C11 -= M1 (no, standard Strassen)
        # Standard Strassen for C = A*B where C is 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(([
            (u_idx(i0, k0), 1), (u_idx(i0+1, k0+1), 1)
        ], [
            (v_idx(k0, j0), 1), (v_idx(k0+1, j0+1), 1)
        ], [
            (w_idx(j0, i0), 1), (w_idx(j0+1, i0+1), 1)
        ]))
        # M2 = (A10+A11)B00
        terms.append(([
            (u_idx(i0+1, k0), 1), (u_idx(i0+1, k0+1), 1)
        ], [
            (v_idx(k0, j0), 1)
        ], [
            (w_idx(j0, i0+1), 1), (w_idx(j0+1, i0+1), -1)
        ]))
        # M3 = A00(B10-B00)
        terms.append(([
            (u_idx(i0, k0), 1)
        ], [
            (v_idx(k0+1, j0), 1), (v_idx(k0, j0), -1)
        ], [
            (w_idx(j0, i0+1), 1), (w_idx(j0+1, i0+1), 1)
        ]))
        # M4 = A11(B01-B11)
        terms.append(([
            (u_idx(i0+1, k0+1), 1)
        ], [
            (v_idx(k0, j0+1), 1), (v_idx(k0+1, j0+1), -1)
        ], [
            (w_idx(j0, i0), 1), (w_idx(j0+1, i0), 1)
        ]))
        # M5 = (A00+A01)B11
        terms.append(([
            (u_idx(i0, k0), 1), (u_idx(i0, k0+1), 1)
        ], [
            (v_idx(k0+1, j0+1), 1)
        ], [
            (w_idx(j0+1, i0+1), 1), (w_idx(j0, i0+1), 1)
        ]))
        # M6 = (A10-A00)(B00+B10)
        terms.append(([
            (u_idx(i0+1, k0), 1), (u_idx(i0, k0), -1)
        ], [
            (v_idx(k0, j0), 1), (v_idx(k0+1, j0), 1)
        ], [
            (w_idx(j0+1, i0+1), 1)
        ]))
        # M7 = (A01-A11)(B01+B11)
        terms.append(([
            (u_idx(i0, k0+1), 1), (u_idx(i0+1, k0+1), -1)
        ], [
            (v_idx(k0, j0+1), 1), (v_idx(k0+1, j0+1), 1)
        ], [
            (w_idx(j0, i0), 1)
        ]))
        
        for r, (u_list, v_list, w_list) in enumerate(terms):
            tr = base_term + r
            for (row, coeff) in u_list:
                U[row][tr] += coeff
            for (row, coeff) in v_list:
                V[row][tr] += coeff
            for (row, coeff) in w_list:
                W[row][tr] += coeff
    
    # Core 1: C[0:2, 0:2] from A[0:2, 0:2] * B[0:2, 0:2]
    strassen_core(0, 0, 0, 0)
    # Core 2: C[0:2, 1:3] from A[0:2, 1:3] * B[1:3, 1:3]
    strassen_core(0, 1, 1, 7)
    # Core 3: C[1:3, 0:2] from A[1:3, 0:2] * B[0:2, 1:3]
    strassen_core(1, 0, 0, 14)
    # Core 4: C[1:3, 1:3] from A[1:3, 1:3] * B[1:3, 1:3]
    strassen_core(1, 1, 1, 21)
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_result, target_rank, deadline):
    """
    Attempt to reduce the rank of a valid decomposition by merging terms.
    Uses greedy term removal with linear dependence checks.
    """
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    best = None
    best_rank = cur_rank
    
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    removed = set()
    for _ in range(cur_rank - target_rank + 1):
        if time.time() > deadline:
            break
        found = False
        for a in range(cur_rank):
            if a in removed:
                continue
            remaining = [x for x in range(cur_rank) if x != a and x not in removed]
            if len(remaining) < target_rank:
                continue
            
            # Check if term a is in the span of remaining terms
            if len(remaining) >= 1:
                A = np.column_stack([term_tensor(s) for s in remaining])
                b = term_tensor(a)
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    if len(residuals) > 0 and residuals[0] < 1e-7:
                        # Verify by direct computation
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = 0.0
                                    for r in remaining:
                                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                    if abs(s - 1.0) > 1e-9:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            removed.add(a)
                            found = True
                            break
                except:
                    pass
        if not found:
            break
    
    if len(removed) > 0:
        remaining = [x for x in range(cur_rank) if x not in removed]
        new_rank = len(remaining)
        if new_rank < best_rank:
            best_rank = new_rank
            best = {
                "U": U[:, remaining].tolist(),
                "V": V[:, remaining].tolist(),
                "W": W[:, remaining].tolist()
            }
    
    return best, best_rank

def _search_rank_reductions(base_result, deadline):
    """
    Search for lower-rank decompositions using linear dependence and
    systematic subset enumeration.
    """
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    best_result = None
    best_rank = cur_rank
    
    # Strategy 1: Greedy removal with linear dependence
    greedy_result, greedy_rank = _try_reduce_rank(base_result, cur_rank - 8, deadline)
    if greedy_result is not None and greedy_rank < best_rank:
        best_rank = greedy_rank
        best_result = greedy_result
    
    # Strategy 2: Check all pairs for linear dependence and remove simultaneously
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    all_tensors = [term_tensor(r) for r in range(cur_rank)]
    
    # Find all removable terms (in span of others)
    removable = set()
    for r in range(cur_rank):
        if time.time() > deadline:
            break
        others = [s for s in range(cur_rank) if s != r]
        if len(others) < 2:
            continue
        A = np.column_stack([all_tensors[s] for s in others])
        b = all_tensors[r]
        try:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-7:
                removable.add(r)
        except:
            pass
    
    if removable:
        remaining = [x for x in range(cur_rank) if x not in removable]
        # Verify the reduced set
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in remaining:
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            new_rank = len(remaining)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {
                    "U": U[:, remaining].tolist(),
                    "V": V[:, remaining].tolist(),
                    "W": W[:, remaining].tolist()
                }
    
    # Strategy 3: Try to reduce by more through iterative removal
    if best_result is not None and time.time() < deadline - 5:
        U2 = np.array(best_result["U"], dtype=float)
        V2 = np.array(best_result["V"], dtype=float)
        W2 = np.array(best_result["W"], dtype=float)
        cur_rank2 = U2.shape[1]
        
        def term_tensor2(r):
            vec = np.zeros(27)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[i*9 + j*3 + k] = U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r]
            return vec
        
        all_tensors2 = [term_tensor2(r) for r in range(cur_rank2)]
        removed2 = set()
        for r in range(cur_rank2):
            if time.time() > deadline:
                break
            others = [s for s in range(cur_rank2) if s != r and s not in removed2]
            if len(others) < 2:
                continue
            A = np.column_stack([all_tensors2[s] for s in others])
            b = all_tensors2[r]
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-7:
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for rr in others:
                                    if rr in removed2:
                                        continue
                                    s += U2[i*m+j, rr] * V2[j*p+k, rr] * W2[k*n+i, rr]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        removed2.add(r)
            except:
                pass
        
        if removed2:
            remaining2 = [x for x in range(cur_rank2) if x not in removed2]
            if len(remaining2) < best_rank:
                best_rank = len(remaining2)
                best_result = {
                    "U": U2[:, remaining2].tolist(),
                    "V": V2[:, remaining2].tolist(),
                    "W": W2[:, remaining2].tolist()
                }
    
    return best_result, best_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Start with the overlapping Strassen rank-28 construction
    result = _overlapping_strassen()
    if not _verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook
        return _schoolbook()
    
    # Try to reduce the rank
    if time.time() < deadline - 10:
        reduced, reduced_rank = _search_rank_reductions(result, deadline - 5)
        if reduced is not None:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END
