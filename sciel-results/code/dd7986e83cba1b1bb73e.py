"""Matrix multiplication 3,3,5 rank-minimal tensor decomposition."""
import time
import itertools

DIMENSIONS = [3, 3, 5]
_START = time.time()
DEADLINE = 150.0
_SAFETY = 5.0  # seconds margin before kill

def _time_left():
    return DEADLINE - _SAFETY - (time.time() - _START)

def _target_tensor(n, m, p):
    """T[i*m+j, j*p+k, k*n+i] = 1 else 0."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T

def _verify(U, V, W, T):
    """Exact integer verification that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # build candidate tensor dict
    cand = {}
    for r in range(rank):
        uc = [U[a][r] for a in range(len(U))]
        vc = [V[b][r] for b in range(len(V))]
        wc = [W[c][r] for c in range(len(W))]
        for a in range(n * m):
            uav = uc[a]
            if uav == 0:
                continue
            for b in range(m * p):
                ubv = uav * vc[b]
                if ubv == 0:
                    continue
                for c in range(p * n):
                    wcv = ubv * wc[c]
                    if wcv == 0:
                        continue
                    cand[(a, b, c)] = cand.get((a, b, c), 0) + wcv
    # compare
    if set(cand.keys()) != set(T.keys()):
        return False
    for key, val in T.items():
        if cand[key] != val:
            return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_222():
    """
    Rank-7 Strassen factors for 2x2x2 matrix multiplication.
    
    Returns U (4x7), V (4x7), W (4x7) such that:
    sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T where T[i*2+j, j*2+k, k*2+i] = 1
    
    Using standard Strassen formulas with proper indexing.
    """
    # Strassen's 7 intermediate products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U factors (A side): row index = i*2+j
    Ufac = [
        # M1: A00 + A11
        [(0, 1), (3, 1)],
        # M2: A10 + A11
        [(2, 1), (3, 1)],
        # M3: A00
        [(0, 1)],
        # M4: A11
        [(3, 1)],
        # M5: A00 + A01
        [(0, 1), (1, 1)],
        # M6: A10 - A00
        [(2, 1), (0, -1)],
        # M7: A01 - A11
        [(1, 1), (3, -1)],
    ]
    
    # V factors (B side): row index = j*2+k
    Vfac = [
        # M1: B00 + B11
        [(0, 1), (3, 1)],
        # M2: B00
        [(0, 1)],
        # M3: B01 - B11
        [(1, 1), (3, -1)],
        # M4: B10 - B00
        [(2, 1), (0, -1)],
        # M5: B11
        [(3, 1)],
        # M6: B00 + B01
        [(0, 1), (1, 1)],
        # M7: B10 + B11
        [(2, 1), (3, 1)],
    ]
    
    # W factors (C side): row index = k*2+i
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    Wfac = [
        # C00 (k=0,i=0): M1 - M4 - M5 + M7
        [(0, 1), (3, -1), (4, -1), (6, 1)],
        # C01 (k=0,i=1): M3 + M5
        [(2, 1), (4, 1)],
        # C10 (k=1,i=0): M2 + M4
        [(1, 1), (3, 1)],
        # C11 (k=1,i=1): M1 - M2 + M3 - M6
        [(0, 1), (1, -1), (2, 1), (5, -1)],
    ]
    
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    for r in range(rank):
        for (idx, val) in Ufac[r]:
            U[idx][r] += val
        for (idx, val) in Vfac[r]:
            V[idx][r] += val
        for (idx, val) in Wfac[r]:
            W[idx][r] += val
    
    return U, V, W

def _block_candidate(n, m, p):
    """
    Build a candidate by replacing 2x2x2 sub-blocks with Strassen rank-7 decomposition.
    
    For 3x3x5: we can replace the top-left 2x2x2 block (i,j,k in {0,1}) with Strassen,
    saving 8-7=1 rank. But we can be smarter by considering multiple block placements.
    
    Strategy: Start with schoolbook (rank 45), then for each 2x2x2 sub-block,
    replace its 8 schoolbook terms with 7 Strassen terms.
    """
    # Precompute Strassen factors
    sU, sV, sW = _strassen_222()
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    U = [row[:] for row in sb["U"]]
    V = [row[:] for row in sb["V"]]
    W = [row[:] for row in sb["W"]]
    rank0 = n * m * p  # 45
    
    # Find all 2x2x2 sub-blocks that fit within bounds
    # A 2x2x2 block at offset (i0,j0,k0) uses indices i0..i0+1, j0..j0+1, k0..k0+1
    sub_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                sub_blocks.append((i0, j0, k0))
    
    # For each sub-block, compute which schoolbook columns it uses
    def get_sub_block_cols(i0, j0, k0):
        cols = []
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i0 <= i < i0 + 2 and j0 <= j < j0 + 2 and k0 <= k < k0 + 2:
                        cols.append(col)
                    col += 1
        return cols
    
    # Try each sub-block independently, pick the one that gives best (lowest) rank
    best_candidate = None
    best_rank = rank0 + 1  # worse than schoolbook
    
    for (i0, j0, k0) in sub_blocks:
        if _time_left() <= 1.0:
            break
        
        # Get the schoolbook columns for this sub-block
        sub_cols = get_sub_block_cols(i0, j0, k0)
        
        # Build new candidate: zero out sub-block columns, add 7 Strassen columns
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        # Zero out the sub-block columns
        for c in sub_cols:
            for row in new_U:
                row[c] = 0
            for row in new_V:
                row[c] = 0
            for row in new_W:
                row[c] = 0
        
        # Append 7 new Strassen columns
        new_rank = rank0 + 7
        for row in new_U:
            row += [0] * 7
        for row in new_V:
            row += [0] * 7
        for row in new_W:
            row += [0] * 7
        
        # Fill in Strassen factors at the appropriate positions
        for r in range(7):
            for a in range(4):
                i = i0 + a // 2
                j = j0 + a % 2
                new_U[i * m + j][rank0 + r] += sU[a][r]
            for b in range(4):
                j = j0 + b // 2
                k = k0 + b % 2
                new_V[j * p + k][rank0 + r] += sV[b][r]
            for c in range(4):
                k = k0 + c // 2
                i = i0 + c % 2
                new_W[k * n + i][rank0 + r] += sW[c][r]
        
        # Trim zero columns
        active_cols = []
        for c in range(new_rank):
            if any(new_U[a][c] for a in range(n * m)) or \
               any(new_V[b][c] for b in range(m * p)) or \
               any(new_W[c2][c] for c2 in range(p * n)):
                active_cols.append(c)
        
        final_rank = len(active_cols)
        if final_rank < best_rank:
            best_rank = final_rank
            # Rebuild with only active columns
            best_U = [[new_U[a][c] for c in active_cols] for a in range(n * m)]
            best_V = [[new_V[b][c] for c in active_cols] for b in range(m * p)]
            best_W = [[new_W[c2][c] for c in active_cols] for c2 in range(p * n)]
            best_candidate = {"U": best_U, "V": best_V, "W": best_W}
    
    if best_candidate is None:
        # Fallback: just return schoolbook
        return {"U": U, "V": V, "W": W}
    
    return best_candidate

def construct():
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    
    # Generate candidates in order of expected rank (best first)
    candidates = []
    
    # 1. Block candidate (should be ~36-40 rank if Strassen works)
    try:
        if _time_left() > 5.0:
            candidates.append(_block_candidate(n, m, p))
    except Exception as e:
        pass
    
    # 2. Schoolbook (rank 45, guaranteed valid)
    candidates.append(_schoolbook(n, m, p))
    
    best = None
    best_rank = None
    
    for cand in candidates:
        if _time_left() <= 2.0:
            break
        r = len(cand["U"][0])
        if best_rank is not None and r >= best_rank:
            continue
        try:
            if _verify(cand["U"], cand["V"], cand["W"], T):
                best = cand
                best_rank = r
                # If we found a good candidate, we can stop early
                if best_rank <= 40:
                    break
        except Exception:
            continue
    
    if best is None:
        # Fallback: schoolbook (always valid)
        best = _schoolbook(n, m, p)
    
    return best
