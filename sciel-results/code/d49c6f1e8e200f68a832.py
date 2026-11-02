"""Matrix multiplication 3,3,5 rank-minimal tensor decomposition."""
import time
import numpy as np

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
        uc = U[:, r]
        vc = V[:, r]
        wc = W[:, r]
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
    """Rank-7 Strassen factors for 2x2x2, indexed i*2+j, j*2+k, k*2+i."""
    Ufac = []
    Vfac = []
    Wfac = []
    
    # M1: (A00+A11)(B00+B11)
    Ufac.append({0: 1, 3: 1}) 
    Vfac.append({0: 1, 3: 1}) 
    Wfac.append({0: 1, 3: 1}) # C00+=1, C11+=1
    
    # M2: (A10+A11)B00
    Ufac.append({2: 1, 3: 1}) 
    Vfac.append({0: 1}) 
    Wfac.append({2: 1, 3: -1}) # C10+=1, C11-=1
    
    # M3: A00(B01-B11)
    Ufac.append({0: 1}) 
    Vfac.append({1: 1, 3: -1}) 
    Wfac.append({1: 1, 3: 1}) # C01+=1, C11+=1
    
    # M4: A11(B10-B00)
    Ufac.append({3: 1}) 
    Vfac.append({2: 1, 0: -1}) 
    Wfac.append({0: 1, 2: 1}) # C00+=1, C10+=1
    
    # M5: (A00+A01)B11
    Ufac.append({0: 1, 1: 1}) 
    Vfac.append({3: 1}) 
    Wfac.append({0: -1, 1: 1}) # C00-=1, C01+=1
    
    # M6: (A10-A00)(B00+B01)
    Ufac.append({2: 1, 0: -1}) 
    Vfac.append({0: 1, 1: 1}) 
    Wfac.append({3: 1}) # C11+=1
    
    # M7: (A01-A11)(B10+B11)
    Ufac.append({1: 1, 3: -1}) 
    Vfac.append({2: 1, 3: 1}) 
    Wfac.append({0: 1}) # C00+=1
    
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    for r in range(7):
        for idx, val in Ufac[r].items():
            U[idx][r] += val
        for idx, val in Vfac[r].items():
            V[idx][r] += val
        for idx, val in Wfac[r].items():
            W[idx][r] += val
            
    return U, V, W

def _block_candidate(n, m, p):
    """
    Hybrid decomposition:
    1. Start with schoolbook for 3x3x5 (rank 45).
    2. Replace the 2x2x2 sub-block (i,j,k in {0,1}) with Strassen (rank 7).
    3. Total rank = 45 - 8 + 7 = 44.
    """
    sb = _schoolbook(n, m, p)
    U = [row[:] for row in sb["U"]]
    V = [row[:] for row in sb["V"]]
    W = [row[:] for row in sb["W"]]
    
    # Identify schoolbook columns for the 2x2x2 sub-block (i,j,k in {0,1})
    # The schoolbook columns are indexed by (i, j, k) in nested loops.
    # We need to zero out the columns corresponding to i<2, j<2, k<2.
    sub_cols = []
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    sub_cols.append(col)
                col += 1
                
    # Zero out these columns
    for c in sub_cols:
        for row in U:
            row[c] = 0
        for row in V:
            row[c] = 0
        for row in W:
            row[c] = 0
            
    # Get Strassen factors
    sU, sV, sW = _strassen_222()
    
    # Append 7 new columns for Strassen
    rank0 = n * m * p
    for r in range(7):
        for row in U:
            row.append(0)
        for row in V:
            row.append(0)
        for row in W:
            row.append(0)
            
    # Fill in Strassen factors in the new columns with correct index mapping
    # Local indices 0..3 map to global indices based on i,j,k in {0,1}
    for r in range(7):
        # Map U
        for local_u in range(4):
            i_loc = local_u // 2
            j_loc = local_u % 2
            global_u = i_loc * m + j_loc
            U[global_u][rank0 + r] += sU[local_u][r]
            
        # Map V
        for local_v in range(4):
            j_loc = local_v // 2
            k_loc = local_v % 2
            global_v = j_loc * p + k_loc
            V[global_v][rank0 + r] += sV[local_v][r]
            
        # Map W
        for local_w in range(4):
            k_loc = local_w // 2
            i_loc = local_w % 2
            global_w = k_loc * n + i_loc
            W[global_w][rank0 + r] += sW[local_w][r]
            
    # Remove zero columns to get minimal rank
    new_U = []
    new_V = []
    new_W = []
    kept_cols = []
    for c in range(len(U[0])):
        is_zero = True
        if any(U[a][c] != 0 for a in range(len(U))):
            is_zero = False
        if any(V[b][c] != 0 for b in range(len(V))):
            is_zero = False
        if any(W[k][c] != 0 for k in range(len(W))):
            is_zero = False
        if not is_zero:
            kept_cols.append(c)
            
    for a in range(len(U)):
        new_U.append([U[a][c] for c in kept_cols])
    for b in range(len(V)):
        new_V.append([V[b][c] for c in kept_cols])
    for k in range(len(W)):
        new_W.append([W[k][c] for c in kept_cols])
        
    return {"U": new_U, "V": new_V, "W": new_W}

def construct():
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    candidates = []
    
    # 1. Schoolbook (Rank 45)
    candidates.append(_schoolbook(n, m, p))
    
    # 2. Block Candidate with Strassen (Rank 44)
    try:
        if _time_left() > 2.0:
            candidates.append(_block_candidate(n, m, p))
    except Exception:
        pass
        
    # Verify candidates
    best = None
    best_rank = None
    for cand in candidates:
        if _time_left() <= 0.0:
            break
        r = len(cand["U"][0])
        if best_rank is not None and r >= best_rank:
            continue
        try:
            if _verify(cand["U"], cand["V"], cand["W"], T):
                best = cand
                best_rank = r
        except Exception:
            continue
            
    if best is None:
        # fallback schoolbook (always valid)
        best = _schoolbook(n, m, p)
        
    return best
