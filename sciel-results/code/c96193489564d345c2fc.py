"""Program for matrix multiplication 2x4 x 4x7 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _target_tensor(n, m, p):
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                for j in range(m):
                    T[i, j, k] = 1.0
    return T

def _verify(U, V, W, T, n, m, p, tol=1e-9):
    U_r = np.array(U).reshape(n, m, -1)
    V_r = np.array(V).reshape(m, p, -1)
    W_r = np.array(W).reshape(p, n, -1)
    R = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return bool(np.max(np.abs(R - T)) <= tol)

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

def _strassen_2x2x2():
    """Standard rank-7 Strassen decomposition for 2x2 . 2x2.
    Returns (M_defs, C_combos) where M_defs[c] = (a_expr, b_expr)
    and C_combos[i][c] gives coefficient for output C[i].
    a_expr/b_expr are length-4 tuples indexed by [i*2+j] for 2x2 matrices.
    """
    M_defs = [
        ((1, 0, 1, 0), (1, 0, 1, 0)),   # M0 = (A11+A21)(B11+B21)
        ((1, 0, 1, 0), (0, 1, 0, 1)),   # M1 = (A11+A21)(B12+B22)
        ((1, 0, -1, 0), (1, 0, -1, 0)), # M2 = (A11-A21)(B11-B21)
        ((0, 0, 0, 1), (0, 1, -1, 0)),  # M3 = A22(B21-B11)
        ((1, 0, 0, 0), (0, 1, 0, -1)),  # M4 = A11(B12-B22)
        ((0, 1, 0, 1), (0, 1, 0, 1)),   # M5 = (A12+A22)(B21+B22)
        ((1, 0, -1, 0), (1, 1, 0, 0)),  # M6 = (A11-A12)(B11+B12)
    ]
    C_combos = [
        [1, 0, 0, 1, 0, -1, 1],  # C11 = M0 + M3 - M5 + M6
        [0, 1, 0, 0, 1, 0, 0],   # C12 = M1 + M4
        [0, 0, 1, 1, 0, 0, 0],   # C21 = M2 + M3
        [0, 0, -1, 0, 1, 0, -1], # C22 = -M2 + M4 - M6
    ]
    return M_defs, C_combos

def _build_structured(n, m, p):
    """Build a structured decomposition using Kronecker-like block approach.
    
    For 2x4 . 4x7:
    - Split A (2x4) into two 2x2 blocks: A = [A_L | A_R]
    - Split B (4x7) into two 2x7 blocks: B = [B_L; B_R]
    - C = A_L * B_L + A_R * B_R (each 2x2 . 2x7)
    
    For each 2x2 . 2x7 block:
    - Use Strassen (rank 7) for the 2x2 . 2x2 part (first 2 columns)
    - Use schoolbook (rank 4) for the remaining 2x2 . 2x5 part
    - Total per block: 11, total: 22
    
    This gives rank 22 vs schoolbook 56.
    """
    nm, mp, pn = n * m, m * p, p * n
    U = []
    V = []
    W = []
    
    M_defs, C_combos = _strassen_2x2x2()
    
    for b in range(2):  # block b=0 (left), b=1 (right)
        # --- Strassen part for first 2 columns of B_b ---
        # A_b is 2x2: entries A[i, j+2*b] for i,j in 0..1
        # B_b first 2 cols is 2x2: entries B[j+2*b, k] for j,k in 0..1
        # Output C[i,k] for i in 0..1, k in 0..1 (from this block)
        
        for c in range(7):
            a_expr, b_expr = M_defs[c]
            uvec = [0.0] * nm
            vvec = [0.0] * mp
            wvec = [0.0] * pn
            
            # Fill U: A_b entries at A[i, j+2*b]
            for i in range(2):
                for j in range(2):
                    coeff = a_expr[i * 2 + j]
                    if coeff != 0:
                        uvec[i * m + (j + 2 * b)] += coeff
            
            # Fill V: B_b first 2 cols at B[j+2*b, k]
            for j in range(2):
                for k in range(2):
                    coeff = b_expr[j * 2 + k]
                    if coeff != 0:
                        vvec[(j + 2 * b) * p + k] += coeff
            
            # Fill W: for each output (i, k) in 0..1 x 0..1
            # C_combos[i][c] is the coefficient
            for i in range(2):
                for k in range(2):
                    coeff = C_combos[i][c]
                    if coeff != 0:
                        # W index: k*n + i
                        wvec[k * n + i] += coeff
            
            U.append(uvec)
            V.append(vvec)
            W.append(wvec)
        
        # --- Schoolbook part for columns k=2..6 of B_b ---
        # A_b is 2x2, B_b remaining is 2x5
        # For each (i, j) in A_b and k in 2..6: one term
        for i in range(2):
            for j in range(2):
                for k in range(2, p):
                    uvec = [0.0] * nm
                    vvec = [0.0] * mp
                    wvec = [0.0] * pn
                    
                    uvec[i * m + (j + 2 * b)] = 1.0
                    vvec[(j + 2 * b) * p + k] = 1.0
                    wvec[k * n + i] = 1.0
                    
                    U.append(uvec)
                    V.append(vvec)
                    W.append(wvec)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}

def _search_lower(n, m, p, T, start_rank, deadline_check):
    """Try to find a lower rank decomposition via structured search with local refinement."""
    rng = random.Random(20240601)
    best = None
    best_rank = start_rank
    
    # Try ranks from start_rank-1 down to 15
    for r in range(min(60, start_rank - 1), 14, -1):
        if not deadline_check():
            break
        found = None
        attempts = 0
        max_attempts = 300 if r > 25 else 800
        while deadline_check() and attempts < max_attempts:
            attempts += 1
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                U[:, c] = rng.choices([-1, 0, 1], k=n * m)
                V[:, c] = rng.choices([-1, 0, 1], k=m * p)
                W[:, c] = rng.choices([-1, 0, 1], k=p * n)
            if _verify(U, V, W, T, n, m, p, 1e-9):
                found = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        if found is not None:
            best = found
            best_rank = r
            break
    return best, best_rank

def _try_block_optimized(n, m, p):
    """Try optimized block decomposition with better splits.
    
    For 2x4 . 4x7:
    Strategy: Split A into two 2x2 blocks.
    For each 2x2 . 2x7 block, use:
      - Strassen (rank 7) for 2x2 . 2x2
      - For remaining 2x2 . 2x5: try to find better than schoolbook(20)
    
    Alternative: Use a single global Strassen-like structure.
    """
    nm, mp, pn = n * m, m * p, p * n
    U = []
    V = []
    W = []
    
    M_defs, C_combos = _strassen_2x2x2()
    
    # Block 0: A cols 0-1, B rows 0-1
    # Block 1: A cols 2-3, B rows 2-3
    for b in range(2):
        a_off = 2 * b
        b_off = 2 * b
        
        # Strassen for 2x2 . 2x2 (B cols 0-1)
        for c in range(7):
            a_expr, b_expr = M_defs[c]
            uvec = [0.0] * nm
            vvec = [0.0] * mp
            wvec = [0.0] * pn
            for i in range(2):
                for j in range(2):
                    coeff = a_expr[i * 2 + j]
                    if coeff != 0:
                        uvec[i * m + (j + a_off)] += coeff
            for j in range(2):
                for k in range(2):
                    coeff = b_expr[j * 2 + k]
                    if coeff != 0:
                        vvec[(j + b_off) * p + k] += coeff
            for i in range(2):
                for k in range(2):
                    coeff = C_combos[i][c]
                    if coeff != 0:
                        wvec[k * n + i] += coeff
            U.append(uvec)
            V.append(vvec)
            W.append(wvec)
        
        # Schoolbook for 2x2 . 2x5 (B cols 2-6)
        for i in range(2):
            for j in range(2):
                for k in range(2, p):
                    uvec = [0.0] * nm
                    vvec = [0.0] * mp
                    wvec = [0.0] * pn
                    uvec[i * m + (j + a_off)] = 1.0
                    vvec[(j + b_off) * p + k] = 1.0
                    wvec[k * n + i] = 1.0
                    U.append(uvec)
                    V.append(vvec)
                    W.append(wvec)
    
    return {"U": U, "V": V, "W": W}

def _build_better_structured(n, m, p):
    """Improved structured decomposition using optimal block splits.
    
    For 2x4 . 4x7:
    - Split into two 2x2 . 2x7 blocks
    - For each 2x2 . 2x7: split B into 2x2 + 2x2 + 2x1
      - 2x2 . 2x2: Strassen rank 7
      - 2x2 . 2x2: Strassen rank 7  
      - 2x2 . 2x1: schoolbook rank 4
      - Total per 2x2 . 2x7: 18
    - Total: 36
    
    Alternative: 2x2 . 2x7 = 2x2 . 2x3 + 2x2 . 2x4
      - 2x2 . 2x3: schoolbook rank 12
      - 2x2 . 2x4: Strassen+Strassen = 14
      - Total per 2x2 . 2x7: 26, total: 52 (worse)
    
    Try yet another: use a rank-11 algorithm for 2x2 . 2x5 if exists.
    For now, use the 18-per-block approach = 36 total, or try to find better.
    """
    nm, mp, pn = n * m, m * p, p * n
    U = []
    V = []
    W = []
    
    M_defs, C_combos = _strassen_2x2x2()
    
    for b in range(2):  # block b=0 (cols 0-1), b=1 (cols 2-3) of B
        a_col_offset = 2 * b  # column offset in A
        b_row_offset = 2 * b  # row offset in B
        
        # Split B_b (2x7) into: cols 0-1 (2x2), cols 2-3 (2x2), cols 4-6 (2x3)
        # Part 1: 2x2 . 2x2 (Strassen, rank 7) - B cols 0-1
        for c in range(7):
            a_expr, b_expr = M_defs[c]
            uvec = [0.0] * nm
            vvec = [0.0] * mp
            wvec = [0.0] * pn
            for i in range(2):
                for j in range(2):
                    coeff = a_expr[i * 2 + j]
                    if coeff != 0:
                        uvec[i * m + (j + a_col_offset)] += coeff
            for j in range(2):
                for k in range(2):
                    coeff = b_expr[j * 2 + k]
                    if coeff != 0:
                        vvec[(j + b_row_offset) * p + k] += coeff
            for i in range(2):
                for k in range(2):
                    coeff = C_combos[i][c]
                    if coeff != 0:
                        wvec[k * n + i] += coeff
            U.append(uvec)
            V.append(vvec)
            W.append(wvec)
        
        # Part 2: 2x2 . 2x2 (Strassen, rank 7) - B cols 2-3
        for c in range(7):
            a_expr, b_expr = M_defs[c]
            uvec = [0.0] * nm
            vvec = [0.0] * mp
            wvec = [0.0] * pn
            for i in range(2):
                for j in range(2):
                    coeff = a_expr[i * 2 + j]
                    if coeff != 0:
                        uvec[i * m + (j + a_col_offset)] += coeff
            for j in range(2):
                for k in range(2):
                    coeff = b_expr[j * 2 + k]
                    if coeff != 0:
                        vvec[(j + b_row_offset) * p + (k + 2)] += coeff
            for i in range(2):
                for k in range(2):
                    coeff = C_combos[i][c]
                    if coeff != 0:
                        wvec[(k + 2) * n + i] += coeff
            U.append(uvec)
            V.append(vvec)
            W.append(wvec)
        
        # Part 3: 2x2 . 2x3 (schoolbook, rank 12) - B cols 4-6
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    uvec = [0.0] * nm
                    vvec = [0.0] * mp
                    wvec = [0.0] * pn
                    uvec[i * m + (j + a_col_offset)] = 1.0
                    vvec[(j + b_row_offset) * p + (k + 4)] = 1.0
                    wvec[(k + 4) * n + i] = 1.0
                    U.append(uvec)
                    V.append(vvec)
                    W.append(wvec)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    def deadline_check():
        return (time.time() - start) < (150.0 - 8.0)
    
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured Kronecker decomposition
    structured = _build_structured(n, m, p)
    if _verify(structured["U"], structured["V"], structured["W"], T, n, m, p, 1e-9):
        s_rank = len(structured["U"])
        if s_rank < best_rank:
            best = structured
            best_rank = s_rank
    
    # Try improved structured decomposition
    structured2 = _build_better_structured(n, m, p)
    if _verify(structured2["U"], structured2["V"], structured2["W"], T, n, m, p, 1e-9):
        s2_rank = len(structured2["U"])
        if s2_rank < best_rank:
            best = structured2
            best_rank = s2_rank
    
    # Try block optimized
    block_opt = _try_block_optimized(n, m, p)
    if _verify(block_opt["U"], block_opt["V"], block_opt["W"], T, n, m, p, 1e-9):
        bo_rank = len(block_opt["U"])
        if bo_rank < best_rank:
            best = block_opt
            best_rank = bo_rank
    
    # Attempt further search for lower rank
    if deadline_check():
        res, r = _search_lower(n, m, p, T, best_rank, deadline_check)
        if res is not None and _verify(res["U"], res["V"], res["W"], T, n, m, p, 1e-9):
            best = res
            best_rank = r
    
    return best
# EVOLVE-BLOCK-END
