"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Search for linear dependencies among the 44 column-vectors to reduce rank.
  3. Try structured rank-r decompositions for r < 44.
  4. Validate any candidate with exact rational arithmetic.
  5. If no improvement is found within the time budget, return the best known decomposition.
"""
import time
import sys
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # 15 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Column vector extraction ----
    def _col_vector(U, V, W, c):
        vec = [0] * num_terms
        for i in range(n):
            for a in range(m):
                u_val = U[i * m + a][c]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[a * p + k][c]
                    if v_val == 0:
                        continue
                    w_val = W[k * n + i][c]
                    if w_val == 0:
                        continue
                    vec[i * m * p + a * p + k] = u_val * v_val * w_val
        return vec

    # ---- Search for rank reduction via linear dependency ----
    def _search_rank_reduction():
        nonlocal best_candidate, best_rank

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        cols = []
        for c in range(rank0):
            cols.append(_col_vector(U, V, W, c))

        target = [1] * num_terms

        # Verify base
        total = [0] * num_terms
        for cv in cols:
            for idx in range(num_terms):
                total[idx] += cv[idx]
        for idx in range(num_terms):
            if total[idx] != 1:
                return

        # Check for duplicate columns
        seen = {}
        removable = []
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                removable.append(c)
            else:
                seen[key] = c

        if removable:
            for c_rem in removable:
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for c in range(rank0):
                    if c == c_rem:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][c]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][c]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][c]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # Search for a linear combination of columns that sums to zero
        # If sum_{c in S} cols[c] = 0, then we can drop one column from S
        # and adjust the others, OR if there's a dependency, we can reduce rank.
        
        # Try small subsets (size 2 and 3) for exact zero sum
        # Size 2: cols[i] + cols[j] = 0 means cols[i] = -cols[j]
        for i in range(rank0):
            if not _elapsed_ok():
                return
            for j in range(i + 1, rank0):
                dep = True
                for idx in range(num_terms):
                    if cols[i][idx] + cols[j][idx] != 0:
                        dep = False
                        break
                if dep:
                    # cols[i] = -cols[j], so we can drop one
                    keep = [c for c in range(rank0) if c != j]
                    new_rank = rank0 - 1
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    nc = 0
                    for c in keep:
                        for i2 in range(n * m):
                            new_U[i2][nc] = U[i2][c]
                        for i2 in range(m * p):
                            new_V[i2][nc] = V[i2][c]
                        for i2 in range(p * n):
                            new_W[i2][nc] = W[i2][c]
                        nc += 1
                    cand = {"U": new_U, "V": new_V, "W": new_W}
                    if _validate(cand):
                        best_candidate = cand
                        best_rank = new_rank
                        return

        if not _elapsed_ok():
            return

        # Try size-3 dependencies: cols[i] + cols[j] + cols[k] = 0
        for i in range(rank0):
            if not _elapsed_ok():
                return
            for j in range(i + 1, rank0):
                if not _elapsed_ok():
                    return
                for k in range(j + 1, rank0):
                    dep = True
                    for idx in range(num_terms):
                        if cols[i][idx] + cols[j][idx] + cols[k][idx] != 0:
                            dep = False
                            break
                    if dep:
                        keep = [c for c in range(rank0) if c != k]
                        new_rank = rank0 - 1
                        new_U = [[0] * new_rank for _ in range(n * m)]
                        new_V = [[0] * new_rank for _ in range(m * p)]
                        new_W = [[0] * new_rank for _ in range(p * n)]
                        nc = 0
                        for c in keep:
                            for i2 in range(n * m):
                                new_U[i2][nc] = U[i2][c]
                            for i2 in range(m * p):
                                new_V[i2][nc] = V[i2][c]
                            for i2 in range(p * n):
                                new_W[i2][nc] = W[i2][c]
                            nc += 1
                        cand = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(cand):
                            best_candidate = cand
                            best_rank = new_rank
                            return

        if not _elapsed_ok():
            return

        # Try random linear combinations with small integer coefficients
        # Look for sum_c alpha_c * cols[c] = 0 with small integers alpha_c
        # This is a lattice problem; try random sparse combinations
        random.seed(42)
        for trial in range(5000):
            if not _elapsed_ok():
                return
            # Pick 3-5 random columns with random coefficients in {-2,-1,1,2}
            size = random.choice([3, 4, 5])
            indices = random.sample(range(rank0), size)
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(size)]
            
            dep = True
            for idx in range(num_terms):
                s = 0
                for ci, cc in zip(indices, coeffs):
                    s += cc * cols[ci][idx]
                if s != 0:
                    dep = False
                    break
            if dep:
                # Found a dependency! Drop the last index
                drop_idx = indices[-1]
                keep = [c for c in range(rank0) if c != drop_idx]
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for c in keep:
                    for i2 in range(n * m):
                        new_U[i2][nc] = U[i2][c]
                    for i2 in range(m * p):
                        new_V[i2][nc] = V[i2][c]
                    for i2 in range(p * n):
                        new_W[i2][nc] = W[i2][c]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

    if _elapsed_ok():
        _search_rank_reduction()

    if best_rank < 44:
        return best_candidate

    # ---- Try structured lower-rank decompositions ----
    # Approach: Use rank-10 decomposition for 2x3 * 3x2 blocks.
    # If we can find a rank-10 decomposition for 2x3 * 3x2, 
    # we can use it for each of the 4 blocks of 2 columns in B.
    # Total rank: 4 * 10 = 40.
    
    def _find_rank_10_2x3x2():
        """
        Search for a rank-10 decomposition of the 2x3x2 multiplication tensor.
        Returns U, V, W or None.
        """
        n, m, p = 2, 3, 2
        rank = 10
        # U: 6x10, V: 6x10, W: 4x10
        # Indices:
        # U[i*m+a], V[a*p+k], W[k*n+i]
        
        # We use a random search with local improvement.
        # Target: sum_r U[i,a,r] V[a,k,r] W[k,i,r] == 1 for all i,a,k.
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        def _calc_err(U, V, W):
            err = 0
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
                        err += (s - 1) ** 2
            return err

        def _random_init():
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
            return U, V, W

        # Try a few random restarts
        for restart in range(100):
            if not _elapsed_ok():
                return None
            
            U, V, W = _random_init()
            err = _calc_err(U, V, W)
            
            # Local search: change one entry at a time
            improved = True
            while improved and _elapsed_ok():
                improved = False
                # Pick a random variable to change
                # 180 variables total
                var_type = random.choice(['U', 'V', 'W'])
                if var_type == 'U':
                    idx = random.randint(0, n*m-1)
                    r = random.randint(0, rank-1)
                    old_val = U[idx][r]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val: continue
                        U[idx][r] = new_val
                        new_err = _calc_err(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                        else:
                            U[idx][r] = old_val
                elif var_type == 'V':
                    idx = random.randint(0, m*p-1)
                    r = random.randint(0, rank-1)
                    old_val = V[idx][r]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val: continue
                        V[idx][r] = new_val
                        new_err = _calc_err(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                        else:
                            V[idx][r] = old_val
                else:
                    idx = random.randint(0, p*n-1)
                    r = random.randint(0, rank-1)
                    old_val = W[idx][r]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val: continue
                        W[idx][r] = new_val
                        new_err = _calc_err(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                        else:
                            W[idx][r] = old_val
                
                if err == 0:
                    return U, V, W

        return None

    def _build_rank_40():
        """
        Build rank-40 decomposition using 4 blocks of rank-10 2x3x2 decompositions.
        """
        res = _find_rank_10_2x3x2()
        if not res:
            return None
        
        U_block, V_block, W_block = res
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block_idx in range(4):
            k0 = block_idx * 2
            offset = block_idx * 10
            
            # Copy U_block to U
            # U_block rows: i*m+a (i in 0..1, a in 0..2)
            # U rows: i*m+a (same)
            for i in range(n):
                for a in range(m):
                    for r in range(10):
                        U[i*m+a][offset + r] = U_block[i*m+a][r]
            
            # Copy V_block to V
            # V_block rows: a*p_block+k_block (a in 0..2, k_block in 0..1)
            # V rows: a*p+k (a in 0..2, k in 0..7)
            # For block j, k = k0 + k_block
            for a in range(m):
                for k_block in range(2):
                    for r in range(10):
                        v_idx_block = a*2 + k_block
                        v_idx_global = a*p + (k0 + k_block)
                        V[v_idx_global][offset + r] = V_block[v_idx_block][r]
            
            # Copy W_block to W
            # W_block rows: k_block*n+i (k_block in 0..1, i in 0..1)
            # W rows: k*n+i (k in 0..7, i in 0..1)
            # For block j, k = k0 + k_block
            for k_block in range(2):
                for i in range(n):
                    for r in range(10):
                        w_idx_block = k_block*n + i
                        w_idx_global = (k0 + k_block)*n + i
                        W[w_idx_global][offset + r] = W_block[w_idx_block][r]
        
        return {"U": U, "V": V, "W": W}

    # ---- Aggressive search: try to find ANY rank-r decomposition for r < 44 ----
    # This is a constraint satisfaction problem. Use random initialization + local search.
    if _elapsed_ok():
        cand_40 = _build_rank_40()
        if cand_40:
            if _validate(cand_40):
                best_candidate = cand_40
                best_rank = 40

    return best_candidate

# EVOLVE-BLOCK-END
