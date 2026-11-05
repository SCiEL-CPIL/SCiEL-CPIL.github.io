import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 5, 6
    nm, mp, pn = n*m, m*p, p*n

    # Precomputed minimal ranks for small block sizes via exhaustive DP.
    # Base: rank(1,a,b)=a*b, rank(a,1,b)=a*b, rank(a,b,1)=a*b
    # rank(2,2,2)=7 (Strassen)
    # rank(2,2,3)=12 (Strassen on 2x2x2 + 4 schoolbook for 2x2x1)
    # rank(2,3,2)=12 (Strassen on 2x2x2 + 4 schoolbook for 2x1x2)
    # rank(3,2,2)=12 (Strassen on 2x2x2 + 4 schoolbook for 1x2x2)
    # rank(2,2,4)=16 (two Strassen 2x2x2 blocks)
    # rank(2,4,2)=16, rank(4,2,2)=16
    # rank(2,3,3)=18, rank(3,2,3)=18, rank(3,3,2)=18
    # rank(3,3,3)=27? Let's compute: 2x2x2(7) + 1x2x2(4) + 2x1x2(4) + 2x2x1(4) + 1x1x2(2) + 1x2x1(2) + 2x1x1(2) + 1x1x1(1) = 26? No that's not how DP works.
    # Actually for 3x3x3: split into 2x3x3 + 1x3x3.
    # rank(2,3,3): split 2x2x3 + 2x1x3 = 12 + 6 = 18. Or 2x3x2 + 2x3x1 = 12 + 6 = 18.
    # rank(1,3,3) = 9. So rank(3,3,3) = 18 + 9 = 27.
    # rank(2,3,4): split 2x2x4 + 2x1x4 = 16 + 8 = 24. Or 2x3x2 + 2x3x2 = 12+12=24.
    # rank(2,4,3): 24. rank(3,4,2): 24.
    # rank(3,3,4): 3x3x2 + 3x3x2 = 18 + 18 = 36? Or 2x3x4 + 1x3x4 = 24 + 12 = 36.
    # rank(4,4,4): 2x4x4 + 2x4x4 = 16+16=32? Or 4x2x4+4x2x4 = 16+16=32. Or 4x4x2+4x4x2=16+16=32.

    def compute_rank(ni, mj, pk):
        """Return minimal rank for ni x mj x pk tensor via DP."""
        # Cache results in a dict
        key = (ni, mj, pk)
        if key in _rank_cache:
            return _rank_cache[key]
        
        if ni == 1 or mj == 1 or pk == 1:
            r = ni * mj * pk
            _rank_cache[key] = r
            return r
        if (ni, mj, pk) == (2, 2, 2):
            _rank_cache[key] = 7
            return 7
        # Symmetry: rank is invariant under permutation of (ni, mj, pk)
        # But the structure of U, V, W matters for construction, not for rank
        sorted_dims = tuple(sorted((ni, mj, pk)))
        # Actually the rank IS symmetric: rank(n,m,p) = rank(perm(n,m,p))
        # Let's use this to reduce computation
        # But for construction we need the actual split, so compute_rank just needs to be correct.
        
        best = ni * mj * pk  # schoolbook
        if ni > 1:
            for s in range(1, ni):
                r = compute_rank(s, mj, pk) + compute_rank(ni - s, mj, pk)
                if r < best:
                    best = r
        if mj > 1:
            for s in range(1, mj):
                r = compute_rank(ni, s, pk) + compute_rank(ni, mj - s, pk)
                if r < best:
                    best = r
        if pk > 1:
            for s in range(1, pk):
                r = compute_rank(ni, mj, s) + compute_rank(ni, mj, pk - s)
                if r < best:
                    best = r
        
        _rank_cache[key] = best
        return best

    _rank_cache = {}
    total_rank = compute_rank(n, m, p)

    def best_split(ni, mj, pk):
        """Return (dim_name, split_size) for the best first split."""
        if ni == 1 or mj == 1 or pk == 1:
            return None
        if (ni, mj, pk) == (2, 2, 2):
            return None
        if (ni, mj, pk) in [(1,1,1), (1,1,2), (1,2,1), (2,1,1), (1,2,2), (2,1,2), (2,2,1), (1,1,3), (1,3,1), (3,1,1)]:
            return None
        # Check small cases where schoolbook is optimal
        if ni <= 2 and mj <= 2 and pk <= 2:
            return None
            
        best_rank = ni * mj * pk
        best_choice = None
        if ni > 1:
            for s in range(1, ni):
                r = compute_rank(s, mj, pk) + compute_rank(ni - s, mj, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('ni', s)
        if mj > 1:
            for s in range(1, mj):
                r = compute_rank(ni, s, pk) + compute_rank(ni, mj - s, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('mj', s)
        if pk > 1:
            for s in range(1, pk):
                r = compute_rank(ni, mj, s) + compute_rank(ni, mj, pk - s)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('pk', s)
        return best_choice

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Strassen for 2x2x2 (rank 7)
    strassen = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]

    def add_strassen(col, i_off, j_off, k_off):
        nonlocal U, V, W
        for um, vm, wm in strassen:
            for ii in range(2):
                for jj in range(2):
                    if um[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj), col] += um[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if vm[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk), col] += vm[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if wm[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii), col] += wm[kk][ii]
            col += 1
        return col

    def add_schoolbook(col, i_off, j_off, k_off, ni, mj, pk):
        nonlocal U, V, W
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    U[(i_off+ii)*m + (j_off+jj), col] = 1.0
                    V[(j_off+jj)*p + (k_off+kk), col] = 1.0
                    W[(k_off+kk)*n + (i_off+ii), col] = 1.0
                    col += 1
        return col

    def decompose(ni, mj, pk, i_off, j_off, k_off, col):
        if time.time() > deadline:
            raise TimeoutError()
        if ni == 2 and mj == 2 and pk == 2:
            return add_strassen(col, i_off, j_off, k_off)
        if ni == 1 or mj == 1 or pk == 1:
            return add_schoolbook(col, i_off, j_off, k_off, ni, mj, pk)
        if ni <= 2 and mj <= 2 and pk <= 2:
            return add_schoolbook(col, i_off, j_off, k_off, ni, mj, pk)
        choice = best_split(ni, mj, pk)
        if choice is None:
            return add_schoolbook(col, i_off, j_off, k_off, ni, mj, pk)
        dim, split = choice
        if dim == 'ni':
            col = decompose(split, mj, pk, i_off, j_off, k_off, col)
            col = decompose(ni - split, mj, pk, i_off + split, j_off, k_off, col)
        elif dim == 'mj':
            col = decompose(ni, split, pk, i_off, j_off, k_off, col)
            col = decompose(ni, mj - split, pk, i_off, j_off + split, k_off, col)
        else:
            col = decompose(ni, mj, split, i_off, j_off, k_off, col)
            col = decompose(ni, mj, pk - split, i_off, j_off, k_off + split, col)
        return col

    col = decompose(n, m, p, 0, 0, 0, 0)
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify and fallback
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(total_rank))
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)

    if error > 1e-6:
        rank_s = n*m*p
        Us = np.zeros((nm, rank_s), dtype=float)
        Vs = np.zeros((mp, rank_s), dtype=float)
        Ws = np.zeros((pn, rank_s), dtype=float)
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, c2] = 1.0
                    Vs[j*p+k, c2] = 1.0
                    Ws[k*n+i, c2] = 1.0
                    c2 += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
