"""5x5x5 matrix multiplication tensor decomposition with rank minimization via factor-support merging."""
import time
import numpy as np
from itertools import combinations

_DIM = (5, 5, 5)
_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 10.0


def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)


def construct():
    n, m, p = _DIM
    N = n * m
    M = m * p
    P = p * n

    # Build block-based initial decomposition
    def build_initial():
        cols_u = []
        cols_v = []
        cols_w = []

        def add(u_d, v_d, w_d):
            if not u_d or not v_d or not w_d:
                return
            cols_u.append(u_d)
            cols_v.append(v_d)
            cols_w.append(w_d)

        # 2+2+1 partition
        blocks = [[0, 1], [2, 3], [4]]

        def schoolbook(il, jl, kl):
            for i in il:
                for j in jl:
                    for k in kl:
                        add({i * m + j: 1.0}, {j * p + k: 1.0}, {k * n + i: 1.0})

        def strassen2(oi, oj, ok):
            terms = [
                ({(0,0):1,(1,1):1},{(0,0):1,(1,1):1},{(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1},{(0,0):1},{(1,0):1,(1,1):-1}),
                ({(0,0):1},{(0,1):1,(1,1):-1},{(0,1):1,(1,1):1}),
                ({(1,1):1},{(1,0):1,(0,0):-1},{(0,0):1,(1,0):1}),
                ({(0,0):1,(0,1):1},{(1,1):1},{(0,1):1,(0,0):-1}),
                ({(1,0):1,(0,0):-1},{(0,0):1,(0,1):1},{(1,1):1}),
                ({(0,1):1,(1,1):-1},{(1,0):1,(1,1):1},{(0,0):1}),
            ]
            for ac, bc, cc in terms:
                ud, vd, wd = {}, {}, {}
                for (li, lj), v in ac.items():
                    ud[oi + li, oj + lj] = v
                for (lj, lk), v in bc.items():
                    vd[oj + lj, ok + lk] = v
                for (li, lk), v in cc.items():
                    wd[ok + lk, oi + li] = v
                ud = {a * m + b: v for (a, b), v in ud.items()}
                vd = {a * p + b: v for (a, b), v in vd.items()}
                wd = {a * n + b: v for (a, b), v in wd.items()}
                add(ud, vd, wd)

        for bi in range(3):
            for bj in range(3):
                for bk in range(3):
                    if len(blocks[bi]) == 2 and len(blocks[bj]) == 2 and len(blocks[bk]) == 2:
                        strassen2(blocks[bi][0], blocks[bj][0], blocks[bk][0])
                    else:
                        schoolbook(blocks[bi], blocks[bj], blocks[bk])

        return cols_u, cols_v, cols_w

    def merge_by_support(ucs, vcs, wcs):
        """Merge rank-1 tensors with identical nonzero factor support."""
        groups = {}
        for idx in range(len(ucs)):
            key = (tuple(sorted(ucs[idx].items())),
                   tuple(sorted(vcs[idx].items())),
                   tuple(sorted(wcs[idx].items())))
            groups.setdefault(key, []).append(idx)

        result_u, result_v, result_w = [], [], []
        for key, indices in groups.items():
            if len(indices) == 1:
                result_u.append(ucs[indices[0]])
                result_v.append(vcs[indices[0]])
                result_w.append(wcs[indices[0]])
            else:
                merged_u = {}
                merged_v = {}
                merged_w = {}
                for i in indices:
                    for k, v in ucs[i].items():
                        merged_u[k] = merged_u.get(k, 0) + v
                    for k, v in vcs[i].items():
                        merged_v[k] = merged_v.get(k, 0) + v
                    for k, v in wcs[i].items():
                        merged_w[k] = merged_w.get(k, 0) + v
                if any(v != 0 for v in merged_u.values()) and any(v != 0 for v in merged_v.values()) and any(v != 0 for v in merged_w.values()):
                    result_u.append(merged_u)
                    result_v.append(merged_v)
                    result_w.append(merged_w)
        return result_u, result_v, result_w

    # Pass 1: initial build + merge
    cu, cv, cw = build_initial()
    n0 = len(cu)
    cu, cv, cw = merge_by_support(cu, cv, cw)
    n1 = len(cu)

    # Pass 2: linear dependence removal
    if not _deadline_exceeded():
        rank = len(cu)
        # Build U matrix
        U = np.zeros((N, rank), dtype=np.float64)
        for c in range(rank):
            for i, v in cu[c].items():
                U[i, c] = v

        # Find linearly dependent columns
        pivots = []
        for c in range(rank):
            if _deadline_exceeded():
                break
            col = U[:, c]
            dependent = False
            for p in pivots:
                pivot_col = U[:, p]
                pv = pivot_col[p]
                if abs(pv) < 1e-12:
                    continue
                # Check if col is scalar multiple of pivot_col
                if abs(pv) > 1e-12:
                    ratio = col[p] / pv
                    if all(abs(col[i] - ratio * pivot_col[i]) < 1e-10 * (1 + abs(ratio)) for i in range(N)):
                        dependent = True
                        break
                # Check if pivot is scalar multiple of col
                if abs(col[p]) > 1e-12:
                    ratio2 = pv / col[p]
                    if all(abs(pivot_col[i] - ratio2 * col[i]) < 1e-10 * (1 + abs(ratio2)) for i in range(N)):
                        # This shouldn't happen since we're going left to right
                        # but skip this column anyway
                        dependent = True
                        break
            if not dependent:
                pivots.append(c)

        if len(pivots) < rank:
            cu = [cu[i] for i in pivots]
            cv = [cv[i] for i in pivots]
            cw = [cw[i] for i in pivots]

        n2 = len(cu)

    # Pass 3: second merge after removal
    if not _deadline_exceeded() and n1 < n0:
        cu, cv, cw = merge_by_support(cu, cv, cw)
        n3 = len(cu)
    else:
        n3 = n2

    rank = len(cu)

    # Build output matrices
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    for c in range(rank):
        for i, v in cu[c].items():
            U[i, c] = v
        for i, v in cv[c].items():
            V[i, c] = v
        for i, v in cw[c].items():
            W[i, c] = v

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
