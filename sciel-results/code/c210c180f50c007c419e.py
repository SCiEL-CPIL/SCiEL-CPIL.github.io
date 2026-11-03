# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 135.0  # safety margin before 150s kill


def _cost_full(a, b, p):
    """Exact integer cost for assignment p (facility i -> location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _init_solution(a, b):
    """Build a strong initial permutation via a greedy / assignment-based heuristic."""
    n = len(a)
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    B2 = B + B.T  # symmetric part for distance pairing
    A2 = A + A.T  # symmetric part for flow

    # Objective (symmetric form): sum_ij A2[i,j] * B2[p[i],p[j]] / 2
    # Greedy: place facilities one by one, each to the location minimizing
    # incremental cost given already-placed facilities.
    placed = np.zeros(n, dtype=bool)
    p = np.empty(n, dtype=np.int64)
    loc_used = np.zeros(n, dtype=bool)
    # contribution of placing facility i at location l:
    #   sum over placed j: A2[i,j]*B2[l, p[j]] + A2[j,i]*B2[p[j], l]
    # = sum over placed j: A2[i,j]*B2[l, p[j]]  (A2 symmetric)
    # Precompute nothing fancy; do O(n^3) greedy which is fine for n=150 (3.4M).
    contrib = np.zeros((n, n), dtype=np.float64)  # contrib[i, l]
    order = np.argsort(-A2.sum(axis=1), kind="stable")  # place high-flow facilities first
    for i in order:
        # choose l minimizing contrib[i,l] among unused
        avail = np.where(~loc_used)[0]
        if len(avail) == 1:
            l = avail[0]
        else:
            l = avail[np.argmin(contrib[i, avail])]
        p[i] = l
        loc_used[l] = True
        # update contrib for remaining facilities k:
        #   contrib[k, x] += A2[k,i]*B2[x, l]
        contrib[:, :] += A2[:, i, None] * B2[:, l][None, :]
    return p.astype(np.int64)


def _tabu_search(a, b, p0, t_limit):
    """Tabu search with fast swap-delta. Returns best permutation."""
    n = len(a)
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    p = p0.copy()

    # r[i] = row-sum contribution: sum_j A[i,j]*B[p[i], p[j]]
    # c[l] = col-sum contribution: sum_j A[j,i]*B[p[j], l] where i is facility at loc l
    # cost = sum_i r[i]
    r = np.sum(A * B[:, p], axis=1)          # r[i] = sum_j A[i,j]*B[p[i],p[j]]
    # c for location l: sum_j A[j, fac_at_l] * B[p[j], l]
    c = np.zeros(n, dtype=np.float64)
    # c[l] = sum_j A[j, inv[p[j]]] * B[p[j], l]
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)
    # For each j, facility fac=inv[p[j]], contributes A[fac, j] * B[p[j], l] to c[l]
    # c[l] = sum_j A[inv[p[j]], j] * B[p[j], l]
    w = A[inv, np.arange(n)]  # w[j] = A[inv[p[j]], j]
    c = B.T @ w               # c[l] = sum_j B[p[j], l] * w[j]

    cur = float(np.sum(r))
    best = cur
    best_p = p.copy()

    tabu = np.zeros(n, dtype=np.int64)  # until-iteration for each facility
    it = 0
    tabu_len = 18
    rng = np.random.default_rng(12345)

    # Precompute A rows and B rows access via p
    while it < 2000000:
        if time.time() > t_limit:
            break
        # Evaluate all single swaps (i, j) i != j
        # delta(i,j) = -r[i] - r[j] - c[p[i]] - c[p[j]]
        #             + A[i,j]*B[p[i],p[j]] + A[j,i]*B[p[j],p[i]]
        #             + sum_k A[i,k]*B[p[j],p[k]] + sum_k A[j,k]*B[p[i],p[k]]
        #             + sum_k A[k,i]*B[p[k],p[j]] + sum_k A[k,j]*B[p[k],p[i]]
        # Compute efficiently:
        # s_i_to_l = sum_k A[i,k]*B[l,p[k]]  = (A[i,:] * B[:,p]) ... no
        # Let's define M[i, l] = sum_k A[i,k]*B[l, p[k]]  (row i, candidate location l)
        # and N[i, l] = sum_k A[k,i]*B[p[k], l]
        # delta(i,j) = -r[i]-r[j]-c[p[i]]-c[p[j]]
        #              + A[i,j]*B[p[i],p[j]] + A[j,i]*B[p[j],p[i]]
        #              + M[i, p[j]] + M[j, p[i]] + N[i, p[j]] + N[j, p[i]]
        # But M[i,p[i]] = r[i] and N[i,p[i]] = c[p[i]] (since p[i] is i's location)
        # So M[i,p[j]] = r[i] - A[i,j]*B[p[i],p[j]]  => we can avoid full M.
        # M[i, p[j]] = sum_k A[i,k]*B[p[j],p[k]] = r[i] - A[i,j]*B[p[j],p[i]]? No.
        # r[i] = sum_k A[i,k]*B[p[i],p[k]]. M[i,p[j]] differs only in B's first index.
        # Not simply related. Compute M and N via matrix products each iteration is O(n^3).
        # n=150 -> 3.4M per iter, too slow for many iters.
        #
        # Alternative: precompute for each i the vector over locations l:
        #   M[i, l] = sum_k A[i,k]*B[l,p[k]]  (n x n)  recompute when p changes.
        # Updating M after a swap is O(n^2) (two rows of B columns change).
        # Let's maintain M (n x n) and N (n x n).
        pass

    # Redo with maintained M and N
    # M[i, l] = sum_k A[i,k]*B[l, p[k]]
    # N[i, l] = sum_k A[k,i]*B[p[k], l]
    M = B @ (A * B[:, p].T)  # M[i,l] = sum_k A[i,k]*B[l,p[k]]
    # verify: (A * B[:,p].T)[i,k] = A[i,k]*B[p[k],i]? No. B[:,p] has shape (n,n): B[x, p[k]] at [x,k].
    # B[:,p].T [k, l] = B[l, p[k]]. So M[i,l] = sum_k A[i,k]*B[:,p].T[k,l] = (A @ B[:,p].T)[i,l].
    M = A @ B[:, p].T
    # N[i,l] = sum_k A[k,i]*B[p[k],l] = (A.T @ B[:,p].T)[i,l]? A.T[k,i]=A[i,k]. no.
    # N[i,l] = sum_k A[k,i]*B[p[k],l]. A[:,i] is column i. (A[:,i] * B[:,p].T row?)
    # B[:,p].T[k,l]=B[l,p[k]]. So N[i,l]=sum_k A[k,i]*B[:,p].T[k,l] = (A[:,i] * B[:,p].T) sum over k
    # = (B[:,p] @ A[:,i])? B[:,p][l,k]=B[p[k],l]. (B[:,p] @ A[:,i])[l] = sum_k B[p[k],l]*A[k,i]. yes = N[i,l].
    N = B[:, p] @ A  # N[l,i] = sum_k B[p[k],l]*A[k,i]; we want N[i,l], so transpose
    N = N.T  # N[i,l] = sum_k A[k,i]*B[p[k],l]

    r = M[np.arange(n), p]  # r[i]=M[i,p[i]]
    c = N[np.arange(n), p]  # c[i]=N[i,p[i]] = sum_k A[k,i]*B[p[k],p[i]]
    cur = float(np.sum(r))
    best = cur
    best_p = p.copy()
    it = 0
    no_improve = 0
    while it < 300000:
        if time.time() > t_limit:
            break
        # delta(i,j) for i!=j:
        # = -r[i]-r[j]-c[i]-c[j] + A[i,j]*B[p[i],p[j]] + A[j,i]*B[p[j],p[i]]
        #   + M[i,p[j]] + M[j,p[i]] + N[i,p[j]] + N[j,p[i]]
        # M[i,p[i]]=r[i], N[i,p[i]]=c[i].
        # Compute full delta matrix D (n x n), D[i,j]
        # Build:
        Di = -r[:, None] - r[None, :] - c[:, None] - c[None, :]
        # add A[i,j]*B[p[i],p[j]]: matrix A * B[np.ix_(p,p)]
        Bpp = B[np.ix_(p, p)]
        Di += A * Bpp
        # add A[j,i]*B[p[j],p[i]] = (A.T * Bpp.T)
        Di += A.T * Bpp.T
        # add M[i,p[j]]: matrix M[:, p]  (M[i, p[j]])
        Di += M[:, p]
        # add M[j,p[i]] = M[p, i] = (M[:, p].T)
        Di += M[:, p].T
        # add N[i,p[j]] = N[:, p]
        Di += N[:, p]
        # add N[j,p[i]] = N[p, i] = N[:, p].T
        Di += N[:, p].T
        np.fill_diagonal(Di, np.inf)
        # tabu: forbid i if tabu[i] > it
        tabu_mask = tabu > it
        Di[tabu_mask, :] = np.inf
        Di[:, tabu_mask] = np.inf
        ii, jj = np.unravel_index(np.argmin(Di), Di.shape)
        d = Di[ii, jj]
        if not np.isfinite(d):
            break
        # perform swap ii, jj
        pi, pj = p[ii], p[jj]
        p[ii], p[jj] = pj, pi
        # update r, c, M, N, cur
        # r[ii] = M[ii, p[ii]] (new), r[jj] = M[jj, p[jj]]
        # But M must be updated first. Update M and N for the two swapped locations.
        # After swap: p[ii]=pj, p[jj]=pi.
        # M[i, l] = sum_k A[i,k]*B[l,p[k]]. Only p[ii],p[jj] changed.
        # M[:, pi] = sum_k A[:,k]*B[pi,p[k]]: k=ii term changes (p[ii]=pj now), k=jj term changes (p[jj]=pi).
        # Update M[:, pi]: old contribution k=ii: A[:,ii]*B[pi,pi]; new: A[:,ii]*B[pi,pj]
        #   k=jj: old A[:,jj]*B[pi,pj]; new A[:,jj]*B[pi,pi]
        # M[:, pi] += A[:,ii]*(B[pi,pj]-B[pi,pi]) + A[:,jj]*(B[pi,pi]-B[pi,pj])
        M[:, pi] += A[:, ii] * (B[pi, pj] - B[pi, pi]) + A[:, jj] * (B[pi, pi] - B[pi, pj])
        # M[:, pj]: k=ii old A[:,ii]*B[pj,pi] new A[:,ii]*B[pj,pj]; k=jj old A[:,jj]*B[pj,pj] new A[:,jj]*B[pj,pi]
        M[:, pj] += A[:, ii] * (B[pj, pj] - B[pj, pi]) + A[:, jj] * (B[pj, pi] - B[pj, pj])
        # N[i, l] = sum_k A[k,i]*B[p[k],l]. p[ii],p[jj] changed.
        # N[:, pi]: k=ii old A[ii,pi]*B[pi,pi] new A[ii,pi]*B[pj,pi]; k=jj old A[jj,pi]*B[pj,pi] new A[jj,pi]*B[pi,pi]
        N[:, pi] += A[ii, pi] * (B[pj, pi] - B[pi, pi]) + A[jj, pi] * (B[pi, pi] - B[pj, pi])
        # N[:, pj]: k=ii old A[ii,pj]*B[pi,pj] new A[ii,pj]*B[pj,pj]; k=jj old A[jj,pj]*B[pj,pj] new A[jj,pj]*B[pi,pj]
        N[:, pj] += A[ii, pj] * (B[pj, pj] - B[pi, pj]) + A[jj, pj] * (B[pi, pj] - B[pj, pj])
        # r[i]=M[i,p[i]], c[i]=N[i,p[i]]
        r[ii] = M[ii, pj]
        r[jj] = M[jj, pi]
        c[ii] = N[ii, pj]
        c[jj] = N[jj, pi]
        cur += d
        # tabu
        tabu[ii] = it + tabu_len
        tabu[jj] = it + tabu_len
        it += 1
        if cur < best - 1e-9:
            best = cur
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
        # diversify tabu length occasionally
        if no_improve > 300:
            tabu_len = 10 + int(rng.integers(0, 25))
            no_improve = 0
    return best_p, best


def solve(a, b):
    n = len(a)
    p0 = _init_solution(a, b)
    # Multi-restart tabu search within deadline
    best_p = p0.copy()
    best_val = _cost_full(a, b, p0)
    restarts = 0
    while time.time() < _DEADLINE - 8.0:
        t_limit = _DEADLINE - 6.0
        p, val = _tabu_search(a, b, p0.copy(), t_limit)
        if val < best_val:
            best_val = val
            best_p = p.copy()
        restarts += 1
        if time.time() > _DEADLINE - 12.0:
            break
    return best_p.tolist()
# EVOLVE-BLOCK-END
