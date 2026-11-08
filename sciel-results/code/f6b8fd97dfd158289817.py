"""Search for a rank-minimal decomposition of the 2x3x10 matrix-mult tensor.

Combines:
  * the current program's structural sparse-search + solve-for-W machinery,
  * tensor-slicing / column-elimination to certify lower-rank solutions,
  * a minimum-weight perfect-matching (MWM) formulation on the bipartite
    graph of input monomials (6 from A, 30 from B) to systematically pick
    the smallest set of bilinear terms that span the 60-dim output space,
  * fast full-tensor verification and an exact rational polish pass.
"""

import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [2, 3, 10]
TARGET_DEADLINE = 140.0  # safety margin under the 150s kill

VALS = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
VALS_EXTENDED = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0])


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Fast full-tensor verification.

    Returns (ok, max_err). Checks ALL tensor entries: the n*p*m non-zero
    entries must equal 1, and every other (a,b,c) triple must equal 0.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    # Reshape to 3D tensors: Ua[i,j,r], Va[j,k,r], Wa[k,i,r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)

    # T[i,k,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    target = np.ones((n, p, m))
    err_nz = float(np.max(np.abs(T - target)))

    if err_nz > tol:
        return False, err_nz

    # Zero entries: check only the specific zero positions
    # For matrix mult, zeros occur at (i*m+j, j'*p+k, k'*n+i') where j != j' or k != k'
    # More efficiently: check all (a,b,c) where the support is 0
    # But we can be smarter: the tensor T_full[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # We need T_full[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and T_full[a,b,c] = 0 otherwise
    
    # Build the full tensor but only check a subset of zeros for speed
    # Actually, let's just build it all - it's only 6*30*20 = 3600 entries
    full = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    
    # Create support
    support = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                support[i * m + j, j * p + k, k * n + i] = 1.0
    
    err_z = float(np.max(np.abs(full - support)))
    max_err = max(err_nz, err_z)
    return (max_err < tol), max_err


def _current_rank55(n, m, p):
    """Build the known rank-55 solution (seed)."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1; W[k1 + 2][col] += 1
        col += 1
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[k0 + 2][col] += 1; W[k1 + 2][col] -= 1
        col += 1
        U[0][col] += 1
        V[k1][col] += 1; V[10 + k1][col] -= 1
        W[k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
        U[4][col] += 1
        V[10 + k0][col] += 1; V[k0][col] -= 1
        W[k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        U[0][col] += 1; U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0][col] -= 1; W[k1][col] += 1
        col += 1
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[k1 + 2][col] += 1
        col += 1
        U[1][col] += 1; U[4][col] -= 1
        V[10 + k0][col] += 1; V[10 + k1][col] += 1
        W[k0][col] += 1
        col += 1
        U[2][col] += 1; V[20 + k0][col] += 1; W[k0][col] += 1
        col += 1
        U[2][col] += 1; V[20 + k1][col] += 1; W[k1][col] += 1
        col += 1
        U[5][col] += 1; V[20 + k0][col] += 1; W[k0 + 2][col] += 1
        col += 1
        U[5][col] += 1; V[20 + k1][col] += 1; W[k1 + 2][col] += 1
        col += 1
    return {"U": U, "V": V, "W": W}


def _solve_W(U, V, n, m, p, R):
    """Given U and V, solve for W so the tensor identity holds.

    For each (i,k): M[j,r] = U[i*m+j,r]*V[j*p+k,r]; need M @ w_ik = 1.
    Returns list-of-lists W or None if infeasible.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, R))
            for j in range(m):
                M[j] = Ua[i * m + j] * Va[j * p + k]
            try:
                w, residuals, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
            except Exception:
                return None
            if residuals.size > 0 and residuals[0] > 1e-7:
                return None
            W[k * n + i] = w
    return W.tolist()


def _check_column_redundant(U, V, W, n, m, p, R, col_idx):
    """Check if column col_idx is redundant by trying to eliminate it.
    
    Returns True if the column can be removed while maintaining validity.
    """
    # Remove column col_idx
    U2 = np.array(U, dtype=float)
    V2 = np.array(V, dtype=float)
    
    U2_reduced = np.delete(U2, col_idx, axis=1)
    V2_reduced = np.delete(V2, col_idx, axis=1)
    
    # Solve for W with reduced U, V
    W2 = _solve_W(U2_reduced.tolist(), V2_reduced.tolist(), n, m, p, R - 1)
    if W2 is None:
        return False
    
    ok, _ = _verify_full(U2_reduced.tolist(), V2_reduced.tolist(), W2, n, m, p)
    return ok


def _rand_sparse(nrows, rank, density, rng, use_extended=False):
    vals = VALS_EXTENDED if use_extended else VALS
    A = np.zeros((nrows, rank))
    for r_ in range(nrows):
        for c_ in range(rank):
            if rng.random() < density:
                A[r_, c_] = vals[rng.randrange(len(vals))]
    return A.tolist()


def _try_rank(R, n, m, p, rng, max_iters=2000, deadline=TARGET_DEADLINE):
    """Random sparse U,V + solve-for-W search for a valid rank-R solution."""
    nm, mp = n * m, m * p
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        # Vary density based on iteration to explore different sparsity levels
        base_density = 0.15 + 0.15 * (it % 4) / 4.0
        use_ext = (it % 3 == 0)
        U = _rand_sparse(nm, R, base_density, rng, use_ext)
        V = _rand_sparse(mp, R, base_density, rng, use_ext)
        W = _solve_W(U, V, n, m, p, R)
        if W is None:
            continue
        ok, _ = _verify_full(U, V, W, n, m, p)
        if ok:
            return U, V, W
    return None


def _mwm_build_cost_matrix(n, m, p, R, rng):
    """Build the cost matrix for the MWM formulation.

    Nodes: 6 A-monomials (i,j) and 30 B-monomials (j,k).
    Edge weight: -1 if (i,j,k) is an output position (i.e. the product
    contributes to C[i,k]), else a small positive penalty.  The MWM then
    picks R disjoint pairs that maximise coverage of the 60 outputs while
    minimising "wasted" pairs.  We solve for a perfect matching of size
    min(36, R) on a balanced bipartite graph.
    """
    a_nodes = [(i, j) for i in range(n) for j in range(m)]  # 6
    b_nodes = [(j, k) for j in range(m) for k in range(p)]  # 30
    na, nb = len(a_nodes), len(b_nodes)

    # Balance the graph: pad the smaller side with dummy nodes.
    N = max(na, nb)
    # Cost matrix N x N.  We want to *minimise* cost, so use negative
    # reward for useful edges.
    cost = np.full((N, N), 1.0, dtype=float)  # default penalty
    for ai, (i, j) in enumerate(a_nodes):
        for bi, (j2, k) in enumerate(b_nodes):
            if j == j2:
                # (i,j,k) is a valid output position -> reward
                cost[ai, bi] = -10.0
    # Dummy nodes: zero cost to match with each other
    for d in range(max(na, nb) - min(na, nb)):
        idx = min(na, nb) + d
        if idx < N:
            cost[idx, idx] = 0.0

    return cost, a_nodes, b_nodes


def _try_rank_mwm(R, n, m, p, rng, max_restarts=50, deadline=TARGET_DEADLINE):
    """MWM-guided structured search for a valid rank-R solution.

    Repeatedly:
      1. Build a randomised cost matrix (perturb useful edges slightly).
      2. Solve the assignment problem.
      3. From the matched pairs, build sparse U, V with the chosen
         monomial positions non-zero (random values from VALS).
      4. Solve for W and verify.
    """
    nm, mp = n * m, m * p
    cost_base, a_nodes, b_nodes = _mwm_build_cost_matrix(n, m, p, R, rng)
    N = cost_base.shape[0]

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        # Randomise the cost to explore different matchings
        cost = cost_base.copy()
        for ai in range(len(a_nodes)):
            for bi in range(len(b_nodes)):
                if cost[ai, bi] < 0:  # useful edge
                    cost[ai, bi] += rng.gauss(0, 0.5)

        row_ind, col_ind = linear_sum_assignment(cost)

        # Extract matched (A-node, B-node) pairs (skip dummy-dummy)
        pairs = []
        for r, c in zip(row_ind, col_ind):
            if r < len(a_nodes) and c < len(b_nodes):
                i, j = a_nodes[r]
                j2, k = b_nodes[c]
                if j == j2:
                    pairs.append((i, j, k))

        if len(pairs) < R:
            continue

        # Build U, V from the matched pairs
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        for r_idx, (i, j, k) in enumerate(pairs[:R]):
            U[i * m + j, r_idx] = VALS[rng.randrange(len(VALS))]
            V[j * p + k, r_idx] = VALS[rng.randrange(len(VALS))]

        # Solve for W
        W = _solve_W(U.tolist(), V.tolist(), n, m, p, R)
        if W is None:
            continue
        ok, _ = _verify_full(U.tolist(), V.tolist(), W, n, m, p)
        if ok:
            return U.tolist(), V.tolist(), W
    return None


def _polish_rational(U, V, W, n, m, p, tol=1e-6):
    """Snap float entries to the nearest integer/half-integer if consistent.

    Returns (U,V,W) with clean entries, or None if snapping breaks validity.
    """
    def snap(M):
        a = np.array(M, dtype=float)
        b = np.round(a * 2.0) / 2.0
        return b.tolist()

    Us, Vs, Ws = snap(U), snap(V), snap(W)
    ok, _ = _verify_full(Us, Vs, Ws, n, m, p, tol=1e-6)
    if ok:
        return Us, Vs, Ws
    return None


def _eliminate_columns(U, V, W, n, m, p, rng, deadline=TARGET_DEADLINE):
    """Tensor-slicing elimination: repeatedly try to drop one rank column.

    A column r is redundant if the tensor it contributes is a linear
    combination of the others, i.e. if we can find new U',V',W' of one
    fewer column matching the full tensor. We test this by removing column r
    and re-solving for W' (and, when needed, absorbing the residual into the
    remaining columns). A practical, exact test: drop column r, keep U,V
    unchanged (zeroed in column r), solve for W'; if the full tensor still
    matches, the column was redundant.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    R = len(U[0])
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        # Try each column; order randomized for diversity
        order = list(range(R))
        rng.shuffle(order)
        for r in order:
            if time.time() > deadline - 5:
                break
            # Candidate: delete column r
            U2 = [row[:r] + row[r + 1:] for row in U]
            V2 = [row[:r] + row[r + 1:] for row in V]
            W2 = _solve_W(U2, V2, n, m, p, R - 1)
            if W2 is None:
                continue
            ok, _ = _verify_full(U2, V2, W2, n, m, p)
            if ok:
                U, V, W = U2, V2, W2
                R -= 1
                improved = True
                break
    return U, V, W


def _local_optimize(U, V, W, n, m, p, rng, deadline=TARGET_DEADLINE, max_rounds=50):
    """Local optimization: perturb U,V slightly and re-solve for W.
    
    This can escape local minima in the elimination process.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    R = len(U[0])
    
    for _ in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed copy
        U_pert = [row[:] for row in U]
        V_pert = [row[:] for row in V]
        
        # Perturb a few random entries
        for _ in range(3):
            r_idx = rng.randrange(len(U_pert))
            c_idx = rng.randrange(R)
            U_pert[r_idx][c_idx] += rng.choice([-0.5, 0.5])
            
            r_idx = rng.randrange(len(V_pert))
            c_idx = rng.randrange(R)
            V_pert[r_idx][c_idx] += rng.choice([-0.5, 0.5])
        
        # Try to solve for W with perturbed U, V
        W_new = _solve_W(U_pert, V_pert, n, m, p, R)
        if W_new is None:
            continue
        
        ok, _ = _verify_full(U_pert, V_pert, W_new, n, m, p)
        if ok:
            U, V, W = U_pert, V_pert, W_new
    
    # After local opt, try elimination again
    U, V, W = _eliminate_columns(U, V, W, n, m, p, rng, deadline)
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)

    # Start with the known rank-55 solution.
    best = _current_rank55(n, m, p)
    best_rank = 55
    ok, err = _verify_full(best["U"], best["V"], best["W"], n, m, p)
    if not ok:
        # Fallback: trivial rank-60 solution.
        U = [[0] * 60 for _ in range(n * m)]
        V = [[0] * 60 for _ in range(m * p)]
        W = [[0] * 60 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = 60

    # 1) Systematically eliminate redundant columns from the seed.
    Ue, Ve, We = _eliminate_columns(best["U"], best["V"], best["W"], n, m, p, rng)
    ok, _ = _verify_full(Ue, Ve, We, n, m, p)
    if ok and len(Ue[0]) < best_rank:
        best = {"U": Ue, "V": Ve, "W": We}
        best_rank = len(Ue[0])

    # 2) Local optimization to escape local minima
    if time.time() < TARGET_DEADLINE - 20:
        Uo, Vo, Wo = _local_optimize(best["U"], best["V"], best["W"], n, m, p, rng, TARGET_DEADLINE)
        ok, _ = _verify_full(Uo, Vo, Wo, n, m, p)
        if ok and len(Uo[0]) < best_rank:
            best = {"U": Uo, "V": Vo, "W": Wo}
            best_rank = len(Uo[0])

    # 3) Rational polish the seed (and any improved solution).
    Up, Vp, Wp = _polish_rational(best["U"], best["V"], best["W"], n, m, p)
    if Up is not None:
        best = {"U": Up, "V": Vp, "W": Wp}

    # 4) Search for lower ranks via MWM-guided structured search + random, then eliminate.
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > TARGET_DEADLINE - 15:
            break
        # Alternate between MWM-guided and random search
        result = _try_rank_mwm(target_rank, n, m, p, rng, max_restarts=100)
        if result is None and time.time() < TARGET_DEADLINE - 15:
            result = _try_rank(target_rank, n, m, p, rng, max_iters=800)
        if result is not None:
            U, V, W = result
            U, V, W = _eliminate_columns(U, V, W, n, m, p, rng)
            # Also try local optimization
            if time.time() < TARGET_DEADLINE - 10:
                U, V, W = _local_optimize(U, V, W, n, m, p, rng, TARGET_DEADLINE)
            ok, _ = _verify_full(U, V, W, n, m, p)
            if ok:
                Ur, Vr, Wr = _polish_rational(U, V, W, n, m, p)
                if Ur is not None:
                    U, V, W = Ur, Vr, Wr
                best = {"U": U, "V": V, "W": W}
                best_rank = len(U[0])
                break

    # Deadline guard.
    if time.time() > TARGET_DEADLINE - 3:
        return best
    return best


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    ok, err = _verify_full(res["U"], res["V"], res["W"], n, m, p)
    print(f"Rank: {len(res['U'][0])}, Valid: {ok}, MaxErr: {err:.2e}")
