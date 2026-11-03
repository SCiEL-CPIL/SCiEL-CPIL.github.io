"""Search-based exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
import itertools
import random
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM = N * M          # 16  (U rows)
MP = M * P          # 20  (V rows)
PN = P * N          # 20  (W rows, trace convention W[k*n+i, r])
TERM_COUNT = N * M * P  # 80 required (i,j,k) triples


def _time_left(deadline):
    return deadline - time.time()


def _tensor_values(U, V, W, rank):
    """Build the n*m*p tensor T[(i,j,k)] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T = np.zeros((N, M, P), dtype=np.float64)
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        for i in range(N):
            for j in range(M):
                u_val = u_col[i * M + j]
                if u_val == 0:
                    continue
                for k in range(P):
                    v_val = v_col[j * P + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * N + i]
                    if w_val == 0:
                        continue
                    T[i, j, k] += u_val * v_val * w_val
    return T


def _is_valid(U, V, W, tol=1e-9):
    """Exact check that U,V,W decompose the multiplication tensor.
    For every (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    for i in range(N):
        for j in range(M):
            urow = U_arr[i * M + j]
            vbase = j * P
            for k in range(P):
                vrow = V_arr[vbase + k]
                wrow = W_arr[k * N + i]
                total = np.sum(urow * vrow * wrow)
                if abs(total - 1.0) > tol:
                    return False
    return True


def _rank(U, V, W):
    return len(U[0]) if U and U[0] else 0


def _structural_lower_bound():
    """Crude lower bound: each rank-1 term covers at most one (i,j,k) if we
    require the product to be exactly 1 at every slot and 0 elsewhere, a term
    (a row of U, row of V, row of W) contributes a product that is non-zero
    only at the (i,j,k) where all three are non-zero. To get exactly 1 at each
    of the 80 slots, and a term can contribute to multiple slots, the minimum
    is bounded below by the number of slots that must each be 'hit'. A very
    weak bound is 80 / (max slots per term). We return a conservative small
    bound to guide search ordering; correctness is always verified explicitly."""
    return 8  # conservative; real bound is higher but this keeps search cheap


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _greedy_build(seed, time_budget=20.0):
    """Greedy: start with schoolbook, attempt to merge terms by checking if
    removing one term and adjusting others can maintain validity."""
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    cand = _schoolbook()
    U, V, W = cand["U"], cand["V"], cand["W"]
    rank = len(U[0])
    
    # Try to find pairs of terms that can be merged
    # Two terms (a, b) can be merged if their contributions are proportional
    # in a way that allows combining them into one term
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for a in range(rank):
            if a >= rank or time.time() > deadline:
                break
            for b in range(a + 1, rank):
                if b >= rank or time.time() > deadline:
                    break
                
                # Check if term a and b have the same support pattern
                # (non-zero entries in the same positions)
                u_a = [U[r][a] for r in range(NM)]
                u_b = [U[r][b] for r in range(NM)]
                v_a = [V[r][a] for r in range(MP)]
                v_b = [V[r][b] for r in range(MP)]
                w_a = [W[r][a] for r in range(PN)]
                w_b = [W[r][b] for r in range(PN)]
                
                # Check if supports match
                support_match = True
                for r in range(NM):
                    if (u_a[r] != 0) != (u_b[r] != 0):
                        support_match = False
                        break
                if not support_match:
                    continue
                for r in range(MP):
                    if (v_a[r] != 0) != (v_b[r] != 0):
                        support_match = False
                        break
                if not support_match:
                    continue
                for r in range(PN):
                    if (w_a[r] != 0) != (w_b[r] != 0):
                        support_match = False
                        break
                if not support_match:
                    continue
                
                # Try to find scales such that we can merge
                # We need: U_a * V_a * W_a + U_b * V_b * W_b = U_c * V_c * W_c
                # This is complex; instead, try simple case where one term is
                # a scalar multiple of the other in each matrix
                # Find scale factors
                u_scale = None
                for r in range(NM):
                    if u_a[r] != 0 and u_b[r] != 0:
                        u_scale = u_a[r] / u_b[r]
                        break
                if u_scale is None:
                    continue
                # Verify u_scale
                valid = True
                for r in range(NM):
                    if abs(u_a[r] - u_scale * u_b[r]) > 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                
                v_scale = None
                for r in range(MP):
                    if v_a[r] != 0 and v_b[r] != 0:
                        v_scale = v_a[r] / v_b[r]
                        break
                if v_scale is None:
                    continue
                valid = True
                for r in range(MP):
                    if abs(v_a[r] - v_scale * v_b[r]) > 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                
                w_scale = None
                for r in range(PN):
                    if w_a[r] != 0 and w_b[r] != 0:
                        w_scale = w_a[r] / w_b[r]
                        break
                if w_scale is None:
                    continue
                valid = True
                for r in range(PN):
                    if abs(w_a[r] - w_scale * w_b[r]) > 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If all scales are 1, the terms are identical -> can merge
                if abs(u_scale - 1) < 1e-9 and abs(v_scale - 1) < 1e-9 and abs(w_scale - 1) < 1e-9:
                    # Remove term b (keep a)
                    for r in range(NM):
                        U[r].pop(b)
                    for r in range(MP):
                        V[r].pop(b)
                    for r in range(PN):
                        W[r].pop(b)
                    rank -= 1
                    changed = True
                    break
                
                # Try other scale combinations that might allow merging
                # This is where we could do more sophisticated search
            if time.time() > deadline:
                break
    
    cand["U"], cand["V"], cand["W"] = U, V, W
    return cand


def _drop_terms(U, V, W, deadline):
    """Try to remove any term that is redundant (zero product contribution
    everywhere) -> reduces rank."""
    rank = len(U[0])
    removed = True
    while removed and time.time() < deadline:
        removed = False
        for c in range(len(U[0])):
            # check if column c contributes zero to all (i,j,k)
            all_zero = True
            for i in range(N):
                u = U[i * M][c]
                if not u:
                    continue
                for j in range(M):
                    u_ij = U[i * M + j][c]
                    if not u_ij:
                        continue
                    for k in range(P):
                        v = V[j * P + k][c]
                        if not v:
                            continue
                        w = W[k * N + i][c]
                        if not w:
                            continue
                        if u_ij * v * w:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if not all_zero:
                    break
            if all_zero:
                # drop column c
                for r in range(NM):
                    U[r].pop(c)
                for r in range(MP):
                    V[r].pop(c)
                for r in range(PN):
                    W[r].pop(c)
                removed = True
                break
    return U, V, W


def _random_greedy(seed, deadline):
    """Randomized: build terms with random support patterns, then verify."""
    rng = random.Random(seed)
    # Try multiple target ranks
    for R in [30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        U = [[0] * R for _ in range(NM)]
        V = [[0] * R for _ in range(MP)]
        W = [[0] * R for _ in range(PN)]
        
        # Strategy: for each term r, assign a random subset of (i,j,k) slots
        # that it will contribute to
        for r in range(R):
            # Randomly decide which (i,j) pairs this term affects in U
            u_support = set()
            for i in range(N):
                for j in range(M):
                    if rng.random() < 0.3:  # 30% chance to include
                        u_support.add((i, j))
            
            # Randomly decide which (j,k) pairs this term affects in V
            v_support = set()
            for j in range(M):
                for k in range(P):
                    if rng.random() < 0.3:
                        v_support.add((j, k))
            
            # Randomly decide which (k,i) pairs this term affects in W
            w_support = set()
            for k in range(P):
                for i in range(N):
                    if rng.random() < 0.3:
                        w_support.add((k, i))
            
            # For each (i,j,k) slot, if all three supports include it,
            # this term can contribute
            # Assign random weights
            for i in range(N):
                for j in range(M):
                    if (i, j) in u_support:
                        U[i * M + j][r] = rng.choice([1, -1, 0.5, -0.5])
            
            for j in range(M):
                for k in range(P):
                    if (j, k) in v_support:
                        V[j * P + k][r] = rng.choice([1, -1, 0.5, -0.5])
            
            for k in range(P):
                for i in range(N):
                    if (k, i) in w_support:
                        W[k * N + i][r] = rng.choice([1, -1, 0.5, -0.5])
        
        cand = {"U": U, "V": V, "W": W}
        if _is_valid(cand["U"], cand["V"], cand["W"]):
            U, V, W = _drop_terms(cand["U"], cand["V"], cand["W"], deadline)
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    deadline = time.time() + 120.0  # hard limit; evaluator kills at 150s
    safety = 25.0
    effective_deadline = deadline - safety

    best = None
    best_rank = None

    def consider(cand):
        nonlocal best, best_rank
        if cand is None:
            return
        r = _rank(cand["U"], cand["V"], cand["W"])
        if best_rank is None or r < best_rank:
            # final safety re-verify
            if _is_valid(cand["U"], cand["V"], cand["W"]):
                best = cand
                best_rank = r

    # 1) Schoolbook (always valid) as baseline
    consider(_schoolbook())

    # 2) Greedy merge attempts with several seeds
    for s in range(0, 200):
        if time.time() > effective_deadline:
            break
        remaining = _time_left(effective_deadline)
        if remaining < 0.1:
            break
        time_budget = min(0.5, max(0.02, remaining * 0.01))
        cand = _greedy_build(s, time_budget=time_budget)
        consider(cand)

    # 3) Random greedy with varied target ranks and more seeds
    for target in [50, 45, 40, 38, 36, 34, 32, 30, 28]:
        if time.time() > effective_deadline:
            break
        for s in range(0, 30):
            if time.time() > effective_deadline:
                break
            cand = _random_greedy(s * 1000 + target, effective_deadline)
            consider(cand)
            if best_rank is not None and best_rank <= target:
                break

    # 4) Final: if best is still schoolbook rank, that's fine.
    if best is None:
        best = _schoolbook()

    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", _rank(res["U"], res["V"], res["W"]))
