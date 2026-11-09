import time
import math
import random
import numpy as np

BASE = 51975

def _divisors(n):
    ds = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
    return sorted(ds)

MODULI = [m for m in _divisors(BASE) if m > 1]

def score_classes(classes):
    covered = bytearray(BASE)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return BASE - sum(covered)

def construct():
    deadline = time.time() + 140.0
    rng = random.Random(98765)

    def build_cnt(used):
        cnt = np.zeros(BASE, dtype=np.int32)
        for m, r in used.items():
            idx = np.arange(r, BASE, m)
            cnt[idx] += 1
        return cnt

    def covered_of(cnt):
        return int(np.count_nonzero(cnt))

    def greedy(order):
        cnt = np.zeros(BASE, dtype=np.int32)
        covered = 0
        used = {}
        for m in order:
            best_r, best_gain = 0, -1
            for r in range(m):
                idx = np.arange(r, BASE, m)
                g = int(np.count_nonzero(cnt[idx] == 0))
                if g > best_gain:
                    best_gain, best_r = g, r
            idx = np.arange(best_r, BASE, m)
            covered += int(np.count_nonzero(cnt[idx] == 0))
            cnt[idx] += 1
            used[m] = best_r
        return used, covered

    orders = [
        sorted(MODULI, reverse=True),
        sorted(MODULI),
        sorted(MODULI, key=lambda m: -BASE // m),
    ]
    for _ in range(6):
        o = MODULI[:]; rng.shuffle(o); orders.append(o)

    best_score = BASE
    best_classes = None
    starts = []
    for o in orders:
        used, covered = greedy(o)
        starts.append((used, covered))
        sc = BASE - covered
        if sc < best_score:
            best_score, best_classes = sc, [[m, used[m]] for m in used]

    def local_search(used0, covered0, max_iter):
        cnt = build_cnt(used0)
        used = dict(used0)
        covered = covered_of(cnt)
        best_sc = BASE - covered
        best_used = dict(used)
        # Precompute index arrays for each modulus
        idx_cache = {m: [np.arange(r, BASE, m) for r in range(m)] for m in MODULI}
        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            roll = rng.random()
            if used and roll < 0.6:
                m = rng.choice(list(used.keys()))
                cur = used[m]
                # Vectorized: remove current, compute uncovered for all residues, pick best
                cnt2 = cnt.copy()
                idx_cur = idx_cache[m][cur]
                cnt2[idx_cur] -= 1
                # For each residue r, count positions where cnt2 == 0
                # Build a 2D view: shape (m, len(idx_r)) but lengths vary by m
                # Instead: for each r, idx = idx_cache[m][r], count cnt2[idx]==0
                # Vectorize over r using broadcasting when possible
                # Since all idx have the same length for fixed m? No, len(range(r, BASE, m)) varies.
                # Use a loop but with precomputed arrays
                best_cov = -1
                best_r = cur
                for r in range(m):
                    idx = idx_cache[m][r]
                    c = int(np.count_nonzero(cnt2[idx] == 0))
                    if c > best_cov:
                        best_cov, best_r = c, r
                # Apply best
                cnt = cnt2
                idx = idx_cache[m][best_r]
                cnt[idx] += 1
                covered = int(np.count_nonzero(cnt))
                used[m] = best_r
            elif roll < 0.85:
                unused = [m for m in MODULI if m not in used]
                if not unused:
                    continue
                m = rng.choice(unused)
                best_r, best_cov = 0, covered
                for r in range(m):
                    idx = idx_cache[m][r]
                    c = covered + int(np.count_nonzero(cnt[idx] == 0))
                    if c > best_cov:
                        best_cov, best_r = c, r
                idx = idx_cache[m][best_r]
                cnt[idx] += 1
                covered = int(np.count_nonzero(cnt))
                used[m] = best_r
            else:
                if not used:
                    continue
                m = rng.choice(list(used.keys()))
                r = used[m]
                idx = idx_cache[m][r]
                cnt[idx] -= 1
                del used[m]
                covered = int(np.count_nonzero(cnt))
            sc = BASE - covered
            if sc < best_sc:
                best_sc, best_used = sc, dict(used)
        return best_used, best_sc

    for used0, covered0 in starts[:5]:
        if time.time() > deadline - 20:
            break
        used, sc = local_search(used0, covered0, max_iter=800)
        if sc < best_score:
            best_score, best_classes = sc, [[m, used[m]] for m in used]

    # ILP polish if time remains
    try:
        if best_classes is not None and time.time() < deadline - 40:
            from scipy.optimize import milp, LinearConstraint, Bounds
            used_set = set(m for m, _ in best_classes)
            # Include all moduli in candidate set for stronger polish
            cand = sorted(MODULI)
            xcols = [m for m in cand]
            n_x = sum(xcols)
            n_c = BASE
            n_var = n_x + n_c
            # objective: maximize sum c_s -> minimize -sum c_s
            c_obj = np.zeros(n_var)
            c_obj[n_x:] = -1.0
            # Build constraint matrix: for each s, c_s - sum_{m: s in class(m, s%m)} x_{m, s%m} <= 0
            # Use sparse representation via scipy
            from scipy.sparse import lil_matrix
            A = lil_matrix((BASE, n_var))
            offsets = [0]
            for m in cand:
                offsets.append(offsets[-1] + m)
            for s in range(BASE):
                A[s, n_x + s] = 1.0
                for mi, m in enumerate(cand):
                    r = s % m
                    A[s, offsets[mi] + r] = -1.0
            A = A.tocsr()
            lb = np.full(n_var, 0.0)
            ub = np.ones(n_var)
            integrality = np.ones(n_var)
            cons = LinearConstraint(A, -np.inf, np.zeros(BASE))
            time_left = deadline - time.time() - 15
            if time_left > 5:
                res = milp(c_obj, constraints=[cons], integrality=integrality, bounds=Bounds(lb, ub),
                           options={"time_limit": time_left, "presolve": True})
                if res.x is not None:
                    new_used = {}
                    off = 0
                    for m in cand:
                        xr = res.x[off:off+m]
                        if xr.max() > 0.5:
                            new_used[m] = int(np.argmax(xr))
                        off += m
                    sc = score_classes([[m, r] for m, r in new_used.items()])
                    if sc < best_score:
                        best_score, best_classes = sc, [[m, new_used[m]] for m in new_used]
    except Exception:
        pass

    if best_classes is None:
        best_classes = [[m, 0] for m in MODULI]
    return best_classes
