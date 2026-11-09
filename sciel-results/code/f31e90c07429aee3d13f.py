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
        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            roll = rng.random()
            if used and roll < 0.6:
                m = rng.choice(list(used.keys()))
                cur = used[m]
                # evaluate all residues
                best_r, best_cov = cur, covered
                base_cnt = cnt.copy()
                for r in range(m):
                    idx = np.arange(r, BASE, m)
                    c = int(np.count_nonzero(base_cnt[idx] == 0)) + covered - int(np.count_nonzero(cnt[cur::m] == 0))
                    # simpler: compute gain relative to removing cur
                    # recompute properly:
                # do it cleanly:
                cnt2 = cnt.copy()
                idx_cur = np.arange(cur, BASE, m)
                cnt2[idx_cur] -= 1
                for r in range(m):
                    idx = np.arange(r, BASE, m)
                    c = int(np.count_nonzero(cnt2[idx] == 0))
                    if c > best_cov:
                        best_cov, best_r = c, r
                # apply best
                cnt = cnt2.copy()
                idx = np.arange(best_r, BASE, m)
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
                    idx = np.arange(r, BASE, m)
                    c = covered + int(np.count_nonzero(cnt[idx] == 0))
                    if c > best_cov:
                        best_cov, best_r = c, r
                idx = np.arange(best_r, BASE, m)
                cnt[idx] += 1
                covered = int(np.count_nonzero(cnt))
                used[m] = best_r
            else:
                if not used:
                    continue
                m = rng.choice(list(used.keys()))
                r = used[m]
                idx = np.arange(r, BASE, m)
                cnt[idx] -= 1
                del used[m]
                covered = int(np.count_nonzero(cnt))
            sc = BASE - covered
            if sc < best_sc:
                best_sc, best_used = sc, dict(used)
        return best_used, best_sc

    for used0, covered0 in starts[:3]:
        if time.time() > deadline - 15:
            break
        used, sc = local_search(used0, covered0, max_iter=400)
        if sc < best_score:
            best_score, best_classes = sc, [[m, used[m]] for m in used]

    # optional ILP polish (small) if time remains
    try:
        if best_classes is not None and time.time() < deadline - 35:
            from scipy.optimize import milp, LinearConstraint, Bounds
            used_set = [m for m, _ in best_classes]
            cand = used_set + [m for m in MODULI if m not in used_set][:5]
            cand = sorted(set(cand))
            # variables: for each m in cand, one residue chosen (<=1)
            # x_{m,r}
            # c_s <= sum x covering s
            # max sum c_s
            n_m = len(cand)
            xcols = []
            for m in cand:
                xcols.append(m)
            n_x = sum(xcols)
            n_c = BASE
            n_var = n_x + n_c
            # objective: maximize sum c_s -> minimize -sum c_s
            c = np.zeros(n_var)
            c[n_x:] = -1.0
            # constraints: for each s, c_s - sum_{m,r: s in class} x_{m,r} <= 0
            A = []
            rows = []
            for s in range(BASE):
                row = np.zeros(n_var)
                row[n_x + s] = 1.0
                # find x covering s
                for mi, m in enumerate(cand):
                    # r = s mod m
                    r = s % m
                    # index of x_{m,r}
                    # offset
                    off = sum(xcols[:mi])
                    idx = off + r
                    row[idx] = -1.0
                rows.append(row)
            A = np.vstack(rows) if rows else np.zeros((0, n_var))
            lb = np.full(n_var, 0.0)
            ub = np.ones(n_var)
            integrality = np.ones(n_var)
            cons = LinearConstraint(A, -np.inf, np.zeros(len(rows)))
            res = milp(c, constraints=[cons], integrality=integrality, bounds=Bounds(lb, ub),
                       options={"time_limit": max(1.0, deadline - time.time() - 20)})
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
