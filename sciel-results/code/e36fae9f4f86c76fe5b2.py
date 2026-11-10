import math
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 112.0  # safety margin before 120s kill
    random.seed(20240517)

    MODS = list(range(20, 201))          # 20..200 inclusive
    n = len(MODS)
    idx = {m: i for i, m in enumerate(MODS)}

    # Precompute gcd matrix
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(MODS[i], MODS[j])
            G[i][j] = g
            G[j][i] = g

    def is_ok(mi, r, sel):
        """sel: list of (mi, r). True if (mi, r) compatible with all in sel."""
        for (mj, rj) in sel:
            g = G[mi][mj]
            if (r % g) == (rj % g):
                return False
        return True

    def score_of(sel):
        return sum(1.0 / MODS[mi] for mi, _ in sel)

    def greedy(randomize):
        sel = []
        selset = set()
        order = MODS[:]
        if randomize:
            random.shuffle(order)
        for m in order:
            mi = idx[m]
            rs = list(range(m))
            if randomize:
                random.shuffle(rs)
            for r in rs:
                if is_ok(mi, r, sel):
                    sel.append((mi, r))
                    selset.add(m)
                    break
        return sel, selset

    def add_pass(sel, selset, order, dl):
        """Monotonic: a modulus that fails now never succeeds after more adds."""
        for m in order:
            if time.time() > dl:
                break
            if m in selset:
                continue
            mi = idx[m]
            for r in range(m):
                if is_ok(mi, r, sel):
                    sel.append((mi, r))
                    selset.add(m)
                    break

    def swap_pass(sel, selset, dl):
        """Replace large selected moduli with smaller unselected ones (score gain)."""
        improved = False
        # focus on highest-gain exchanges
        rm_cands = sorted(selset, reverse=True)[:40]
        add_cands = [m for m in MODS if m not in selset][:40]
        for m_rm in rm_cands:
            if time.time() > dl:
                break
            mi_rm = idx[m_rm]
            tmp = [x for x in sel if x[0] != mi_rm]
            best = None
            for m_add in add_cands:
                if m_add >= m_rm:
                    continue
                mi_add = idx[m_add]
                for r in range(m_add):
                    if is_ok(mi_add, r, tmp):
                        best = (mi_add, r)
                        break
                if best is not None:
                    break
            if best is not None:
                mi_add, r = best
                sel = tmp + [(mi_add, r)]
                selset = (selset - {m_rm}) | {MODS[mi_add]}
                improved = True
        return improved, sel, selset

    def local_search(sel, selset, dl):
        # fill with additions
        while time.time() < dl:
            before = len(sel)
            add_pass(sel, selset, MODS, dl)
            if len(sel) == before:
                break
        # swap + re-add until stable
        improved = True
        while improved and time.time() < dl:
            improved, sel, selset = swap_pass(sel, selset, dl)
            if improved:
                add_pass(sel, selset, MODS, dl)
        return sel, selset

    best_sel = None
    best_val = -1.0

    # deterministic strong start
    sel, selset = greedy(False)
    sel, selset = local_search(sel, selset, deadline - 6)
    v = score_of(sel)
    if v > best_val:
        best_val, best_sel = v, sel

    # iterated randomized local search
    while time.time() < deadline - 8:
        sub_dl = min(deadline - 4, time.time() + 9.0)
        sel, selset = greedy(True)
        sel, selset = local_search(sel, selset, sub_dl)
        v = score_of(sel)
        if v > best_val + 1e-12:
            best_val, best_sel = v, sel

    if not best_sel:
        # minimal valid fallback
        best_sel = [(idx[20], 0), (idx[21], 1)]

    return [[MODS[mi], r] for mi, r in best_sel]
