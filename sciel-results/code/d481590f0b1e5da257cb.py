import time
import random
import math

def _distinct4(a, b, c, d):
    cnt = 1
    if b != a:
        cnt += 1
    if c != a and c != b:
        cnt += 1
    if d != a and d != b and d != c:
        cnt += 1
    return cnt

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = 128
    # precompute 4-APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    m = len(aps)
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    # seed greedy fallback
    def seed_greedy():
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    s = {color, colors[x - 1 - d], colors[x - 1 - 2 * d], colors[x - 1 - 3 * d]}
                    if len(s) < 3:
                        ok = False
                        break
                if ok:
                    for d in range(1, (x - 1) // 2 + 1):
                        if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                            ok = False
                            break
                if ok:
                    colors.append(color)
                    break
        return colors
    
    seed_colors = seed_greedy()
    best = seed_colors
    best_k = len(set(seed_colors))
    
    def try_k(k, time_budget, seed):
        rng = random.Random(seed)
        end = time.time() + time_budget
        best_local = None
        best_viol = None
        while time.time() < end:
            c = [rng.randrange(k) for _ in range(n)]
            ap_dist = [_distinct4(c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]]) for ap in aps]
            viol = sum(1 for d in ap_dist if d < 3)
            if viol == 0:
                return c
            if best_viol is None or viol < best_viol:
                best_viol = viol
                best_local = c[:]
            # SA
            cur = c[:]
            cur_dist = ap_dist[:]
            cur_viol = viol
            temp = 10.0
            step = 0
            while time.time() < end:
                step += 1
                pos = rng.randrange(n)
                old_col = cur[pos]
                new_col = rng.randrange(k)
                if new_col == old_col:
                    continue
                delta = 0
                for ap_idx in pos_aps[pos]:
                    ap = aps[ap_idx]
                    old_d = cur_dist[ap_idx]
                    # new distinct
                    a, b, cc, dd = ap
                    na = new_col if a == pos else cur[a]
                    nb = new_col if b == pos else cur[b]
                    nc = new_col if cc == pos else cur[cc]
                    nd = new_col if dd == pos else cur[dd]
                    new_d = _distinct4(na, nb, nc, nd)
                    if old_d < 3 and new_d >= 3:
                        delta -= 1
                    elif old_d >= 3 and new_d < 3:
                        delta += 1
                    cur_dist[ap_idx] = new_d
                new_viol = cur_viol + delta
                if new_viol < cur_viol or (new_viol > cur_viol and rng.random() < math.exp((cur_viol - new_viol) / temp)):
                    cur[pos] = new_col
                    cur_viol = new_viol
                    if cur_viol == 0:
                        return cur
                    if best_viol is None or cur_viol < best_viol:
                        best_viol = cur_viol
                        best_local = cur[:]
                else:
                    # revert
                    for ap_idx in pos_aps[pos]:
                        ap = aps[ap_idx]
                        a, b, cc, dd = ap
                        na = old_col if a == pos else cur[a]
                        nb = old_col if b == pos else cur[b]
                        nc = old_col if cc == pos else cur[cc]
                        nd = old_col if dd == pos else cur[dd]
                        cur_dist[ap_idx] = _distinct4(na, nb, nc, nd)
                if step % 1000 == 0:
                    temp = max(0.01, temp * 0.99)
        if best_viol == 0:
            return best_local
        return None
    
    # Try k from 3 up
    k = 3
    while time.time() < deadline and k <= best_k:
        budget = min(15.0, deadline - time.time())
        if budget < 2.0:
            break
        res = try_k(k, budget, seed=1000 + k)
        if res is not None:
            # valid k-coloring found
            best = res
            best_k = k
            break
        k += 1
    
    # Verify best is valid
    def is_valid(cols):
        for ap in aps:
            if _distinct4(cols[ap[0]], cols[ap[1]], cols[ap[2]], cols[ap[3]]) < 3:
                return False
        return True
    
    if not is_valid(best):
        best = seed_colors
    return best
