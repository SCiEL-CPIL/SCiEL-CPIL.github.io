"""Structural redesign: model as Max-Weight Independent Set on the
lcm-conflict graph over {2..256}; solve with a strong local search
(single-flip + pair-swap + random restarts) under an explicit wall-clock
deadline guard.  Weight of vertex a is 1/a (exact rational via Fraction)."""

def construct():
    import time
    from fractions import Fraction
    from math import gcd, lcm
    import random

    DEADLINE = time.time() + 75.0          # safety margin vs 90 s kill
    N = 256
    # vertices are integers 2..256
    vals = list(range(2, N + 1))
    idx = {v: i for i, v in enumerate(vals)}
    m = len(vals)

    # conflict: edge (a,b) iff lcm(a,b) <= 256  (we must AVOID these pairs)
    # Build adjacency bitmasks for speed.
    adj = [0] * m
    for i in range(m):
        a = vals[i]
        for j in range(i + 1, m):
            b = vals[j]
            # lcm = a*b//gcd
            g = gcd(a, b)
            l = (a // g) * b
            if l <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # weights as Fraction for exactness; also float for fast comparison
    w_frac = [Fraction(1, vals[i]) for i in range(m)]
    w_float = [1.0 / vals[i] for i in range(m)]

    def solution_value(mask):
        s = Fraction(0, 1)
        for i in range(m):
            if (mask >> i) & 1:
                s += w_frac[i]
        return s

    def is_valid(mask):
        # no two selected vertices adjacent
        for i in range(m):
            if (mask >> i) & 1:
                if adj[i] & mask:
                    return False
        return True

    def greedy_init(order):
        mask = 0
        for i in order:
            if not (adj[i] & mask):
                mask |= (1 << i)
        return mask

    # ---- initial solutions ----
    # 1) greedy by descending weight (smallest a first)
    by_w = sorted(range(m), key=lambda i: -w_float[i])
    # 2) greedy by ascending degree (most isolated first)
    deg = [bin(adj[i]).count("1") for i in range(m)]
    by_deg = sorted(range(m), key=lambda i: (deg[i], -w_float[i]))
    # 3) greedy by weight/degree ratio
    by_ratio = sorted(range(m), key=lambda i: -(w_float[i] / (deg[i] + 1)))

    best_mask = 0
    best_val = Fraction(0, 1)
    best_float = 0.0
    for order in (by_w, by_deg, by_ratio):
        mk = greedy_init(order)
        v = solution_value(mk)
        vf = sum(w_float[i] for i in range(m) if (mk >> i) & 1)
        if vf > best_float:
            best_float = vf
            best_val = v
            best_mask = mk

    rng = random.Random(12345)

    def try_improve(mask):
        nonlocal best_mask, best_val, best_float
        improved = True
        while improved:
            improved = False
            # single add
            for i in range(m):
                if not ((mask >> i) & 1) and not (adj[i] & mask):
                    nm = mask | (1 << i)
                    nv = best_float + w_float[i]
                    if nv > best_float + 1e-15:
                        mask = nm
                        best_float = nv
                        improved = True
            # single remove + re-add (escape)
            for i in range(m):
                if (mask >> i) & 1:
                    nm = mask & ~(1 << i)
                    # after removing i, we may be able to add others
                    for j in range(m):
                        if not ((nm >> j) & 1) and not (adj[j] & nm):
                            nm2 = nm | (1 << j)
                            nv = best_float - w_float[i] + w_float[j]
                            if nv > best_float + 1e-15:
                                mask = nm2
                                best_float = nv
                                improved = True
                                break
            # pair swap: remove i, add j
            for i in range(m):
                if not ((mask >> i) & 1):
                    continue
                for j in range(m):
                    if (mask >> j) & 1 or j == i:
                        continue
                    if adj[i] & (mask & ~(1 << i)) or adj[j] & (mask & ~(1 << i)):
                        continue
                    if w_float[j] <= w_float[i]:
                        continue
                    nm = (mask & ~(1 << i)) | (1 << j)
                    nv = best_float - w_float[i] + w_float[j]
                    if nv > best_float + 1e-15:
                        mask = nm
                        best_float = nv
                        improved = True
        return mask

    # local search on each init
    for order in (by_w, by_deg, by_ratio):
        mk = greedy_init(order)
        bf = sum(w_float[i] for i in range(m) if (mk >> i) & 1)
        # run improvement with local bookkeeping
        local_best = bf
        local_mask = mk
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(m):
                if not ((local_mask >> i) & 1) and not (adj[i] & local_mask):
                    local_mask |= (1 << i)
                    local_best += w_float[i]
                    improved = True
            for i in range(m):
                if (local_mask >> i) & 1:
                    nm = local_mask & ~(1 << i)
                    for j in range(m):
                        if not ((nm >> j) & 1) and not (adj[j] & nm):
                            nm2 = nm | (1 << j)
                            nv = local_best - w_float[i] + w_float[j]
                            if nv > local_best + 1e-15:
                                local_mask = nm2
                                local_best = nv
                                improved = True
                                break
            for i in range(m):
                if not ((local_mask >> i) & 1):
                    continue
                for j in range(m):
                    if (local_mask >> j) & 1 or j == i:
                        continue
                    if adj[i] & (local_mask & ~(1 << i)) or adj[j] & (local_mask & ~(1 << i)):
                        continue
                    if w_float[j] <= w_float[i]:
                        continue
                    nm = (local_mask & ~(1 << i)) | (1 << j)
                    nv = local_best - w_float[i] + w_float[j]
                    if nv > local_best + 1e-15:
                        local_mask = nm
                        local_best = nv
                        improved = True
        if local_best > best_float:
            best_float = local_best
            best_mask = local_mask
            best_val = solution_value(local_mask)

    # random restarts / perturbations within remaining time
    restart = 0
    while time.time() < DEADLINE - 2.0:
        restart += 1
        # start from best, perturb: remove a random subset, re-greedy
        base = best_mask
        if rng.random() < 0.5 and best_mask:
            # drop a few random selected vertices
            sel = [i for i in range(m) if (best_mask >> i) & 1]
            k = min(len(sel), rng.randint(1, 4))
            drop = set(rng.sample(sel, k))
            base = best_mask
            for i in drop:
                base &= ~(1 << i)
        # greedy fill base
        mk = base
        order = sorted(range(m), key=lambda i: -w_float[i])
        for i in order:
            if not ((mk >> i) & 1) and not (adj[i] & mk):
                mk |= (1 << i)
        # quick local improve (single add + swap)
        lf = sum(w_float[i] for i in range(m) if (mk >> i) & 1)
        improved = True
        while improved and time.time() < DEADLINE - 2.0:
            improved = False
            for i in range(m):
                if not ((mk >> i) & 1) and not (adj[i] & mk):
                    mk |= (1 << i)
                    lf += w_float[i]
                    improved = True
            for i in range(m):
                if (mk >> i) & 1:
                    nm = mk & ~(1 << i)
                    for j in range(m):
                        if not ((nm >> j) & 1) and not (adj[j] & nm):
                            nm2 = nm | (1 << j)
                            nv = lf - w_float[i] + w_float[j]
                            if nv > lf + 1e-15:
                                mk = nm2
                                lf = nv
                                improved = True
                                break
            for i in range(m):
                if not ((mk >> i) & 1):
                    continue
                for j in range(m):
                    if (mk >> j) & 1 or j == i:
                        continue
                    if adj[i] & (mk & ~(1 << i)) or adj[j] & (mk & ~(1 << i)):
                        continue
                    if w_float[j] <= w_float[i]:
                        continue
                    nm = (mk & ~(1 << i)) | (1 << j)
                    nv = lf - w_float[i] + w_float[j]
                    if nv > lf + 1e-15:
                        mk = nm
                        lf = nv
                        improved = True
        if lf > best_float:
            best_float = lf
            best_mask = mk
            best_val = solution_value(mk)

    # final safety: ensure validity and nonempty
    if best_mask == 0:
        # fallback: largest single number (fewest conflicts)
        best_mask = 1 << (m - 1)
        best_val = solution_value(best_mask)
    if not is_valid(best_mask):
        # repair by greedy rebuild
        mk = 0
        for i in sorted(range(m), key=lambda i: -w_float[i]):
            if not (adj[i] & mk):
                mk |= (1 << i)
        best_mask = mk
        best_val = solution_value(mk)

    return [vals[i] for i in range(m) if (best_mask >> i) & 1]
