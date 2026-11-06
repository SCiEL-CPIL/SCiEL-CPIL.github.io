import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55

    N = 512
    LIMIT = 512

    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        m = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= LIMIT:
                m |= (1 << (j - 1))
        adj[i] = m

    full = (1 << N) - 1

    def is_clique(S):
        if S == 0:
            return True
        temp = S
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            if (S & ~adj[v]) != 0:
                return False
            temp ^= lsb
        return True

    def greedy_by_degree():
        P = full
        R = []
        remaining = full
        while remaining:
            best = -1
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                d = (P & adj[v]).bit_count()
                if d > best_deg:
                    best_deg = d
                    best = v
                temp ^= lsb
            R.append(best)
            P &= adj[best]
            remaining &= ~ (1 << (best - 1))
            remaining &= P
        return R

    def greedy_by_degree_reverse():
        P = full
        R = []
        remaining = full
        while remaining:
            best = -1
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                d = (P & adj[v]).bit_count()
                if d < best_deg or best == -1:
                    best_deg = d
                    best = v
                temp ^= lsb
            R.append(best)
            P &= adj[best]
            remaining &= ~ (1 << (best - 1))
            remaining &= P
        return R

    def greedy_random(seed_val=42):
        import random
        rng = random.Random(seed_val)
        P = full
        R = []
        remaining = full
        while remaining:
            candidates = []
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                d = (P & adj[v]).bit_count()
                candidates.append((d, v))
                temp ^= lsb
            rng.shuffle(candidates)
            best_v = candidates[0][1]
            best_d = candidates[0][0]
            for d, v in candidates:
                if d > best_d:
                    best_d = d
                    best_v = v
            R.append(best_v)
            P &= adj[best_v]
            remaining &= ~ (1 << (best_v - 1))
            remaining &= P
        return R

    # Try multiple greedy strategies
    best = greedy_by_degree()
    best_size = len(best)
    
    alt1 = greedy_by_degree_reverse()
    if len(alt1) > best_size:
        best = alt1
        best_size = len(alt1)
    
    for s in [1, 2, 3, 7, 13]:
        alt = greedy_random(s)
        if len(alt) > best_size:
            best = alt
            best_size = len(alt)

    best_mask = 0
    for v in best:
        best_mask |= (1 << (v - 1))

    def color_bound(P):
        if P == 0:
            return 0
        colors = 0
        uncolored = P
        while uncolored:
            colors += 1
            indep = 0
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                if (indep & adj[v]) == 0:
                    indep |= lsb
                    uncolored ^= lsb
                temp ^= lsb
        return colors

    import sys
    sys.setrecursionlimit(100000)

    def bk(R, P, X):
        nonlocal best, best_size, best_mask
        if time.time() > deadline:
            return
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best = R[:]
                m = 0
                for v in R:
                    m |= (1 << (v - 1))
                best_mask = m
            return
        if len(R) + P.bit_count() <= best_size:
            return
        cb = color_bound(P)
        if len(R) + cb <= best_size:
            return
        PX = P | X
        if PX == 0:
            return
        best_ov = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length()
            ov = (P & adj[u]).bit_count()
            if ov > best_ov:
                best_ov = ov
                pivot = u
            temp ^= lsb
        if pivot == -1:
            return
        cand = P & ~adj[pivot]
        temp = cand
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            temp ^= lsb
            bk(R + [v], P & adj[v], X & adj[v])
            P &= ~lsb
            X |= lsb
            if time.time() > deadline:
                return

    bk([], full, 0)

    # Enhanced local search polish
    if time.time() < deadline - 2:
        cur = best_mask
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try single add
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if cur & bit:
                    continue
                if (cur & ~adj[v]) == 0:
                    cur |= bit
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap: remove one, add one
            for v in range(1, N + 1):
                bit_v = 1 << (v - 1)
                if cur & bit_v:
                    continue
                conflicts = cur & ~adj[v]
                if conflicts == 0:
                    cur |= bit_v
                    improved = True
                    break
                temp = conflicts
                found = False
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length()
                    newcur = (cur & ~lsb) | bit_v
                    if is_clique(newcur):
                        cur = newcur
                        improved = True
                        found = True
                        break
                    temp ^= lsb
                if found:
                    break
            if improved:
                continue
            # Try 2-swap: remove two, add one (net -1, but may enable future adds)
            # Actually try: remove one, add two
            for v1 in range(1, N + 1):
                bit_v1 = 1 << (v1 - 1)
                if cur & bit_v1:
                    continue
                conflicts1 = cur & ~adj[v1]
                if conflicts1 == 0:
                    continue  # already handled by add
                for v2 in range(v1 + 1, N + 1):
                    bit_v2 = 1 << (v2 - 1)
                    if cur & bit_v2:
                        continue
                    conflicts2 = cur & ~adj[v2]
                    # Need v1 and v2 to be compatible with each other
                    if not (adj[v1] & bit_v2):
                        continue
                    # Find a set of vertices to remove that conflicts with both
                    # Try removing one vertex that conflicts with at least one
                    remove_candidates = conflicts1 | conflicts2
                    temp = remove_candidates
                    found = False
                    while temp and time.time() < deadline - 2:
                        lsb = temp & -temp
                        u = lsb.bit_length()
                        newcur = (cur & ~lsb) | bit_v1 | bit_v2
                        if is_clique(newcur):
                            cur = newcur
                            improved = True
                            found = True
                            break
                        temp ^= lsb
                    if found:
                        break
                if found:
                    break
            if improved:
                continue
            # Try 2-swap: remove two, add two
            for v1 in range(1, N + 1):
                bit_v1 = 1 << (v1 - 1)
                if cur & bit_v1:
                    continue
                for v2 in range(v1 + 1, N + 1):
                    bit_v2 = 1 << (v2 - 1)
                    if cur & bit_v2:
                        continue
                    if not (adj[v1] & bit_v2):
                        continue
                    conflicts = (cur & ~adj[v1]) | (cur & ~adj[v2])
                    if conflicts == 0:
                        continue
                    # Try removing two vertices
                    temp1 = conflicts
                    found = False
                    while temp1 and time.time() < deadline - 2:
                        lsb1 = temp1 & -temp1
                        u1 = lsb1.bit_length()
                        remaining_conf = (cur & ~adj[v1] & ~lsb1) | (cur & ~adj[v2] & ~lsb1)
                        temp2 = remaining_conf
                        while temp2 and time.time() < deadline - 2:
                            lsb2 = temp2 & -temp2
                            u2 = lsb2.bit_length()
                            newcur = (cur & ~lsb1 & ~lsb2) | bit_v1 | bit_v2
                            if is_clique(newcur):
                                cur = newcur
                                improved = True
                                found = True
                                break
                            temp2 ^= lsb2
                        if found:
                            break
                        temp1 ^= lsb1
                    if found:
                        break
                if found:
                    break

        if cur.bit_count() > best_size:
            best_size = cur.bit_count()
            best = [v for v in range(1, N + 1) if cur & (1 << (v - 1))]
            best_mask = cur

    return best
