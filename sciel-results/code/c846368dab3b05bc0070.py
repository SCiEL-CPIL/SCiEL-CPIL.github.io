"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0   # hard safety margin before the 150s wall-clock kill

    n = 16
    triples = list(combinations(range(n), 3))
    M = len(triples)                      # 560
    idx = {t: i for i, t in enumerate(triples)}
    rev = triples

    # For each point, the list of triple-indices containing it.
    point_triples = [[] for _ in range(n)]
    for i, (a, b, c) in enumerate(triples):
        point_triples[a].append(i)
        point_triples[b].append(i)
        point_triples[c].append(i)

    # Precompute every K(2,2,2) configuration as an 8-bit mask over triple-indices.
    # A configuration = 3 disjoint pairs; the 8 transversal triples are all
    # (x in P1, y in P2, z in P3). We need a state with no config whose 8 bits
    # are all set.
    configs = []
    seen = set()
    pts = range(n)
    for a, b in combinations(pts, 2):
        for c, d in combinations(pts, 2):
            if a in (c, d) or b in (c, d):
                continue
            for e, f in combinations(pts, 2):
                if e in (a, b, c, d) or f in (a, b, c, d):
                    continue
                # 8 transversal triples
                mask = 0
                for x in (a, b):
                    for y in (c, d):
                        for z in (e, f):
                            # order the triple for the canonical index
                            t = tuple(sorted((x, y, z)))
                            mask |= (1 << idx[t])
                if mask not in seen:
                    seen.add(mask)
                    configs.append(mask)
    C = len(configs)

    # ---- validity: no config has all 8 bits set ----
    def valid(state):
        for m in configs:
            if (state & m) == m:
                return False
        return True

    # ---- local search ----
    def local_search(state):
        cur = state
        curv = cur.bit_count()
        best = cur
        bestv = curv
        while True:
            if time.time() > DEADLINE:
                break
            improved = False

            # 1) try to ADD the best triple (max new degree, no violation)
            best_add = -1
            best_add_deg = -1
            for i in range(M):
                if (cur >> i) & 1:
                    continue
                if valid(cur | (1 << i)):
                    deg = 0
                    for j in point_triples[rev[i][0]]:
                        deg += (cur >> j) & 1
                    for j in point_triples[rev[i][1]]:
                        deg += (cur >> j) & 1
                    for j in point_triples[rev[i][2]]:
                        deg += (cur >> j) & 1
                    if deg > best_add_deg:
                        best_add_deg = deg
                        best_add = i
            if best_add >= 0:
                cur |= (1 << best_add)
                curv += 1
                improved = True
                if curv > bestv:
                    bestv = curv
                    best = cur
                continue

            # 2) try to REMOVE the worst triple (min degree)
            if curv > 0:
                worst = -1
                worst_deg = 1 << 30
                for i in range(M):
                    if not ((cur >> i) & 1):
                        continue
                    deg = 0
                    for j in point_triples[rev[i][0]]:
                        deg += (cur >> j) & 1
                    for j in point_triples[rev[i][1]]:
                        deg += (cur >> j) & 1
                    for j in point_triples[rev[i][2]]:
                        deg += (cur >> j) & 1
                    if deg < worst_deg:
                        worst_deg = deg
                        worst = i
                if worst >= 0:
                    cur &= ~(1 << worst)
                    curv -= 1
                    improved = True
                    if curv > bestv:
                        bestv = curv
                        best = cur

            # 3) 1-flip moves (toggle a single bit to improve)
            for i in range(M):
                if (cur >> i) & 1:
                    cand = cur & ~(1 << i)
                    cv = curv - 1
                    if cv > curv:
                        pass
                    if cv > curv:
                        continue
                    # only consider if it improves (removals already handled)
                    continue
                else:
                    cand = cur | (1 << i)
                    if valid(cand):
                        cv = curv + 1
                        if cv > curv:
                            cur = cand
                            curv = cv
                            improved = True
                            if curv > bestv:
                                bestv = curv
                                best = cur
                            break

            # 4) 2-flip moves: remove one present, add one absent
            if curv > 0:
                present = [i for i in range(M) if (cur >> i) & 1]
                absent = [i for i in range(M) if not ((cur >> i) & 1)]
                found = False
                for ri in present:
                    base = cur & ~(1 << ri)
                    for ai in absent:
                        cand = base | (1 << ai)
                        if valid(cand):
                            # net 0 change in count, but may enable future adds;
                            # accept only if it lets an immediate add happen
                            # (keep it simple: accept the flip to escape plateau)
                            cur = cand
                            improved = True
                            found = True
                            break
                    if found:
                        break

            if not improved:
                break
        return best, bestv

    # ---- greedy initial build (fast, deterministic) ----
    state = 0
    for i in range(M):
        if valid(state | (1 << i)):
            state |= (1 << i)
    best_state = state
    best_v = state.bit_count()

    # ---- local search from greedy ----
    if time.time() < DEADLINE:
        s, v = local_search(state)
        if v > best_v:
            best_state, best_v = s, v

    # ---- randomized restarts to escape local optima ----
    rng = random.Random(12345)
    while time.time() < DEADLINE:
        # random-ish seed: random subset, then greedy fill
        seed_state = 0
        for i in range(M):
            if rng.random() < 0.35:
                seed_state |= (1 << i)
        # repair: drop anything that creates a violation
        for m in configs:
            if (seed_state & m) == m:
                # remove one random member of this config
                bits = [i for i in range(M) if ((seed_state >> i) & 1) and ((m >> i) & 1)]
                if bits:
                    seed_state &= ~(1 << rng.choice(bits))
        # greedy fill
        for i in range(M):
            if valid(seed_state | (1 << i)):
                seed_state |= (1 << i)
        if time.time() >= DEADLINE:
            break
        s, v = local_search(seed_state)
        if v > best_v:
            best_state, best_v = s, v

    # ---- final result ----
    result = [list(rev[i]) for i in range(M) if (best_state >> i) & 1]
    return result
# EVOLVE-BLOCK-END
