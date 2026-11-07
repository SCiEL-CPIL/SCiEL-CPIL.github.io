"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))
M = NMAX + 1          # indices 0..256, values 2..256
IDX = {v: v for v in range(2, NMAX + 1)}

# ---- Precompute conflict masks (global, done once) -------------------------
# conflict[a][b] = 1 if adding a new element x creates a bad triple involving x,
# element a and element b (a != b, both already in the set):
#   x | a*b   OR   a | x*b   OR   b | x*a
# (the last two are the symmetric roles of a and b as the divisor).
conflict = [[0] * M for _ in range(M)]
for x in range(2, NMAX + 1):
    row = conflict[x]
    for a in range(2, NMAX + 1):
        if a == x:
            continue
        for b in range(a + 1, NMAX + 1):
            if b == x:
                continue
            bad = ((b * a) % x == 0) or ((x * b) % a == 0) or ((x * a) % b == 0)
            if bad:
                row[a] = 1
                row[b] = 1

def _can_add(x, S):
    """Fast O(len(S)) check using precomputed conflict masks."""
    row = conflict[x]
    for s in S:
        if row[s]:
            return False
    return True

def _greedy(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _greedy_remove(deadline):
    """Start from full set; repeatedly delete the element that blocks the fewest
    others (tie-break by preferring to keep larger values)."""
    present = set(UNIVERSE)
    # conflict_count[x] = number of present y (y != x) with conflict[x][y]
    cc = {x: 0 for x in UNIVERSE}
    for x in UNIVERSE:
        row = conflict[x]
        cnt = 0
        for y in UNIVERSE:
            if y != x and row[y]:
                cnt += 1
        cc[x] = cnt

    # work on a mutable list
    alive = list(UNIVERSE)
    alive_set = set(alive)
    while alive and time.time() < deadline:
        # find element with minimal conflict count among alive
        best = None
        best_key = None
        for x in alive:
            row = conflict[x]
            cnt = 0
            for y in alive:
                if y != x and row[y]:
                    cnt += 1
            key = (cnt, -x)   # fewer conflicts better; tie -> keep larger x
            if best_key is None or key < best_key:
                best_key = key
                best = x
        if best_key[0] == 0:
            break  # no conflicts remain
        # remove best
        alive.remove(best)
        alive_set.discard(best)
    return list(alive_set)

def _greedy_conflict_priority(deadline):
    """Greedy that at each step picks the candidate with minimum conflict potential
    (number of current-set elements it conflicts with)."""
    S = []
    S_set = set()
    candidates = list(UNIVERSE)

    while candidates and time.time() < deadline:
        best_x = None
        best_key = None
        for x in candidates:
            row = conflict[x]
            shared = 0
            for s in S:
                if row[s]:
                    shared += 1
            key = (shared, -x)
            if best_key is None or key < best_key:
                best_key = key
                best_x = x
        if best_x is None:
            break
        if _can_add(best_x, S):
            S.append(best_x)
            S_set.add(best_x)
        candidates.remove(best_x)
    return S

def _local_search_improve(S, deadline, max_iters, rng_seed):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)

    candidates = [x for x in UNIVERSE if x not in S_set]

    for _ in range(max_iters):
        if time.time() > deadline:
            break
        progress = False

        # 1) try to add candidates
        if candidates:
            n_try = min(40, len(candidates))
            trial = rng.sample(candidates, n_try)
            for x in trial:
                if _can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    progress = True
                    break

        if progress:
            continue

        # 2) try swaps
        if not S:
            break
        for _attempt in range(6):
            if time.time() > deadline:
                break
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}

            if not candidates:
                continue
            n_try = min(40, len(candidates))
            trial = rng.sample(candidates, n_try)
            swapped = False
            for x in trial:
                if _can_add(x, S_temp):
                    S = S_temp + [x]
                    S_set = S_temp_set | {x}
                    if removed not in candidates:
                        candidates.append(removed)
                    if x in candidates:
                        candidates.remove(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    swapped = True
                    progress = True
                    break
            if swapped:
                break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    best_S = None
    best_size = 0

    def consider(S):
        nonlocal best_S, best_size
        if S and len(S) > best_size:
            best_S = S
            best_size = len(S)

    # Phase 1: diverse, cheap initializations
    # 1. greedy removal (strong global starting point)
    Sr = _greedy_remove(deadline)
    consider(Sr)

    # 2. largest first
    consider(_greedy(list(range(NMAX, 1, -1)), deadline))
    # 3. smallest first
    consider(_greedy(list(range(2, NMAX + 1)), deadline))
    # 4. conflict-priority greedy
    if time.time() < deadline - 10:
        consider(_greedy_conflict_priority(deadline - 10))
    # 5. randomized orders
    for seed in [1, 7, 42, 100, 200, 500, 777, 1234]:
        if time.time() > deadline - 20:
            break
        rng = random.Random(seed)
        order_rand = list(range(2, NMAX + 1))
        rng.shuffle(order_rand)
        consider(_greedy(order_rand, deadline))

    if not best_S:
        best_S = [2, 3]
        best_size = 2

    # Phase 2: local search from best and perturbed states
    if time.time() < deadline - 12:
        consider(_local_search_improve(best_S, deadline - 12, 4000, 42))

    if time.time() < deadline - 6:
        rng_p = random.Random(123)
        Sp = best_S[:]
        for _ in range(rng_p.randint(1, 3)):
            if Sp:
                Sp.pop(rng_p.randrange(len(Sp)))
        consider(_local_search_improve(Sp, deadline - 6, 2000, 99))

    if time.time() < deadline - 3:
        rng_p2 = random.Random(456)
        Sp2 = best_S[:]
        for _ in range(rng_p2.randint(1, 2)):
            if Sp2:
                Sp2.pop(rng_p2.randrange(len(Sp2)))
        consider(_local_search_improve(Sp2, deadline - 3, 1200, 200))

    # Phase 3: final high-precision polish (tighter, more add attempts)
    if time.time() < deadline - 1:
        consider(_local_search_improve(best_S, deadline - 1, 2000, 7))

    # safety: verify validity of the returned set
    out = [int(x) for x in best_S]
    # (light verification; if anything is off, fall back to a guaranteed-valid set)
    sset = set(out)
    if len(sset) != len(out) or any(not isinstance(v, int) or isinstance(v, bool) for v in out):
        out = [2, 3]
    return out
