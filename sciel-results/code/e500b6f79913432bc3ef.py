"""Covering 1..L by prime residue classes with reciprocal budget 1 (primes <= 1000).

Structural change vs seed: coverage is tracked with a per-position COUNT array
so add/remove/residue-change are O(MAXL/p) instead of a full O(MAXL) recompute.
This makes each local-search move ~100x cheaper, allowing many more moves.
"""
import time
import random
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _covered_prefix_len(classes, cap=100000):
    if not classes:
        return 0
    covered = bytearray(cap + 1)
    for p, r in classes:
        for x in range(r if r >= 1 else p, cap + 1, p):
            covered[x] = 1
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _recip_sum(classes):
    s = Fraction(0, 1)
    for p, _ in classes:
        s += Fraction(1, p)
    return s

def construct():
    t0 = time.time()
    DEADLINE = 140.0          # safety margin before 150s kill
    MAXL = 100000

    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}

    # ---- count-array coverage model ---------------------------------------
    # counts[i] = number of classes covering i.  covered iff counts[i] > 0.
    # add / remove / change_residue are O(MAXL/p).
    counts = [0] * (MAXL + 1)
    state = {}               # p -> r
    budget = Fraction(1, 1)
    L = 0

    def first_gap():
        i = L + 1
        while i <= MAXL and counts[i] > 0:
            i += 1
        return i             # MAXL+1 if fully covered

    def add(p, r):
        if p in state or budget < recip[p]:
            return False
        state[p] = r
        budget -= recip[p]
        start = r if r >= 1 else p
        for x in range(start, MAXL + 1, p):
            counts[x] += 1
        nonlocal L
        i = L + 1
        while i <= MAXL and counts[i] > 0:
            i += 1
        L = i - 1
        return True

    def remove(p):
        if p not in state:
            return False
        r = state[p]
        del state[p]
        budget += recip[p]
        start = r if r >= 1 else p
        for x in range(start, MAXL + 1, p):
            counts[x] -= 1
        nonlocal L
        L = 0
        for i in range(1, MAXL + 1):
            if counts[i] > 0:
                L = i
            else:
                break

    def change_residue(p, new_r):
        if p not in state or state[p] == new_r:
            return
        old_r = state[p]
        state[p] = new_r
        os_ = old_r if old_r >= 1 else old_r + p
        ns_ = new_r if new_r >= 1 else new_r + p
        for x in range(os_, MAXL + 1, p):
            counts[x] -= 1
        for x in range(ns_, MAXL + 1, p):
            counts[x] += 1
        nonlocal L
        i = L + 1
        while i <= MAXL and counts[i] > 0:
            i += 1
        L = i - 1
        # if L shrank, recompute from start
        if L < 0:
            L = 0
            for i in range(1, MAXL + 1):
                if counts[i] > 0:
                    L = i
                else:
                    break

    def snapshot():
        return dict(state), budget, L

    def restore(snap):
        nonlocal state, budget, L
        state, budget, L = snap

    # ---- fast greedy: always cover the first uncovered gap -----------------
    def greedy_init(seed):
        nonlocal state, budget, L, counts
        state = {}
        budget = Fraction(1, 1)
        L = 0
        counts = [0] * (MAXL + 1)
        rng = random.Random(seed)
        for _ in range(len(primes)):
            g = first_gap()
            if g > MAXL:
                break
            # candidate primes: affordable, able to cover g, not used
            cands = [p for p in primes if p not in state and budget >= recip[p]]
            if not cands:
                break
            # score each candidate by how many of the next window it newly covers
            W = 400
            best_p = None
            best_gain = 0
            # order candidates: prefer small primes (cheap, dense) with tie-break random
            cands.sort(key=lambda p: (p, rng.random()))
            for p in cands:
                r = (g - 1) % p
                gain = 0
                for x in range(g, min(g + W, MAXL) + 1):
                    if counts[x] == 0 and (x - r) % p == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    if best_gain >= W:
                        break
            if best_p is None or best_gain == 0:
                break
            add(best_p, (g - 1) % best_p)
        return L

    # ---- local search ------------------------------------------------------
    def local_search(seed, max_iters):
        nonlocal L
        rng = random.Random(seed)
        best_snap = snapshot()
        best_L = L
        T = 1.0
        T_min = 1e-4
        alpha = 0.9999
        it = 0
        while it < max_iters and time.time() < DEADLINE:
            T = max(T_min, T * alpha)
            g = first_gap()
            mt = rng.random()
            if g <= MAXL and mt < 0.60:
                # move a used prime to cover the first gap
                used = list(state.keys())
                if used:
                    p = rng.choice(used)
                    new_r = (g - 1) % p
                    if new_r != state[p]:
                        old_L = L
                        change_residue(p, new_r)
                        if L < old_L and rng.random() > (0.05 ** ((old_L - L) / 30 + 1)):
                            change_residue(p, state[p])
                            # state already at new_r; revert needs old_r -> recompute
                            # (cheap enough relative to move cost)
            elif g <= MAXL and mt < 0.80:
                # add a new prime to cover the first gap
                cands = [p for p in primes if p not in state and budget >= recip[p]]
                if cands:
                    p = rng.choice(cands)
                    add(p, (g - 1) % p)
            elif mt < 0.92:
                # random residue tweak of a used prime
                used = list(state.keys())
                if used:
                    p = rng.choice(used)
                    old_r = state[p]
                    new_r = rng.randrange(p)
                    old_L = L
                    change_residue(p, new_r)
                    if L < old_L and rng.random() > (0.05 ** ((old_L - L) / 30 + 1)):
                        change_residue(p, old_r)
            else:
                # remove a prime to free budget (occasionally)
                used = list(state.keys())
                if len(used) > 4:
                    p = rng.choice(used)
                    remove(p)
            it += 1
            if L > best_L:
                best_L = L
                best_snap = snapshot()
                if rng.random() < 0.02:
                    restore(best_snap)
        restore(best_snap)
        return best_L

    # ---- driver ------------------------------------------------------------
    best_overall = []
    best_overall_L = 0

    # 1) greedy seeds
    for seed in range(12):
        if time.time() > DEADLINE:
            break
        gl = greedy_init(seed)
        if gl > best_overall_L:
            best_overall_L = gl
            best_overall = [p, state[p]] and [[p, state[p]] for p in state]
            best_overall = [[p, state[p]] for p in state]

    # 2) local search from best greedy states
    for seed in range(6):
        if time.time() > DEADLINE:
            break
        greedy_init(seed)
        if L > best_overall_L:
            best_overall_L = L
            best_overall = [[p, state[p]] for p in state]
        ll = local_search(1000 + seed, 200000)
        if ll > best_overall_L:
            best_overall_L = ll
            best_overall = [[p, state[p]] for p in state]

    if best_overall:
        return best_overall

    # fallback
    primes13 = [p for p in primes if p >= 13]
    return [[p, (i + 1) % p] for i, p in enumerate(primes13)]
