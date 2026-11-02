"""Structural change: Max-SAT-style local search (WalkSAT) over block selection,
with a fast incidence-based coverage oracle and a wall-clock deadline guard."""
import time
import random
import itertools

import numpy as np

_DEADLINE = 150.0
_MARGIN = 18.0

# ---------- combinatorial helpers ----------
def _choose(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    if r == 0:
        return 1
    res = 1
    for i in range(1, r + 1):
        res = res * (n - r + i) // i
    return res


def _all_subsets(s, r):
    return itertools.combinations(s, r)


# ---------- incidence / coverage oracle ----------
class CoverOracle:
    def __init__(self, v, k, t):
        self.v = v
        self.k = k
        self.t = t
        self.block_count = _choose(v, k)
        self.tset_count = _choose(v, t)

        # map each t-subset to an index
        self.tset_index = {}
        idx = 0
        for c in _all_subsets(range(v), t):
            self.tset_index[c] = idx
            idx += 1

        # incidence: for each t-set, the list of blocks (as point tuples) that cover it
        self.incidence = [[] for _ in range(self.tset_count)]
        for blk in _all_subsets(range(v), k):
            for sub in _all_subsets(blk, t):
                self.incidence[self.tset_index[sub]].append(blk)
        # convert to numpy arrays for fast membership
        self.incidence_arr = [np.array(x) if x else np.array([], dtype=int)
                              for x in self.incidence]
        self.avg_inc = self.tset_count and (
            sum(len(a) for a in self.incidence) / self.tset_count)

    def blocks_for(self, tset):
        return self.incidence_arr[self.tset_index[tset]]


# ---------- search ----------
def _initial_seed(oracle, rng, target):
    """Randomly pick ~target blocks; return (selected set, covered bitset)."""
    v, k, t = oracle.v, oracle.k, oracle.t
    selected = set()
    covered = np.zeros(oracle.tset_count, dtype=np.uint8)
    for _ in range(target):
        b = tuple(rng.sample(range(v), k))
        selected.add(b)
        for sub in _all_subsets(b, t):
            covered[oracle.tset_index[sub]] = 1
    return selected, covered


def _walksat(oracle, selected, covered, rng, budget):
    v, k, t = oracle.v, oracle.k, oracle.t
    end = time.time() + budget
    best = (len(selected), set(selected))
    n_sel = len(selected)

    while time.time() < end:
        # pick an uncovered t-set
        uncov = np.nonzero(1 - covered)[0]
        if uncov.size == 0:
            break
        ti = int(rng.choice(uncov))
        tset = tuple(next(c for c, i in oracle.tset_index.items() if i == ti))
        # fallback: rebuild tset from index via inverse map
        inv = oracle.tset_index
        # find the tset for index ti
        tset = None
        for c, i in inv.items():
            if i == ti:
                tset = c
                break

        cand = oracle.blocks_for(tset)
        cand = [b for b in cand if b not in selected]
        if not cand:
            continue

        # evaluate each candidate: new coverage gained minus 1 (the block cost)
        best_b = None
        best_score = -10**9
        for b in cand:
            gained = 0
            for sub in _all_subsets(b, t):
                if covered[oracle.tset_index[sub]] == 0:
                    gained += 1
            score = gained - 1
            if score > best_score:
                best_score = score
                best_b = b

        if best_b is None:
            continue

        # remove a random selected block (to keep size near target)
        to_remove = next(iter(rng.sample(list(selected), 1)))
        # apply swap
        for sub in _all_subsets(best_b, t):
            covered[oracle.tset_index[sub]] = 1
        for sub in _all_subsets(to_remove, t):
            i = oracle.tset_index[sub]
            cnt = 0
            for b2 in oracle.blocks_for(sub):
                if b2 in selected and b2 != to_remove:
                    cnt += 1
            if cnt == 0:
                covered[i] = 0
        selected.discard(to_remove)
        selected.add(best_b)

        if len(selected) < best[0]:
            best = (len(selected), set(selected))
            if best[0] <= oracle.tset_count:  # sanity
                pass

    return best


def construct(v, k, t):
    start = time.time()
    limit = _DEADLINE - _MARGIN
    rng = random.Random(20240517)
    oracle = CoverOracle(v, k, t)

    # target number of blocks to keep the search tractable
    lower = 0
    # a rough lower bound: ceil(C(v,t)/C(k,t))
    lower = (oracle.tset_count + _choose(k, t) - 1) // _choose(k, t)
    target = max(lower, int(oracle.avg_inc * 0.6))
    target = min(target, oracle.tset_count)

    best = None
    # initial seed + several local-search passes with different seeds
    attempts = 40
    for a in range(attempts):
        if time.time() > start + limit:
            break
        rng.seed(20240517 + a)
        selected, covered = _initial_seed(oracle, rng, target)
        rem = start + limit - time.time()
        if rem < 1.0:
            break
        res = _walksat(oracle, selected, covered, rng, max(rem, 0.5))
        if best is None or res[0] < best[0]:
            best = res
        # if we reached a perfect cover, stop early
        if best[0] <= lower:
            break

    if best is None:
        # fallback: return an initial seed (may be invalid but within budget)
        selected, covered = _initial_seed(oracle, rng, target)
        return [list(b) for b in sorted(selected)]

    return [list(b) for b in sorted(best[1])]
