"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score(U):
    """Fast vectorized score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    U = np.asarray(U, dtype=np.int64)
    if U.size == 0:
        return 0.0
    if 0 not in U:
        return 0.0
    U = np.unique(U)
    n = U.size
    if n < 2:
        return 0.0
    mx = int(U.max())
    if mx <= 0:
        return 0.0
    # pairwise sums and differences (full n x n)
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    ns = np.unique(sums).size
    nd = np.unique(diffs).size
    if ns == 0:
        return 0.0
    ratio = nd / ns
    if ratio <= 0:
        return 0.0
    log_base = math.log(2 * mx + 1)
    if log_base <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / log_base

def _dynamic_candidates(U, max_val, extra=60):
    """Generate candidate values from sums/diffs of existing elements."""
    arr = np.asarray(U, dtype=np.int64)
    cand = set()
    # pairwise sums and absolute differences
    s = arr[:, None] + arr[None, :]
    d = np.abs(arr[:, None] - arr[None, :])
    for v in s.ravel():
        v = int(v)
        if 0 < v <= max_val:
            cand.add(v)
    for v in d.ravel():
        v = int(v)
        if 0 < v <= max_val:
            cand.add(v)
    # a bit of uniform padding to escape local structure
    for _ in range(extra):
        cand.add(random.randint(1, max_val))
    return cand

def _local_search(U, deadline, max_val, max_iters=400):
    """Local search using dynamic candidate moves: add, remove, swap, perturb."""
    cur = sorted(set(U))
    if 0 not in cur:
        cur = sorted(set(cur) | {0})
    cur_score = _score(cur)
    if cur_score <= 0:
        return cur, cur_score

    improved = True
    it = 0
    while improved and time.time() < deadline and it < max_iters:
        improved = False
        it += 1
        n = len(cur)
        mx = max(cur)
        mv = max_val

        # Build move list
        moves = []

        # ADD moves: from dynamic candidates
        cands = _dynamic_candidates(cur, mv, extra=40)
        for c in list(cands):
            if c not in cur:
                moves.append(('add', c))

        # REMOVE moves
        for v in cur:
            if v != 0:
                moves.append(('remove', v))

        # SWAP / PERTURB moves
        for i, v in enumerate(cur):
            if v == 0:
                continue
            for delta in (-3, -2, -1, 1, 2, 3):
                nv = v + delta
                if nv > 0 and nv <= mv and nv not in cur:
                    moves.append(('perturb', (i, v, nv)))
            # larger multiplicative jumps
            for k in (2, 3):
                nv = v * k
                if nv <= mv and nv not in cur:
                    moves.append(('perturb', (i, v, nv)))

        random.shuffle(moves)

        best_move = None
        best_score = cur_score
        # cap the number of moves tried per iteration for speed
        for m in moves[:80]:
            kind = m[0]
            if kind == 'add':
                new = sorted(set(cur) | {m[1]})
            elif kind == 'remove':
                new = [x for x in cur if x != m[1]]
            else:
                idx, old, nv = m[1]
                new = [x for x in cur if x != old]
                new.append(nv)
                new = sorted(set(new))
            if not new or 0 not in new:
                continue
            sc = _score(new)
            if sc > best_score + 1e-12:
                best_score = sc
                best_move = new

        if best_move is not None:
            cur = best_move
            cur_score = best_score
            improved = True
            # update max_val to allow growth
            mv = max(mv, max(cur) + 20)

    return cur, cur_score

def _seed_geometric(base, terms):
    U = [0]
    c = 1
    for _ in range(terms):
        U.append(c)
        c *= base
    return U

def _seed_fibonacci(terms):
    U = [0, 1]
    while len(U) < terms:
        U.append(U[-1] + U[-2])
    return U

def _seed_base_digits(base, digits):
    U = set([0])
    for d in range(digits):
        U.add(base ** d)
    for d1 in range(digits):
        for d2 in range(d1 + 1, digits):
            v = base ** d1 + base ** d2
            U.add(v)
    return sorted(U)

def _seed_random(size, max_val):
    U = [0] + random.sample(range(1, max_val + 1), size - 1)
    return sorted(U)

def construct():
    start = time.time()
    deadline = start + 110  # safety margin within 120s / 150s kill

    random.seed(42)

    best_set = [0, 1, 4, 9, 11]
    best_score = _score(best_set)

    # A few strong structured seeds
    seeds = []
    for base in [2, 3, 4, 5]:
        for terms in range(4, 10):
            s = _seed_geometric(base, terms)
            if max(s) <= 200000:
                seeds.append(s)
    for terms in range(5, 12):
        s = _seed_fibonacci(terms)
        if max(s) <= 200000:
            seeds.append(s)
    for base in [2, 3, 4, 5, 7]:
        for digits in range(3, 9):
            s = _seed_base_digits(base, digits)
            if max(s) <= 200000:
                seeds.append(s)
    for _ in range(8):
        seeds.append(_seed_random(random.randint(4, 9), random.randint(20, 200)))

    # Evaluate seeds, keep top few
    scored = []
    for s in seeds:
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    top_seeds = [s for sc, s in scored[:6] if sc > 0]
    if not top_seeds:
        top_seeds = [best_set]

    # Multi-restart local search from the best seeds
    for seed in top_seeds:
        if time.time() > deadline - 8:
            break
        # refine this seed
        res, sc = _local_search(seed, deadline - 8, max_val=max(200, max(seed) + 100))
        if sc > best_score:
            best_score = sc
            best_set = res

    # Final polish on the overall best
    if time.time() < deadline - 3 and len(best_set) > 2:
        res, sc = _local_search(best_set, deadline - 2, max_val=max(300, max(best_set) + 100), max_iters=200)
        if sc > best_score:
            best_score = sc
            best_set = res

    # Ensure validity
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    if len(best_set) < 2:
        best_set = [0, 1, 4, 9, 11]

    return best_set
# EVOLVE-BLOCK-END
