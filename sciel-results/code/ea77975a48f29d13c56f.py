import time
import random

N_POINTS = 16
N_MASKS = 1 << N_POINTS
FULL_MASK = N_MASKS - 1

# ---- Precompute ----
# All 560 triples as masks
TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((1 << a) | (1 << b) | (1 << c))
NUM_TRIPLES = len(TRIPLES)

# Mask -> index
MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}

# Pairs: map 16x16 -> 0..119
PAIR_IDX = [[-1] * N_POINTS for _ in range(N_POINTS)]
PAIR_LIST = []
pid = 0
for i in range(N_POINTS):
    for j in range(i + 1, N_POINTS):
        PAIR_IDX[i][j] = pid
        PAIR_IDX[j][i] = pid
        PAIR_LIST.append((i, j))
        pid += 1
NUM_PAIRS = len(PAIR_LIST)

# For each triple: its 3 pair indices and its 2^13 supersets
TRIPLE_PAIRS = []
TRIPLE_SUPERSETS = []
for t in TRIPLES:
    pts = [i for i in range(N_POINTS) if (t >> i) & 1]
    TRIPLE_PAIRS.append((PAIR_IDX[pts[0]][pts[1]],
                         PAIR_IDX[pts[0]][pts[2]],
                         PAIR_IDX[pts[1]][pts[2]]))
    comp = FULL_MASK ^ t
    subs = []
    s = comp
    while True:
        subs.append(t | s)
        if s == 0:
            break
        s = (s - 1) & comp
    TRIPLE_SUPERSETS.append(subs)

# Masks sorted by popcount descending (for fast alpha)
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)
SORTED_MASKS = sorted(range(N_MASKS), key=lambda x: POP_COUNT[x], reverse=True)


def construct():
    deadline = time.time() + 112.0
    rng = random.Random(20240611)

    # State
    in_sys = [False] * NUM_TRIPLES
    sys_list = []          # list of active indices
    pair_count = [0] * NUM_PAIRS
    blocked = [0] * N_MASKS
    # coverage[i] = number of system triples containing point i
    coverage = [0] * N_POINTS

    def add(idx):
        if in_sys[idx]:
            return
        in_sys[idx] = True
        sys_list.append(idx)
        p0, p1, p2 = TRIPLE_PAIRS[idx]
        pair_count[p0] += 1
        pair_count[p1] += 1
        pair_count[p2] += 1
        for s in TRIPLE_SUPERSETS[idx]:
            blocked[s] += 1
        t = TRIPLES[idx]
        for i in range(N_POINTS):
            if (t >> i) & 1:
                coverage[i] += 1

    def remove(idx):
        if not in_sys[idx]:
            return
        in_sys[idx] = False
        # remove from sys_list (swap with last)
        pos = sys_list.index(idx)
        last = sys_list[-1]
        sys_list[pos] = last
        sys_list.pop()
        p0, p1, p2 = TRIPLE_PAIRS[idx]
        pair_count[p0] -= 1
        pair_count[p1] -= 1
        pair_count[p2] -= 1
        for s in TRIPLE_SUPERSETS[idx]:
            blocked[s] -= 1
        t = TRIPLES[idx]
        for i in range(N_POINTS):
            if (t >> i) & 1:
                coverage[i] -= 1

    def can_add(idx):
        p0, p1, p2 = TRIPLE_PAIRS[idx]
        return (pair_count[p0] < 2 and pair_count[p1] < 2 and pair_count[p2] < 2)

    def alpha_and_set():
        # find largest mask with blocked == 0
        for s in SORTED_MASKS:
            if blocked[s] == 0:
                return POP_COUNT[s], s
        return 0, 0

    def greedy_build(bias_set, max_time):
        # bias_set: mask of points to prioritize (current alpha set)
        # order: triples hitting many bias points first, then random
        order = list(range(NUM_TRIPLES))
        rng.shuffle(order)
        # sort by (-#hit_bias, random) using key
        def key(i):
            t = TRIPLES[i]
            hb = POP_COUNT[t & bias_set]
            return (-hb, rng.random())
        order.sort(key=key)
        for i in order:
            if time.time() > max_time:
                break
            if can_add(i):
                add(i)

    def removal_pass(alpha_set, max_time):
        # Remove system triples fully contained in alpha_set (breaks it)
        # alpha_set is independent -> no system triple contained in it, so this
        # set is empty; the real effect comes from coverage-based removal below.
        # We instead remove triples that cover few points (redundant).
        removed = 0
        # find triples with minimal coverage contribution
        for _ in range(200):
            if time.time() > max_time:
                break
            if not sys_list:
                break
            # pick a random active triple, check if removing it keeps alpha same or lowers
            idx = rng.choice(sys_list)
            t = TRIPLES[idx]
            # coverage contribution: points it uniquely covers
            # count points where coverage==1 (only this triple covers them)
            unique = 0
            for i in range(N_POINTS):
                if (t >> i) & 1 and coverage[i] == 1:
                    unique += 1
            # also: does it help hit alpha_set?
            hits_alpha = POP_COUNT[t & alpha_set]
            # remove if it covers no unique point OR covers alpha_set poorly
            if unique == 0 or (unique <= 1 and hits_alpha == 0):
                remove(idx)
                removed += 1
            else:
                break
        return removed

    best_alpha = 10**9
    best_indices = None

    # ---- Main search loop ----
    for restart in range(40):
        # Reset
        in_sys = [False] * NUM_TRIPLES
        sys_list = []
        pair_count = [0] * NUM_PAIRS
        blocked = [0] * N_MASKS
        coverage = [0] * N_POINTS

        # Phase A: greedy build (biased to hit a random large set)
        bias = FULL_MASK  # start with no bias, build a large system
        greedy_build(bias, deadline - 2.0)

        # Phase B: iterative remove + re-add to shrink alpha
        for it in range(30):
            if time.time() > deadline - 3.0:
                break
            a, s = alpha_and_set()
            # Record best
            if a < best_alpha:
                best_alpha = a
                best_indices = list(sys_list)
                if best_alpha <= 5:
                    break
            # Removal: remove triples that are redundant (cover few unique points)
            # and do not hit the alpha set
            improved = False
            for _ in range(60):
                if time.time() > deadline - 3.0:
                    break
                if not sys_list:
                    break
                idx = rng.choice(sys_list)
                t = TRIPLES[idx]
                unique = 0
                for i in range(N_POINTS):
                    if (t >> i) & 1 and coverage[i] == 1:
                        unique += 1
                hits_alpha = POP_COUNT[t & s]
                if unique == 0 or (unique <= 1 and hits_alpha == 0):
                    remove(idx)
                    improved = True
                else:
                    break
            # Re-add: greedily add triples biased to hit the current alpha set
            if time.time() > deadline - 3.0:
                break
            order = list(range(NUM_TRIPLES))
            rng.shuffle(order)
            def key2(i):
                t = TRIPLES[i]
                hb = POP_COUNT[t & s]
                return (-hb, rng.random())
            order.sort(key=key2)
            for i in order:
                if time.time() > deadline - 3.0:
                    break
                if can_add(i):
                    add(i)
            if not improved and it > 3:
                # try a random swap to escape local structure
                if sys_list:
                    idx_out = rng.choice(sys_list)
                    remove(idx_out)
                    # add a random feasible triple
                    for _ in range(200):
                        idx_in = rng.randrange(NUM_TRIPLES)
                        if can_add(idx_in):
                            add(idx_in)
                            break
                    if time.time() > deadline - 3.0:
                        break

        # final record
        a, s = alpha_and_set()
        if a < best_alpha:
            best_alpha = a
            best_indices = list(sys_list)
        if best_alpha <= 5:
            break

    # ---- Build result from best_indices ----
    if best_indices is None:
        # fallback: empty system (alpha = 16, valid)
        return []
    result = []
    for idx in best_indices:
        t = TRIPLES[idx]
        pts = [i for i in range(N_POINTS) if (t >> i) & 1]
        result.append(pts)
    return result
