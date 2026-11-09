"""Sorting network on 18 wires: search for a small, verified (0-1 principle) network."""
import time
import random
import numpy as np

N = 18
MASKS = (1 << N) - 1          # all 2^18 Boolean inputs
DEADLINE = 80.0               # safety margin under 90s wall-clock
_rng = random.Random(20240517)

# ---- global state (lazy) ----
MASKS_ARR = None
NET = None
COUNT = None
BEST_NET = None
BEST_COUNT = 153
FROZEN = None                 # best network as tuple-of-tuples
DEAD = False


def _init_masks():
    global MASKS_ARR
    if MASKS_ARR is None:
        MASKS_ARR = np.arange(1 << N, dtype=np.uint32)
    return MASKS_ARR


def _deadline_hit():
    return (time.time() - START_T) >= DEADLINE


def _apply_net(masks, net):
    """Simulate `net` (list of (i,j)) on all Boolean masks. min->i, max->j."""
    for i, j in net:
        ij = (1 << i) | (1 << j)
        m = masks & ij
        lo = m & (1 << i)
        hi = m & (1 << j)
        masks = (masks & ~ij) | np.where(m == 0, 0, np.where(m == ij, lo, hi))
    return masks


def _verify(net):
    """True iff net sorts every Boolean input (0-1 principle)."""
    masks = _init_masks()
    for i, j in net:
        ij = (1 << i) | (1 << j)
        m = masks & ij
        lo = m & (1 << i)
        hi = m & (1 << j)
        masks = (masks & ~ij) | np.where(m == 0, 0, np.where(m == ij, lo, hi))
    return bool(np.all(masks == MASKS))


def _depth(net):
    """Parallel depth (longest path) of the network."""
    dp = [0] * N
    for i, j in net:
        dp[j] = max(dp[j], dp[i] + 1)
    return max(dp)


def _greedy_layer(remaining):
    """Pick the comparator fixing the most `remaining` unsorted masks."""
    masks = _init_masks()
    best_key = -1
    best_c = (0, 1)
    for i in range(N):
        for j in range(i + 1, N):
            ij = (1 << i) | (1 << j)
            m = masks & ij
            lo = m & (1 << i)
            hi = m & (1 << j)
            fixed = np.where(m == 0, 0, np.where(m == ij, lo, hi))
            key = int(np.count_nonzero(remaining & fixed))
            if key > best_key:
                best_key = key
                best_c = (i, j)
    return best_c


def _build_greedy():
    """Build a depth-parallel network layer by layer until sorted."""
    masks = _init_masks()
    net = []
    for _ in range(300):
        if _verify(net):
            break
        remaining = MASKS ^ masks
        c = _greedy_layer(remaining)
        i, j = c
        ij = (1 << i) | (1 << j)
        m = masks & ij
        lo = m & (1 << i)
        hi = m & (1 << j)
        masks = (masks & ~ij) | np.where(m == 0, 0, np.where(m == ij, lo, hi))
        net.append(c)
        if _deadline_hit():
            break
    return net


def _local_search(net, max_iter=400):
    """Local search: add / swap / delete / reorder, keep verified improvements."""
    global BEST_NET, BEST_COUNT, FROZEN
    net = list(net)
    if not _verify(net):
        return net
    best = len(net)
    for it in range(max_iter):
        if _deadline_hit():
            break
        if it % 40 == 0 and len(net) < best:
            best = len(net)
            BEST_NET = list(net)
            BEST_COUNT = best
            FROZEN = tuple(net)
        if len(net) <= 0:
            break
        r = _rng.random()
        if r < 0.34:  # add a comparator
            i, j = _greedy_layer(MASKS ^ _apply_net(_init_masks(), net))
            cand = net + [(i, j)]
            if len(cand) < len(net) + 1 and _verify(cand) and len(cand) < best:
                net = cand
        elif r < 0.52:  # swap two comparators
            if len(net) < 2:
                continue
            a, b = _rng.sample(range(len(net)), 2)
            cand = list(net)
            cand[a], cand[b] = cand[b], cand[a]
            if _verify(cand) and len(cand) < best:
                net = cand
        elif r < 0.62:  # reorder (move one comparator)
            a = _rng.randrange(len(net))
            b = _rng.randrange(len(net))
            cand = list(net)
            c = cand.pop(a)
            cand.insert(b, c)
            if _verify(cand) and len(cand) < best:
                net = cand
        else:  # delete one comparator
            a = _rng.randrange(len(net))
            cand = list(net)
            del cand[a]
            if len(cand) < best and _verify(cand):
                net = cand
    if len(net) < best:
        best = len(net)
        BEST_NET = list(net)
        BEST_COUNT = best
        FROZEN = tuple(net)
    return net


def _seed_network():
    # Odd-even transposition (verified), 153 comparators for N=18.
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _finalize():
    global BEST_NET, BEST_COUNT, FROZEN
    if FROZEN is not None:
        net = list(FROZEN)
    elif BEST_NET is not None:
        net = list(BEST_NET)
    else:
        net = _seed_network()
    if _verify(net):
        if len(net) < BEST_COUNT:
            BEST_COUNT = len(net)
        return net
    # last-resort verified fallback
    return _seed_network()


START_T = time.time()


def construct():
    global NET, COUNT, BEST_NET, BEST_COUNT, FROZEN, DEAD
    if DEAD:
        return _finalize()

    # 1) start from a verified seed
    NET = _seed_network()
    COUNT = len(NET)
    if not _verify(NET):
        return _seed_network()
    BEST_NET = list(NET)
    BEST_COUNT = COUNT
    FROZEN = tuple(NET)

    # 2) greedy depth-parallel construction (usually far below 153)
    g = _build_greedy()
    if _verify(g) and len(g) < BEST_COUNT:
        BEST_NET = list(g)
        BEST_COUNT = len(g)
        FROZEN = tuple(g)
        NET = list(g)

    # 3) local search to shave comparators
    if not _deadline_hit():
        NET = _local_search(NET, max_iter=400)

    # 4) a second independent greedy attempt if time remains
    if not _deadline_hit():
        g2 = _build_greedy()
        if _verify(g2) and len(g2) < BEST_COUNT:
            BEST_NET = list(g2)
            BEST_COUNT = len(g2)
            FROZEN = tuple(g2)
            NET = list(g2)
            if not _deadline_hit():
                NET = _local_search(NET, max_iter=200)

    DEAD = True
    return _finalize()
