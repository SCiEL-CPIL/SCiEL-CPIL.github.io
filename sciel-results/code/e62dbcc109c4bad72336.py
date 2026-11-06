import numpy as np
import time
import random

N = 17
DEADLINE = 78.0  # safety margin before 90s kill

# --- Precompute bit-unpacking constants ---
SHIFTS = np.arange(N - 1, -1, -1, dtype=np.uint32)
INDICES = np.arange(1 << N, dtype=np.uint32)
# (2^N, N) uint8 array of all bit inputs, row = integer's bits
BITS = ((INDICES[:, None] >> SHIFTS) & 1).astype(np.uint8)
M = 1 << N  # 131072


def _build_masks(net):
    """Precompute per-wire uint32 bitmasks of '1' over all inputs for a network."""
    masks = np.empty((N, M), dtype=np.uint32)
    for i in range(N):
        masks[i] = BITS[:, i].astype(np.uint32)
    for i, j in net:
        a = masks[i]
        b = masks[j]
        lo = a & b
        hi = a | b
        masks[i] = lo
        masks[j] = hi
    return masks


def _is_valid(net):
    """True if net sorts all 2^N inputs."""
    masks = _build_masks(net)
    for k in range(N - 1):
        if np.any(masks[k] & masks[k + 1]):
            return False
    return True


def _depth(net):
    """Parallel depth (max number of comparators on any wire path)."""
    d = [0] * N
    for i, j in net:
        nd = max(d[i], d[j]) + 1
        d[i] = nd
        d[j] = nd
    return max(d) if d else 0


# --- Seed networks ---
def _odd_even_transposition():
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net


def _bitonic_sort():
    """Batcher bitonic sorting network."""
    def bitonic_merge(lo, hi, step):
        # ascending merge of a bitonic sequence in [lo, hi)
        comps = []
        for i in range(lo, hi, 2 * step):
            for j in range(i, i + step):
                comps.append((j, j + step))
        return comps

    net = []
    size = N
    # recursively build
    def rec(lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        rec(lo, lo + half, 1)          # ascending
        rec(lo + half, hi, -1)         # descending
        # merge into ascending
        step = half
        while step > 0:
            for i in range(lo, hi, 2 * step):
                for j in range(i, i + step):
                    net.append((j, j + step))
            step //= 2

    rec(0, N, 1)
    return net


def _akl():
    """AKL (Ajtai-Komlos-Szemeredi) style network: pairwise + bitonic merge."""
    net = []
    # Phase 1: pairwise (i, i+N/2) for N even; here N=17 odd, so use even count
    m = N - (N % 2)  # 16
    for i in range(m // 2):
        net.append((i, i + m // 2))
    # Phase 2: bitonic merge of the m wires (ascending)
    def merge(lo, hi):
        length = hi - lo
        if length <= 1:
            return
        step = length // 2
        while step > 0:
            for i in range(lo, hi, 2 * step):
                for j in range(i, i + step):
                    net.append((j, j + step))
            step //= 2
    merge(0, m)
    # The 17th wire (index 16) is a singleton; add comparators to fold it in.
    # Insert wire 16 by comparing against a chain and merging.
    for i in range(m - 1, -1, -1):
        net.append((i, 16))
        net.append((i, i + 1) if i + 1 < m else (i, 16))
    # Simpler robust approach: just append a full bitonic over all 17 to be safe.
    return net


def _bitonic_all():
    """Bitonic sort over all 17 wires (robust seed)."""
    net = []
    def rec(lo, hi):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        rec(lo, lo + half)        # ascending
        # descending on right half
        def desc(a, b):
            l = b - a
            if l <= 1:
                return
            h = l // 2
            desc(a + h, b)        # ascending
            desc(a, a + h)        # ascending
            step = h
            while step > 0:
                for i in range(a, b, 2 * step):
                    for j in range(i, i + step):
                        net.append((j, j + step))
                step //= 2
        desc(lo + half, hi)
        # merge ascending
        step = half
        while step > 0:
            for i in range(lo, hi, 2 * step):
                for j in range(i, i + step):
                    net.append((j, j + step))
            step //= 2
    rec(0, N)
    return net


# --- Local search moves ---
def _try_remove(net, i):
    saved = net[i]
    del net[i]
    ok = _is_valid(net)
    if not ok:
        net.insert(i, saved)
    return ok


def _try_swap(net, i, j):
    a, b = net[i], net[j]
    net[i], net[j] = b, a
    ok = _is_valid(net)
    if not ok:
        net[i], net[j] = a, b
    return ok


def _try_insert(net, i, j, k):
    """Move net[i] to position k (after removing at i, insert at k)."""
    comp = net[i]
    del net[i]
    if k > i:
        k -= 1
    net.insert(k, comp)
    ok = _is_valid(net)
    if not ok:
        del net[k]
        net.insert(i, comp)
    return ok


def _try_replace(net, i, newcomp):
    """Replace net[i] with newcomp (a different pair)."""
    old = net[i]
    net[i] = newcomp
    ok = _is_valid(net)
    if not ok:
        net[i] = old
    return ok


def _local_search(net, deadline, rng):
    """Aggressive local search: deletions, swaps, moves, and single-pair mutations."""
    best = net[:]
    best_size = len(net)
    best_depth = _depth(best)

    # Precompute all valid candidate pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def time_left():
        return time.time() < deadline

    improved = True
    passes = 0
    while improved and time_left() and passes < 400:
        passes += 1
        improved = False

        # --- Pass A: deletions (front to back) ---
        i = 0
        while i < len(net) and time_left():
            if _try_remove(net, i):
                improved = True
                if len(net) < best_size:
                    best = net[:]
                    best_size = len(net)
                    best_depth = _depth(best)
                # do not increment i; re-examine same index
            else:
                i += 1

        # --- Pass B: swaps (random pairs) ---
        if time_left():
            L = len(net)
            for _ in range(min(4000, L * L // 4)):
                if not time_left():
                    break
                i = rng.randrange(L)
                j = rng.randrange(L)
                if i == j:
                    continue
                if _try_swap(net, i, j):
                    improved = True
                    if len(net) < best_size:
                        best = net[:]
                        best_size = len(net)
                        best_depth = _depth(best)

        # --- Pass C: moves (relocate a comparator) ---
        if time_left():
            L = len(net)
            for _ in range(min(4000, L * 3)):
                if not time_left():
                    break
                i = rng.randrange(L)
                k = rng.randrange(L)
                if _try_insert(net, i, 0, k):
                    improved = True
                    if len(net) < best_size:
                        best = net[:]
                        best_size = len(net)
                        best_depth = _depth(best)

        # --- Pass D: single-pair mutation (replace one comparator with another) ---
        if time_left():
            L = len(net)
            for _ in range(min(3000, L * 4)):
                if not time_left():
                    break
                i = rng.randrange(L)
                newcomp = all_pairs[rng.randrange(len(all_pairs))]
                if newcomp == net[i]:
                    continue
                if _try_replace(net, i, newcomp):
                    improved = True
                    if len(net) < best_size:
                        best = net[:]
                        best_size = len(net)
                        best_depth = _depth(best)

        # Keep best as the working net for next pass if it shrank
        if len(net) > best_size:
            net = best[:]

    return best


def construct():
    """
    Constructs a size-efficient sorting network for N=17 wires.
    Returns an ordered list of (i, j) comparator pairs that sort all inputs.
    """
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)

    # Build candidate seed networks, verify, pick the smallest valid one.
    seeds = []
    oe = _odd_even_transposition()
    seeds.append(oe)
    ba = _bitonic_all()
    seeds.append(ba)
    # Try AKL too (may be smaller)
    try:
        ak = _akl()
        seeds.append(ak)
    except Exception:
        pass

    best_net = None
    best_size = 10 ** 9
    for s in seeds:
        if time.time() > deadline:
            break
        if len(s) >= best_size:
            continue
        if _is_valid(s):
            best_net = s
            best_size = len(s)

    if best_net is None:
        # Fallback: use odd-even (always valid)
        best_net = oe

    # Local search from the best seed
    if time.time() < deadline:
        improved_net = _local_search(best_net, deadline, rng)
        if len(improved_net) < len(best_net):
            best_net = improved_net

    # Final safety: ensure validity before returning
    if not _is_valid(best_net):
        # reconstruct a valid network
        for s in seeds:
            if _is_valid(s):
                best_net = s
                break
        else:
            best_net = oe

    return best_net
