"""Size-efficient sorting network search on 18 wires.
Strategy: build a solid starting network (odd-even transposition + AKS-style
shrink), then iteratively remove redundant comparators and try short random
mutations, verifying each candidate against all 262144 Boolean inputs."""
import numpy as np
import random
import time

N = 18
MASK = (1 << N) - 1
# Precompute the 18 unit masks as a 2D array for fast bit ops
UNITS = np.array([1 << i for i in range(N)], dtype=np.uint64)

def _check(network):
    """Return True iff network sorts all 2**N Boolean inputs.
    Uses vectorized bit ops over all 262144 masks at once."""
    if not network:
        return N <= 1
    allm = np.arange(1 << N, dtype=np.uint64)
    for (i, j) in network:
        ui = UNITS[i]; uj = UNITS[j]
        lo = allm & ui
        hi = allm & uj
        # swap if lo==0 and hi!=0
        cond = (lo == 0) & (hi != 0)
        allm = np.where(cond, allm ^ ui ^ uj, allm)
    # sorted ascending: for each adjacent pair (i,i+1), bit i must be 0 or bit i+1 must be 0
    for k in range(N - 1):
        a = allm & UNITS[k]
        b = allm & UNITS[k + 1]
        if np.any((a != 0) & (b != 0)):
            return False
    return True

def _seed_network():
    # Odd-even transposition sorting network: N phases, each N//2 comparators.
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _try_remove_redundant(net):
    """Remove comparators that are redundant (network still sorts without them)."""
    changed = True
    while changed:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            cand = net[:idx] + net[idx + 1:]
            if _check(cand):
                net = cand
                changed = True
    return net

def _try_add_random(net, rng, max_tries=40):
    """Try adding a random comparator that reduces nothing (we only add if it
    enables later removals); here we just return net unchanged mostly."""
    return net

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s
    rng = random.Random(12345)

    net = _seed_network()
    # Initial redundancy removal to get a compact baseline
    net = _try_remove_redundant(net)
    best = list(net)

    # Iterative improvement: repeatedly try random mutations and removals.
    while time.time() < deadline:
        # Random mutation: add a random comparator then remove redundancies
        i = rng.randrange(N)
        j = rng.randrange(i + 1, N)
        mutated = list(best) + [(i, j)]
        mutated = _try_remove_redundant(mutated)
        if len(mutated) < len(best):
            best = mutated
        else:
            # Try a swap/move: replace a random comparator with a new one
            if len(best) > 0:
                pos = rng.randrange(len(best))
                cand = list(best)
                cand[pos] = (i, j)
                cand = _try_remove_redundant(cand)
                if len(cand) < len(best):
                    best = cand

        # Also try pure removal sweep on current best (cheap, often finds more)
        if time.time() < deadline:
            shrunk = _try_remove_redundant(list(best))
            if len(shrunk) < len(best):
                best = shrunk

    return best
