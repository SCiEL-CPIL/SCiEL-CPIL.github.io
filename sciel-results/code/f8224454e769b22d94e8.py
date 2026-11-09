"""Combined program: decoupled large/small construction + fast local search."""
import time

# EVOLVE-BLOCK-START
def _has200(subset):
    """True iff some sub-collection of subset sums to exactly 200."""
    reachable = {0}
    for x in subset:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return True
        reachable |= new
    return False


def _greedy(order):
    """Greedily add elements in `order` while no 200-sum is created."""
    selected = []
    reachable = {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected


def _local_search(subset):
    """Add any excluded element that stays valid, then try swaps to enable more."""
    S = set(subset)
    # Phase 1: pure additions (monotone, fast convergence)
    changed = True
    while changed:
        changed = False
        for x in range(1, 101):
            if x not in S and not _has200(S | {x}):
                S.add(x)
                changed = True
    # Phase 2: swaps (remove one, add two) to break plateaus
    for _ in range(60):
        improved = False
        for a in range(1, 101):
            if a in S:
                for b in range(a + 1, 101):
                    if b in S:
                        for c in range(1, 101):
                            if c not in S:
                                trial = (S - {a, b}) | {c}
                                if not _has200(trial):
                                    S = trial
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break
    return list(S)


def _seed_order():
    # Large numbers first (they cannot form a 200-sum alone), then descending small.
    return list(range(100, 0, -1))


def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin under the 90 s kill

    def elapsed():
        return time.time() - t0

    # Initial construction: greedy over descending order, then local refinement.
    cand = _greedy(_seed_order())
    cand = _local_search(cand)
    best = list(cand)

    # Try a couple of alternative orderings if time remains.
    import random
    rng = random.Random(12345)
    extra = 0
    while elapsed() < DEADLINE - 5.0 and extra < 30:
        order = list(range(1, 101))
        rng.shuffle(order)
        cand = _greedy(order)
        cand = _local_search(cand)
        if len(cand) > len(best):
            best = list(cand)
        extra += 1

    # Final safety: ensure nonempty and valid.
    if not best:
        best = list(range(100, 0, -1))
    if _has200(best):
        best = list(range(67, 101))  # guaranteed valid (all > 100? no, 67..100, sum-free for 200? 100+100 not allowed, 99+101 no)
    return best
# EVOLVE-BLOCK-END
