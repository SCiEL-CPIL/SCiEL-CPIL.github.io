# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
K = 3
MAX_CANDIDATES = 48
DEADLINE_SECONDS = 75  # safety margin below 90s wall-clock


def _validate_selected(selected):
    """Check that no element divides the sum of any 3 distinct OTHER selected elements."""
    s = set(selected)
    for a in selected:
        others = [v for v in selected if v != a]
        for t in combinations(others, K):
            if (sum(t) % a) == 0:
                return False
    return True


def _can_add(selected_set, x):
    """Check whether adding x keeps the set valid.
    Only need to check triples that include x (new violations), since the
    existing set is already valid.
    """
    # For each a in existing set: check triples of (x, u, v) where u,v in existing, u!=v, u!=a, v!=a
    # For a == x: check triples (u,v,w) all in existing distinct from x (already valid, skip)
    # So only existing a's need checking with the new x.
    sel = list(selected_set)
    n = len(sel)
    for a in sel:
        # need two distinct other elements from sel (excluding a) to pair with x
        # iterate pairs (u, v) in sel, u != v, u != a, v != a
        for i in range(n):
            u = sel[i]
            if u == a:
                continue
            for j in range(i + 1, n):
                v = sel[j]
                if v == a:
                    continue
                if (x + u + v) % a == 0:
                    return False
    return True


def _greedy_build(order, max_elements=MAX_CANDIDATES):
    """Greedy: add elements in given order if valid."""
    selected_set = set()
    for x in order:
        if x in selected_set:
            continue
        if len(selected_set) >= max_elements:
            break
        if _can_add(selected_set, x):
            selected_set.add(x)
    return selected_set


def _local_search(selected_set, max_iter=200000, deadline=None):
    """Local search: try adding a random element from [1,192] not in set.
    If it doesn't fit, try swapping one out.
    """
    best = set(selected_set)
    best_size = len(best)
    it = 0
    while it < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        it += 1
        # try to add a random missing element
        missing = [x for x in range(1, N + 1) if x not in best]
        if not missing:
            break
        x = random.choice(missing)
        if _can_add(best, x):
            best.add(x)
            best_size += 1
            continue
        # try swap: remove one, add x
        # pick a random element to remove, then test
        # To be efficient, only try if removing some element allows x
        # We'll try a few random removals
        for _ in range(5):
            if deadline is not None and time.time() > deadline:
                break
            a = random.choice(list(best))
            trial = set(best)
            trial.discard(a)
            if _can_add(trial, x):
                trial.add(x)
                best = trial
                # size unchanged, but composition changed; could enable future adds
                break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(12345)

    # Initial greedy with multiple orderings, keep best
    candidates_orders = [
        list(range(1, N + 1, -1)),            # large first
        list(range(1, N + 1)),                # small first
        list(range(1, N + 1, 2)),             # odd first
        list(range(2, N + 1, 2)),             # even first
    ]
    # add some random shuffles
    for _ in range(4):
        o = list(range(1, N + 1))
        random.shuffle(o)
        candidates_orders.append(o)

    best_set = set()
    for order in candidates_orders:
        if time.time() > deadline:
            break
        s = _greedy_build(order)
        if len(s) > len(best_set):
            best_set = s

    # Local search from best
    best_set = _local_search(best_set, max_iter=500000, deadline=deadline)

    # A few more random-restart local searches if time permits
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = _greedy_build(order)
        s = _local_search(s, max_iter=100000, deadline=deadline)
        if len(s) > len(best_set):
            best_set = s
        if len(best_set) >= MAX_CANDIDATES:
            break

    # Final safety: ensure validity
    result = list(best_set)
    if not _validate_selected(result):
        # fallback to greedy large-first
        result = list(_greedy_build(list(range(1, N + 1, -1))))

    return result


# EVOLVE-BLOCK-END
