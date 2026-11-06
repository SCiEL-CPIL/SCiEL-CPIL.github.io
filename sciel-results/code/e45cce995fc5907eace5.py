"""Bootstrap search program for Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools
import math

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None


def find_violation_fast(adj_masks, n):
    """Find any violating subset (size >= 6) that is regular, or None.
    adj_masks: list of integers where bit j is set if edge (i, j) exists.
    """
    # Check subsets of size 6 first, as they are most numerous and likely to exist
    # We can stop at the first violation found
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            degs = []
            for v in combo:
                degs.append((adj_masks[v] & mask).bit_count())
            
            d0 = degs[0]
            for d in degs[1:]:
                if d != d0:
                    break
            else:
                return combo
    return None


def random_graph(n, p=0.5, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    adj_masks = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append([u, v])
    return adj_masks, edges


def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e


def local_search(adj_masks, n, max_iters=100, deadline=None):
    rng = random.Random()
    current = adj_masks[:]

    viol = find_violation_fast(current, n)
    if viol is None:
        return n, get_edges_from_masks(current, n), True

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        viol = find_violation_fast(current, n)
        if viol is None:
            return n, get_edges_from_masks(current, n), True

        vlist = list(viol)
        if len(vlist) >= 2:
            u, v = rng.sample(vlist, 2)
        else:
            u, v = rng.choice([(a, b) for a in range(n) for b in range(a + 1, n)])

        # Toggle edge
        current[u] ^= (1 << v)
        current[v] ^= (1 << u)

    viol = find_violation_fast(current, n)
    return n, get_edges_from_masks(current, n), viol is None


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        attempts = 0
        # Allocate more attempts for smaller n if we haven't found a large one yet?
        # Or just fix a reasonable number.
        max_attempts = 5 if n >= 18 else 20

        while time.time() < deadline and attempts < max_attempts:
            attempts += 1

            # Try different probabilities
            p = rng.random() * 0.4 + 0.3 # 0.3 to 0.7

            adj_masks, edges = random_graph(n, p=p, seed=None)

            viol = find_violation_fast(adj_masks, n)

            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                    break
            else:
                new_n, new_edges, new_valid = local_search(
                    adj_masks, n,
                    max_iters=20 if n <= 15 else 10,
                    deadline=deadline
                )
                if new_valid and new_n > best_n:
                    best_n = new_n
                    best_edges = new_edges
                    best_valid = True
                    break

        if best_valid and best_n >= n:
            continue

    if not best_valid or best_n < 5:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
