"""Bootstrap search program: Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools
import sys

DEADLINE = 150
SAFETY_MARGIN = 12
START_TIME = None


def find_violation_fast(adj_masks, n):
    """Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None.
    Optimized: precompute adjacency bitmasks, check size-6 first, early exit on mismatch.
    """
    # Precompute for speed: adj_masks already stored as int bitmasks
    # Check size-6 subsets first (most numerous, most likely to exist)
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            return combo
    
    # Check larger subsets (less numerous)
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            ok = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    ok = False
                    break
            if ok:
                return combo
    return None


# Keep original name for compatibility
def find_violation(adj_masks, n):
    return find_violation_fast(adj_masks, n)


def random_graph(n, p, rng):
    """Generate a random graph with given edge probability p."""
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
    """Convert adjacency masks to edge list."""
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e


def local_search(adj_masks, n, rng, deadline):
    """Flip edges to eliminate regular induced subgraphs. Prefer flipping edges inside a violating set."""
    cur = adj_masks[:]

    viol = find_violation_fast(cur, n)
    if viol is None:
        return n, get_edges_from_masks(cur, n), True

    it = 0
    max_iters = 300
    while it < max_iters:
        if time.time() > deadline:
            break

        viol = find_violation_fast(cur, n)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = set(viol)
        # Collect candidate edges: prefer edges within the violating set
        internal_edges = []
        internal_nonedges = []
        for u in vset:
            for v in vset:
                if u < v:
                    if cur[u] & (1 << v):
                        internal_edges.append((u, v))
                    else:
                        internal_nonedges.append((u, v))
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

        # Strategy: flip an edge in the violating set to break regularity
        # Prefer flipping an existing edge (removing it) or adding a non-edge
        if internal_edges and internal_nonedges:
            # Mix: 50% remove existing, 50% add missing
            if rng.random() < 0.5:
                u, v = rng.choice(internal_edges)
            else:
                u, v = rng.choice(internal_nonedges)
        elif internal_edges:
            u, v = rng.choice(internal_edges)
        elif internal_nonedges:
            u, v = rng.choice(internal_nonedges)
        else:
            u, v = rng.choice(all_edges)

        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        it += 1

    viol = find_violation_fast(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # Increase attempts for larger n where finding valid graphs is harder
        if n >= 21:
            max_attempts = 12
        elif n >= 19:
            max_attempts = 10
        else:
            max_attempts = 15

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Vary edge probability: try a range of densities
            # Regular induced subgraphs are more likely at p=0.5, so avoid that
            p_choices = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            p = p_choices[rng.randrange(len(p_choices))]

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break  # found a valid graph for this n, move to next n

            # Local search to repair
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            # Already have a valid graph at this size or larger
            continue

    if not best_valid or best_n < 5:
        # Fallback: return a small valid graph
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
