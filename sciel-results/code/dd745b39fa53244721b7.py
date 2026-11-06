"""Bootstrap search program for Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools
import math

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None


def find_violation_fast(adj_mask, n):
    """Find any violating subset (size >= 6) that is regular, or None.
    adj_mask: list of int, adj_mask[i] has bit j set if edge (i,j) exists.
    """
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            m = k
            degs = [0] * m
            for i in range(m):
                vi = combo[i]
                row = adj_mask[vi]
                for j in range(i + 1, m):
                    if (row >> combo[j]) & 1:
                        degs[i] += 1
                        degs[j] += 1
            d0 = degs[0]
            for d in degs[1:]:
                if d != d0:
                    break
            else:
                return combo
    return None


def adj_to_mask(adj, n):
    """Convert adjacency matrix to list of bitmasks."""
    mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                mask[u] |= (1 << v)
                mask[v] |= (1 << u)
    return mask


def mask_to_edges(mask, n):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for u in range(n):
        m = mask[u]
        for v in range(u + 1, n):
            if (m >> v) & 1:
                edges.append([u, v])
    return edges


def flip_edge(mask, u, v):
    """Flip edge (u,v) in bitmask representation."""
    mask[u] ^= (1 << v)
    mask[v] ^= (1 << u)


def random_graph_mask(n, p=0.5, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                mask[u] |= (1 << v)
                mask[v] |= (1 << u)
    return mask


def quadratic_residue_graph(n):
    """Construct a graph using quadratic residues in Z_n."""
    if n % 2 == 0 or n % 4 != 3:
        return None
    mask = [0] * n
    # Quadratic residues mod n
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    residues.discard(0)
    for u in range(n):
        for v in range(u + 1, n):
            d = (v - u) % n
            if d in residues or (n - d) in residues:
                mask[u] |= (1 << v)
                mask[v] |= (1 << u)
    return mask


def local_search_mask(mask, n, max_iters=50, deadline=None):
    """Local search using bitmask representation."""
    rng = random.Random()
    current = mask[:]

    viol = find_violation_fast(current, n)
    if viol is None:
        return n, mask_to_edges(current, n), True

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        viol = find_violation_fast(current, n)
        if viol is None:
            return n, mask_to_edges(current, n), True

        vlist = list(viol)
        if len(vlist) >= 2:
            u, v = rng.sample(vlist, 2)
        else:
            u, v = rng.choice([(a, b) for a in range(n) for b in range(a + 1, n)])

        flip_edge(current, u, v)

    viol = find_violation_fast(current, n)
    return n, mask_to_edges(current, n), viol is None


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(12345)

    # Try deterministic constructions first
    for n in [21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
        mask = quadratic_residue_graph(n)
        if mask is not None:
            viol = find_violation_fast(mask, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = mask_to_edges(mask, n)
                    best_valid = True
                break

    # Random search with local improvement
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break
        if best_valid and best_n >= n:
            continue

        attempts = 0
        max_attempts = 20 if n >= 18 else 50

        while time.time() < deadline and attempts < max_attempts:
            attempts += 1

            p = rng.random() * 0.4 + 0.3

            mask = random_graph_mask(n, p=p, seed=None)

            viol = find_violation_fast(mask, n)

            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = mask_to_edges(mask, n)
                    best_valid = True
                    break
            else:
                new_n, new_edges, new_valid = local_search_mask(
                    mask, n,
                    max_iters=30 if n <= 15 else 15,
                    deadline=deadline
                )
                if new_valid and new_n > best_n:
                    best_n = new_n
                    best_edges = new_edges
                    best_valid = True
                    break

    if not best_valid or best_n < 5:
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
