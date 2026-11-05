"""K2,3-free graph on 20 vertices: maximize edges."""
import time
import random

N = 20
ALL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]

    # Phase 1: many randomized greedy restarts
    best_adj = [0] * N
    best_edges = []
    best_count = 0

    n_restarts = 300
    for r in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        adj, edges = _greedy(random.Random(100000 + r))
        c = len(edges)
        if c > best_count:
            best_count = c
            best_adj = adj
            best_edges = edges

    # Phase 2: local search on best (and on other near-best solutions)
    # Collect a few top solutions
    solutions = []
    for r in range(80):
        if time.time() > deadline - 45.0:
            break
        adj, edges = _greedy(random.Random(200000 + r))
        solutions.append((len(edges), adj, edges))
    solutions.sort(reverse=True)

    seen = set()
    for c, adj, edges in solutions[:6]:
        if time.time() > deadline - 8.0:
            break
        key = _edge_key(edges)
        if key in seen:
            continue
        seen.add(key)
        adj2, edges2, c2 = _local_search(adj, edges, deadline)
        if c2 > best_count:
            best_count = c2
            best_adj = adj2
            best_edges = edges2

    return best_edges


def _greedy(rng):
    """Randomized greedy: add edges in random order if K2,3-free."""
    adj = [0] * N
    edges = []
    order = list(range(N * (N - 1) // 2))
    rng.shuffle(order)
    # map index -> (u,v)
    # precompute mapping
    idx_map = []
    k = 0
    for u in range(N):
        for v in range(u + 1, N):
            idx_map.append((u, v))
            k += 1
    for idx in order:
        u, v = idx_map[idx]
        if _can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
    return adj, edges


def _can_add(adj, u, v):
    """Return True if adding edge (u,v) keeps graph K2,3-free.
    adj is list of bitmasks (current graph WITHOUT (u,v))."""
    au = adj[u]
    av = adj[v]

    # common neighbors of u and v (without the new edge)
    common_uv = au & av
    if common_uv.bit_count() >= 3:
        return False

    # For each neighbor w of v (w != u): after adding, common(u,w) = (au & adj[w]) + [v if v in adj[w]]
    # v is in adj[w] iff w in av
    w = av
    while w:
        low = w & (-w)
        w ^= low
        wi = low.bit_length() - 1
        if wi == u:
            continue
        # common of u and wi
        c = (au & adj[wi]).bit_count()
        if (av >> wi) & 1:  # v in adj[wi]
            c += 1
        if c >= 3:
            return False

    # For each neighbor w of u (w != v): after adding, common(v,w) = (av & adj[w]) + [u if u in adj[w]]
    w = au
    while w:
        low = w & (-w)
        w ^= low
        wi = low.bit_length() - 1
        if wi == v:
            continue
        c = (av & adj[wi]).bit_count()
        if (au >> wi) & 1:  # u in adj[wi]
            c += 1
        if c >= 3:
            return False

    return True


def _edge_key(edges):
    s = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        s.add((u, v))
    return frozenset(s)


def _local_search(adj, edges, deadline):
    """Improve by: remove one edge, greedily re-add all possible; keep if net gain.
    Also try removing two edges. Returns (adj, edges, count)."""
    adj = list(adj)
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    # adjacency bitmask already in adj
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        if u > v:
            u, v = v, u
        edge_set.discard((u, v))

    def greedy_fill():
        """Add any possible edge repeatedly until none."""
        changed = True
        while changed:
            changed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) in edge_set:
                        continue
                    if _can_add(adj, u, v):
                        add_edge(u, v)
                        changed = True
                        if time.time() > deadline - 4.0:
                            return

    # Precompute all edge list
    all_e = [(u, v) for u in range(N) for v in range(u + 1, N)]

    best_adj = list(adj)
    best_set = set(edge_set)
    best_count = len(edge_set)
    rng = random.Random(42)

    iters = 0
    while time.time() < deadline - 6.0:
        iters += 1
        # pick a random existing edge to remove
        if not edge_set:
            break
        e = rng.choice(list(edge_set))
        u, v = e
        # snapshot
        old_adj = list(adj)
        old_set = set(edge_set)
        remove_edge(u, v)
        greedy_fill()
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_adj = list(adj)
            best_set = set(edge_set)
        else:
            # revert
            adj = old_adj
            edge_set = old_set

        # Occasionally try 2-edge removal
        if iters % 3 == 0 and len(edge_set) >= 2:
            lst = list(edge_set)
            i = rng.randrange(len(lst))
            j = rng.randrange(len(lst))
            if i == j:
                continue
            e1, e2 = lst[i], lst[j]
            old_adj = list(adj)
            old_set = set(edge_set)
            remove_edge(*e1)
            remove_edge(*e2)
            greedy_fill()
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_adj = list(adj)
                best_set = set(edge_set)
            else:
                adj = old_adj
                edge_set = old_set

        if iters > 20000:
            break

    # Rebuild edges list from best_set
    best_edges = [(u, v) for (u, v) in best_set]
    return best_adj, best_edges, best_count
# EVOLVE-BLOCK-END
