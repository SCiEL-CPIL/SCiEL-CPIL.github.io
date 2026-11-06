import time
import random

N = 20
C = 20
DEADLINE = 145.0


def _validate(edges):
    """Return True iff every 4-cycle is rainbow."""
    if not edges:
        return True
    adj = {}
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _has_c4(adj, neigh, u, v):
    """Check if adding edge (u,v) would create a 4-cycle."""
    common = neigh[u] & neigh[v]
    common.discard(u)
    common.discard(v)
    return len(common) >= 2


def _greedy_c4_free(order, rng):
    """Build a C4-free graph greedily."""
    adj = {}
    neigh = [set() for _ in range(N)]
    edges = []

    for (u, v) in order:
        if _has_c4(adj, neigh, u, v):
            continue
        adj[(u, v)] = 1
        adj[(v, u)] = 1
        neigh[u].add(v)
        neigh[v].add(u)
        edges.append([u, v, 1])

    return edges


def _local_search_c4_free(edges, adj, neigh, rng, time_budget):
    """Local search: remove an edge, try to add more edges back."""
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    for _ in range(2000):
        if time.time() > time_budget:
            break

        existing = set()
        for u, v, c in edges:
            existing.add((min(u, v), max(u, v)))

        candidates = [(u, v) for (u, v) in all_pairs if (u, v) not in existing]
        if not candidates:
            break

        # Try to add edges
        rng.shuffle(candidates)
        added_any = False
        for (u, v) in candidates:
            if time.time() > time_budget:
                break
            if not _has_c4(adj, neigh, u, v):
                adj[(u, v)] = 1
                adj[(v, u)] = 1
                neigh[u].add(v)
                neigh[v].add(u)
                edges.append([u, v, 1])
                added_any = True

        if not added_any and edges:
            # Remove a random edge and try again
            idx = rng.randrange(len(edges))
            a, b, c = edges[idx]
            edges.pop(idx)
            del adj[(a, b)]
            del adj[(b, a)]
            neigh[a].discard(b)
            neigh[b].discard(a)

    return edges


def _color_c4_free(edges, rng):
    """Color a C4-free graph. Since there are no 4-cycles, any proper coloring works.
    We just need to assign colors 0..19 such that adjacent edges get different colors.
    Use greedy coloring."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None] * N for _ in range(N)]

    for u, v, _ in edges:
        adj[(u, v)] = True
        adj[(v, u)] = True
        neigh[u].add(v)
        neigh[v].add(u)

    # Greedy proper edge coloring
    for u, v, _ in edges:
        used = set()
        for w in neigh[u]:
            if w != v and color_at[u][w] is not None:
                used.add(color_at[u][w])
        for w in neigh[v]:
            if w != u and color_at[v][w] is not None:
                used.add(color_at[v][w])

        for c in range(C):
            if c not in used:
                color_at[u][v] = c
                color_at[v][u] = c
                break
        else:
            # Should not happen with 20 colors and max degree < 20
            color_at[u][v] = 0
            color_at[v][u] = 0

    return [[u, v, color_at[u][v]] for u, v, _ in edges]


def construct():
    start = time.time()
    best_edges = []
    best_len = 0

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    # Multiple random greedy attempts to build C4-free graphs
    for trial in range(50):
        if time.time() > DEADLINE - 30:
            break

        rng = random.Random(trial * 7919 + 13)

        order = all_pairs[:]
        rng.shuffle(order)

        edges = _greedy_c4_free(order, rng)

        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]

    # Local search on best
    if best_edges and time.time() < DEADLINE - 15:
        adj = {}
        neigh = [set() for _ in range(N)]
        for u, v, c in best_edges:
            adj[(u, v)] = 1
            adj[(v, u)] = 1
            neigh[u].add(v)
            neigh[v].add(u)

        rng = random.Random(42)
        improved = _local_search_c4_free(best_edges[:], adj, neigh, rng, DEADLINE - 5)

        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved

    # Color the C4-free graph
    if best_edges:
        rng = random.Random(7)
        colored = _color_c4_free(best_edges, rng)
        return colored
    else:
        # Fallback: star (no 4-cycles)
        return [[0, v, v % C] for v in range(1, N)]


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
