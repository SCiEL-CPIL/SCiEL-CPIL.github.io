import time
import random
import sys

N = 20
C = 20
DEADLINE = 138.0  # safety margin under 150s limit


def _validate(edges):
    """Return True iff every 4-cycle is rainbow (4 distinct colours)."""
    adj = {}
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c

    def has_edge(a, b):
        return (a, b) in adj

    # Build neighbour lists
    neigh = {i: [] for i in range(N)}
    for u, v, c in edges:
        neigh[u].append(v)
        neigh[v].append(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            # common neighbours of u and v
            common = [w for w in neigh[u] if has_edge(v, w) and w != u]
            for x in common:
                for y in common:
                    if x >= y:
                        continue
                    # 4-cycle u - x - v - y - u
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _count_violations(adj, neigh, color_of):
    """Count the number of non-rainbow 4-cycles."""
    violations = 0
    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = [w for w in neigh[u] if (v, w) in adj and w != u]
            for i in range(len(common)):
                x = common[i]
                for j in range(i + 1, len(common)):
                    y = common[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        violations += 1
    return violations


def _check_add_fast(adj, neigh, color_of, u, v, c):
    """Fast check for adding edge (u,v) with colour c."""
    # Common neighbours of u and v
    common = neigh[u] & neigh[v]
    common.discard(u)
    common.discard(v)
    common_list = list(common)
    
    # Cycles not using the new edge: u - x - v - y - u
    for i in range(len(common_list)):
        x = common_list[i]
        for y in common_list[i + 1:]:
            cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
            if len(cols) != 4:
                return False
    
    # Cycles using the new edge (u,v): u - v - x - y - u
    for x in neigh[v]:
        if x == u:
            continue
        for y in neigh[u]:
            if y == v or y == x:
                continue
            key = (x, y) if x < y else (y, x)
            if key not in color_of:
                continue
            cols = {c, adj[(v, x)], adj[(x, y)], adj[(y, u)]}
            if len(cols) != 4:
                return False
    return True


def _try_greedy_order(order, rng):
    """Attempt to build a graph by adding edges in the given order."""
    edges = []
    adj = {}
    neigh = [set() for _ in range(N)]
    color_of = {}

    def add_edge(u, v, c):
        edges.append([u, v, c])
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)
        key = (u, v) if u < v else (v, u)
        color_of[key] = c

    for (u, v) in order:
        cols = list(range(C))
        rng.shuffle(cols)
        placed = False
        for c in cols:
            if _check_add_fast(adj, neigh, color_of, u, v, c):
                add_edge(u, v, c)
                placed = True
                break
    return edges


def _local_search(edges, adj, neigh, color_of, rng, time_budget):
    """Local search: try to recolor edges to remove violations, then add edges."""
    # First, try to fix violations by recoloring
    for _ in range(200):
        if time.time() > time_budget:
            break
        violations = _count_violations(adj, neigh, color_of)
        if violations == 0:
            break
        
        # Pick a random edge involved in a violation and try to recolor it
        # For simplicity, pick a random edge and try all colors
        if not edges:
            break
        idx = rng.randrange(len(edges))
        u, v, c = edges[idx]
        key = (u, v) if u < v else (v, u)
        
        # Try to remove this edge
        # Temporarily remove
        del adj[(u, v)]
        del adj[(v, u)]
        neigh[u].discard(v)
        neigh[v].discard(u)
        del color_of[key]
        edges.pop(idx)
        
        new_violations = _count_violations(adj, neigh, color_of)
        if new_violations < violations:
            # Keep it removed
            continue
        else:
            # Put it back
            edges.append([u, v, c])
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            color_of[key] = c
    
    # Now try to add more edges
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    existing = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    candidates = [p for p in all_pairs if p not in existing]
    rng.shuffle(candidates)
    
    for (u, v) in candidates:
        if time.time() > time_budget:
            break
        cols = list(range(C))
        rng.shuffle(cols)
        for c in cols:
            if _check_add_fast(adj, neigh, color_of, u, v, c):
                edges.append([u, v, c])
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                key = (u, v) if u < v else (v, u)
                color_of[key] = c
                break
    
    return edges


def construct():
    start = time.time()
    best_edges = None
    best_len = 0

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    # Strategy 1: Multiple random greedy orders
    for seed in range(20):
        if time.time() > DEADLINE:
            break
        rng = random.Random(seed)
        order = all_pairs[:]
        rng.shuffle(order)
        edges = _try_greedy_order(order, rng)
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]
    
    # Strategy 2: Local search improvement on the best found
    if best_edges and len(best_edges) > 0:
        adj = {}
        neigh = [set() for _ in range(N)]
        color_of = {}
        for u, v, c in best_edges:
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            key = (u, v) if u < v else (v, u)
            color_of[key] = c
        
        budget = min(time.time() + 10.0, DEADLINE - 5)
        improved = _local_search(best_edges[:], adj, neigh, color_of, random.Random(42), budget)
        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved

    # Strategy 3: Start with a sparse valid base and grow
    if time.time() < DEADLINE - 10:
        # Start with a random matching
        rng = random.Random(999)
        perm = list(range(N))
        rng.shuffle(perm)
        base_edges = []
        used = set()
        for i in range(0, N - 1, 2):
            u, v = perm[i], perm[i + 1]
            if u not in used and v not in used:
                c = i % C
                base_edges.append([u, v, c])
                used.add(u)
                used.add(v)
        
        adj = {}
        neigh = [set() for _ in range(N)]
        color_of = {}
        for u, v, c in base_edges:
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            key = (u, v) if u < v else (v, u)
            color_of[key] = c
        
        # Grow greedily
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        existing = set((min(e[0], e[1]), max(e[0], e[1])) for e in base_edges)
        candidates = [p for p in all_pairs if p not in existing]
        rng.shuffle(candidates)
        
        for (u, v) in candidates:
            if time.time() > DEADLINE - 5:
                break
            cols = list(range(C))
            rng.shuffle(cols)
            for c in cols:
                if _check_add_fast(adj, neigh, color_of, u, v, c):
                    base_edges.append([u, v, c])
                    adj[(u, v)] = c
                    adj[(v, u)] = c
                    neigh[u].add(v)
                    neigh[v].add(u)
                    key = (u, v) if u < v else (v, u)
                    color_of[key] = c
                    break
        
        if len(base_edges) > best_len:
            best_len = len(base_edges)
            best_edges = base_edges

    if best_edges is None:
        # Fallback: star
        return [[0, v, 0] for v in range(1, N)]
    return best_edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
