import time
import random

def _deadline_hit(deadline):
    return time.time() > deadline

def _verify_c4_free(adj, n):
    """Check if graph with adjacency bitmasks is C4-free."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] & adj[v]).bit_count() > 1:
                return False
    return True

def _max_independent_set(adj, n, deadline):
    """
    Find a maximum independent set using bitmask backtracking.
    Returns (size, set_bitmask).
    """
    best_size = 0
    best_set = 0

    def backtrack(remaining, current_size, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
        if not remaining:
            return
        if current_size + remaining.bit_count() <= best_size:
            return

        # Heuristic: pick vertex with minimum degree in remaining subgraph
        # among the remaining vertices
        min_v = -1
        min_deg = 100
        temp = remaining
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp &= temp - 1

        if min_v == -1:
            return

        # Branch 1: Include min_v
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        backtrack(new_remaining, current_size + 1, current_set | (1 << min_v))

        if time.time() > deadline:
            return

        # Branch 2: Exclude min_v
        backtrack(remaining & ~(1 << min_v), current_size, current_set)

    backtrack((1 << n) - 1, 0, 0)
    return best_size, best_set

def _has_indep_11(adj, n, deadline):
    """Check if there exists an independent set of size 11."""
    size, _ = _max_independent_set(adj, n, deadline)
    return size >= 11

def _find_indep_11(adj, n, deadline):
    """
    Find an independent set of size >= 11, or return None if none exists.
    Returns the set as a bitmask.
    """
    size, set_mask = _max_independent_set(adj, n, deadline)
    if size >= 11:
        return set_mask
    return None

def _can_add_edge(adj, n, u, v):
    """
    Check if adding edge (u, v) keeps the graph C4-free.
    adj: list of int bitmasks
    u, v: vertices not already adjacent
    """
    if u == v:
        return False
    # u and v not already connected
    if (adj[u] >> v) & 1:
        return False
    # |N(u) & N(v)| <= 1
    common = adj[u] & adj[v]
    if common.bit_count() > 1:
        return False
    # For all w in N(u), N(w) & N(v) must be empty
    # This ensures no path u-w-x-v exists which would form C4
    nu = adj[u]
    nv = adj[v]

    temp = nu
    while temp:
        w = (temp & -temp).bit_length() - 1
        if adj[w] & nv:
            return False
        temp &= temp - 1

    temp = nv
    while temp:
        x = (temp & -temp).bit_length() - 1
        if adj[x] & nu:
            return False
        temp &= temp - 1

    return True

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _construct_polarity_graph_q(q):
    """
    Construct polarity graph of F_q (q a prime power, using prime for simplicity).
    Vertices: (x, y) for x, y in F_q, total q^2 vertices.
    Edge between (x1, y1) and (x2, y2) if x1*x2 + y1*y2 == 0.
    This graph is C4-free and has alpha ~ q + sqrt(q).
    """
    if q == 2:
        # 4 vertices, cycle C4? No, polarity graph of F_2:
        # Vertices: 00, 01, 10, 11
        # 00 adjacent to all (0*any+0*any=0)
        # 01: 0*x+1*y=y, adjacent if y=0, so adjacent to 00, 10
        # 10: 1*x+0*y=x, adjacent if x=0, so adjacent to 00, 01
        # 11: 1*x+1*y=x+y, adjacent if x+y=0, so adjacent to 00, 11 (self?)
        # Actually self-adjacency: 1*1+1*1=0 in F_2, so self-loop
        # Let me just hardcode for small cases
        pass

    # For q=5: 25 vertices
    # For q=7: 49 vertices (too many, we want ~39)
    # Let's use q=5 (25 vertices) and add more

    n = q * q
    adj = [0] * n

    def idx(x, y):
        return x * q + y

    for x1 in range(q):
        for y1 in range(q):
            u = idx(x1, y1)
            for x2 in range(q):
                for y2 in range(q):
                    v = idx(x2, y2)
                    if u < v:
                        if (x1 * x2 + y1 * y2) % q == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
    return adj, n

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]

    best_result = {'n': 30, 'edges': fallback_edges}

    # Strategy: Start with polarity graph of F_5 (25 vertices, alpha ~ 7)
    # Then add vertices and edges to reach n=39 while keeping alpha < 11

    for q in [5]:
        if _deadline_hit(deadline):
            break

        adj, n_core = _construct_polarity_graph_q(q)
        n_target = 39
        n_add = n_target - n_core  # 14 new vertices

        # Initialize with core + isolated vertices
        adj = adj + [0] * n_add
        n = n_target

        # Check current alpha
        size, _ = _max_independent_set(adj, n, deadline)
        # Polarity graph of F_5 has alpha = 5 + 1 = 6? Actually alpha ~ sqrt(25) + 5 = 10?
        # Let me just check. For q=5, the polarity graph has 25 vertices, degree 5,
        # and alpha is at most 5 + sqrt(5) ~ 7. So with 14 isolated, alpha ~ 7 + 14 = 21.
        # We need to reduce this to < 11.

        # Add edges to reduce alpha
        # We'll try to add edges between new vertices and core, and among new vertices

        for attempt in range(200):
            if _deadline_hit(deadline - 5):
                break

            random.seed(attempt * 7919 + 31)

            # Reset to core
            adj = [0] * n_target
            adj_core, _ = _construct_polarity_graph_q(q)
            adj = adj_core + [0] * n_add

            # Generate all possible edges that can be added (not in core)
            # and shuffle
            all_new_edges = []
            for i in range(n_core, n_target):
                for j in range(i + 1, n_target):
                    all_new_edges.append((i, j))
            for i in range(n_core):
                for j in range(n_core, n_target):
                    all_new_edges.append((i, j))
            # Also edges within core that are not present
            for i in range(n_core):
                for j in range(i + 1, n_core):
                    if not ((adj[i] >> j) & 1):
                        all_new_edges.append((i, j))

            random.shuffle(all_new_edges)

            # Greedily add edges that are C4-safe and help reduce alpha
            for u, v in all_new_edges:
                if _deadline_hit(deadline - 5):
                    break
                if not _can_add_edge(adj, n_target, u, v):
                    continue
                # Add edge
                _add_edge(adj, u, v)

            # Check if alpha < 11
            size, _ = _max_independent_set(adj, n_target, deadline)
            if size < 11:
                # Verify C4-free
                if _verify_c4_free(adj, n_target):
                    # Extract edges
                    edges = []
                    for u in range(n_target):
                        for v in range(u + 1, n_target):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    return {'n': n_target, 'edges': edges}

    # If polarity graph approach didn't work, try random greedy as fallback
    for attempt in range(30):
        if _deadline_hit(deadline - 10):
            break

        random.seed(attempt * 12345 + 67890)
        n_target = 39
        adj = [0] * n_target
        edges = []

        all_edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        random.shuffle(all_edges)

        for u, v in all_edges:
            if _deadline_hit(deadline - 10):
                break
            if not _can_add_edge(adj, n_target, u, v):
                continue
            _add_edge(adj, u, v)
            edges.append([u, v])

        if not _verify_c4_free(adj, n_target):
            continue

        size, _ = _max_independent_set(adj, n_target, deadline)
        if size < 11:
            return {'n': n_target, 'edges': edges}

    return best_result
