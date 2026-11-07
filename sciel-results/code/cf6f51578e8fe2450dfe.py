"""Triangle-free diameter-four augmentation with simulated annealing."""
import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_base_adj(n, base_edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj


def _bfs_all_dists(n, adj):
    """Return list of distance arrays from each source."""
    dists = []
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        dists.append(dist)
    return dists


def _max_diameter(dists):
    md = 0
    for dist in dists:
        for d in dist:
            if d == -1:
                return -1
            if d > md:
                md = d
    return md


def _is_triangle_free(adj, n, added_edges):
    """Check if adding these edges to base keeps triangle-free."""
    adj2 = [row[:] for row in adj]
    for u, v in added_edges:
        if adj2[u][v]:
            return False
        for w in range(n):
            if w != u and w != v and adj2[u][w] and adj2[v][w]:
                return False
        adj2[u][v] = True
        adj2[v][u] = True
    return True


def _diameter_with_edges(n, base_adj, added_edges):
    """Compute diameter after adding edges; return -1 if invalid."""
    if not _is_triangle_free(base_adj, n, added_edges):
        return -1
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        adj[u][v] = True
        adj[v][u] = True
    dists = _bfs_all_dists(n, adj)
    return _max_diameter(dists)


def _precompute_valid_pairs(n, base_adj):
    """Find all pairs that can be added without creating a triangle."""
    valid = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v]:
                # Check no common neighbor
                ok = True
                for w in range(n):
                    if w != u and w != v and base_adj[u][w] and base_adj[v][w]:
                        ok = False
                        break
                if ok:
                    valid.append((u, v))
    return valid


def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])

    # Precompute all valid single-edge additions
    valid_pairs = _precompute_valid_pairs(n, base_adj)
    valid_set = set(valid_pairs)

    # Initial solution: known 8-edge solution
    seed_edges = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    best_edges = list(seed_edges)
    best_diam = _diameter_with_edges(n, base_adj, best_edges)

    # Simulated annealing: try to minimize number of edges
    # State: set of edges. Move: remove one edge, or swap one edge.
    random.seed(42)

    # First, try greedy reduction: can we remove any edge and still have diam <= 4?
    for attempt in range(200):
        if time.time() > deadline:
            break
        current = list(best_edges)
        improved = False
        for i in range(len(current)):
            candidate = [e for j, e in enumerate(current) if j != i]
            d = _diameter_with_edges(n, base_adj, candidate)
            if 0 <= d <= 4:
                best_edges = candidate
                best_diam = d
                improved = True
                break
        if not improved:
            break

    # Simulated annealing to try to find smaller solutions
    if len(best_edges) > 5:
        current = list(best_edges)
        current_diam = _diameter_with_edges(n, base_adj, current)
        T = 1.0
        cooling = 0.995
        min_T = 0.01

        while T > min_T and time.time() < deadline:
            # Choose a move
            move_type = random.random()
            if move_type < 0.4 and len(current) > 0:
                # Remove a random edge
                idx = random.randint(0, len(current) - 1)
                new_edges = [e for j, e in enumerate(current) if j != idx]
            elif move_type < 0.8 and len(current) > 0:
                # Swap: remove one, add one
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                new_edges = [e for j, e in enumerate(current) if j != idx]
                # Add a random valid edge not already present
                current_set = set(new_edges)
                candidates = [p for p in valid_pairs if p not in current_set]
                if candidates:
                    add_edge = random.choice(candidates)
                    new_edges.append(add_edge)
                else:
                    continue
            else:
                # Add a random edge
                current_set = set(current)
                candidates = [p for p in valid_pairs if p not in current_set]
                if candidates:
                    add_edge = random.choice(candidates)
                    new_edges = current + [add_edge]
                else:
                    continue

            new_diam = _diameter_with_edges(n, base_adj, new_edges)

            # Cost: prefer fewer edges, then valid diameter
            old_cost = len(current) * 100 + (0 if (0 <= current_diam <= 4) else 1000)
            new_cost = len(new_edges) * 100 + (0 if (0 <= new_diam <= 4) else 1000)
            delta = new_cost - old_cost

            if delta <= 0 or random.random() < min(1.0, max(0.0, 1.0 - delta / (T * 100))):
                current = new_edges
                current_diam = new_diam
                if 0 <= current_diam <= 4 and len(current) < len(best_edges):
                    best_edges = list(current)
                    best_diam = current_diam

            T *= cooling

    # Final greedy local search
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        # Try removing edges
        for i in range(len(best_edges)):
            candidate = [e for j, e in enumerate(best_edges) if j != i]
            d = _diameter_with_edges(n, base_adj, candidate)
            if 0 <= d <= 4:
                best_edges = candidate
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add another
        for i in range(len(best_edges)):
            if time.time() > deadline:
                break
            base = [e for j, e in enumerate(best_edges) if j != i]
            base_set = set(base)
            for cand in valid_pairs:
                if cand in base_set:
                    continue
                new_e = base + [cand]
                d = _diameter_with_edges(n, base_adj, new_e)
                if 0 <= d <= 4 and len(new_e) < len(best_edges):
                    best_edges = new_e
                    improved = True
                    break
            if improved:
                break

    # Validate final answer
    final_diam = _diameter_with_edges(n, base_adj, best_edges)
    if final_diam < 0 or final_diam > 4:
        # Fallback to seed
        best_edges = seed_edges

    return [list(e) for e in best_edges]
