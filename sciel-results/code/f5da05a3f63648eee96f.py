"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 40

# EVOLVE-BLOCK-START

def _build_tree(seed=0):
    """Build a random tree on N vertices. Returns (adj, parent, depth, children)."""
    rng = random.Random(seed)
    # Build a random spanning tree via random growing
    parent = [-1] * N
    depth = [0] * N
    children = [[] for _ in range(N)]
    parent[0] = -1
    depth[0] = 0
    frontier = [0]
    unvisited = set(range(1, N))
    rng.shuffle(frontier)
    while frontier:
        u = frontier.pop(0)
        # pick a random unvisited vertex to attach
        if unvisited:
            v = rng.choice(list(unvisited))
            unvisited.discard(v)
            parent[v] = u
            depth[v] = depth[u] + 1
            children[u].append(v)
            frontier.append(v)
    return parent, depth, children


def _compute_paths(parent, depth, children):
    """For each vertex, compute the path from it to root (list of vertices)."""
    paths = {}
    def dfs(u, current):
        paths[u] = current + [u]
        for c in children[u]:
            dfs(c, paths[u])
    dfs(0, [])
    return paths


def _find_back_edge_for_length(L, paths, adj_set, parent, depth, children):
    """Find a back-edge (v, ancestor) such that the cycle formed has exactly L vertices.
    
    A back-edge from v to its ancestor a (where a is on the path from v to root)
    creates a cycle of length (depth[v] - depth[a] + 1) if we count vertices on the
    path from a to v plus the back-edge.
    
    Actually, the cycle is: a -> ... -> v -> a (back-edge). The number of vertices
    in this cycle is (depth[v] - depth[a] + 1).
    
    So we need depth[v] - depth[a] + 1 = L, i.e., depth[v] - depth[a] = L - 1.
    """
    needed = L - 1  # depth difference
    for v in range(N):
        d_v = depth[v]
        if d_v < needed:
            continue
        # Find ancestor a at depth d_v - needed
        target_depth = d_v - needed
        # Walk up from v to find ancestor at target_depth
        cur = v
        while depth[cur] > target_depth:
            cur = parent[cur]
        if depth[cur] == target_depth:
            # Check that edge (v, cur) is not already a tree edge
            if cur != parent[v]:  # Not a tree edge
                edge = (min(v, cur), max(v, cur))
                if edge not in adj_set:
                    return (v, cur), edge
    return None, None


def _verify_cycle(cycle, adj_set):
    """Verify that the given cycle (list of vertices) is valid: all consecutive pairs
    and last-first pair are edges, and all vertices are distinct."""
    L = len(cycle)
    if L < 3:
        return False
    if len(set(cycle)) != L:
        return False
    for i in range(L):
        u = cycle[i]
        v = cycle[(i + 1) % L]
        edge = (min(u, v), max(u, v))
        if edge not in adj_set:
            return False
    return True


def _get_cycle_from_back_edge(v, a, paths):
    """Get the cycle vertex list formed by back-edge (v, a) where a is ancestor of v."""
    path = paths[v]  # path from root to v
    # Find a in path
    try:
        idx = path.index(a)
    except ValueError:
        return None
    # Cycle is: a, path[idx+1], ..., path[-1]=v, then back to a
    cycle = path[idx:]
    return cycle


def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    best_edges = None
    best_cycles = None
    best_count = float('inf')

    def time_ok():
        return time.time() - start_time < deadline - 5

    # Try multiple random trees
    for tree_seed in range(50):
        if not time_ok():
            break

        parent, depth, children = _build_tree(tree_seed)
        paths = _compute_paths(parent, depth, children)

        # Tree edges
        tree_edges = set()
        for v in range(1, N):
            u = parent[v]
            tree_edges.add((min(u, v), max(u, v)))

        # For each length L in 3..40, find a back-edge
        back_edges = {}
        cycles = {}
        success = True
        adj_set = set(tree_edges)

        for L in range(3, N + 1):
            if not time_ok():
                success = False
                break
            (v, a), edge = _find_back_edge_for_length(L, paths, adj_set, parent, depth, children)
            if edge is None:
                success = False
                break
            back_edges[L] = edge
            adj_set.add(edge)
            cycle = _get_cycle_from_back_edge(v, a, paths)
            if cycle is None:
                success = False
                break
            cycles[L] = cycle

        if not success:
            continue

        # Verify all cycles
        all_valid = True
        for L in range(3, N + 1):
            if not _verify_cycle(cycles[L], adj_set):
                all_valid = False
                break
        if not all_valid:
            continue

        total_edges = len(adj_set)
        if total_edges < best_count:
            best_count = total_edges
            best_edges = sorted(adj_set)
            best_cycles = [cycles[L] for L in range(3, N + 1)]

        if best_count <= 40 + 19:  # Can't do better than 59 easily
            break

    # If no valid construction found, use the proven 59-edge construction
    if best_edges is None:
        edges = set()
        for i in range(N):
            edges.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
        for k in range(2, 21):
            edges.add((0, k))
        best_edges = sorted(edges)
        best_cycles = []
        for L in range(3, N + 1):
            if L == N:
                best_cycles.append(list(range(N)))
            else:
                k1 = L - 1
                if 2 <= k1 <= 20:
                    best_cycles.append(list(range(0, k1 + 1)))
                else:
                    k2 = N - L + 1
                    if 2 <= k2 <= 20:
                        best_cycles.append([0] + list(range(k2, N)))
                    else:
                        # Should not happen
                        best_cycles.append(list(range(0, L)))

    return {'edges': best_edges, 'cycles': best_cycles}

# EVOLVE-BLOCK-END
