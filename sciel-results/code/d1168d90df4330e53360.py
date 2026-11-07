"""Packing edge-disjoint Hamiltonian cycles on the 21-vertex host.
Combines iterative backtracking (Warnsdorff + degree pruning + forced moves)
with a max-flow 2-factor feasibility check to prune dead-end branches early.
"""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _build_adj(edge_list, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _cycle_edges(cycle):
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.add(_edge_key(u, v))
    return edges


def _validate(cycles, n, edge_set):
    if not cycles:
        return False
    used_edges = set()
    all_vertices = None
    for cycle in cycles:
        if len(cycle) != n:
            return False
        if len(set(cycle)) != n:
            return False
        verts = set(cycle)
        if all_vertices is None:
            all_vertices = verts
        elif verts != all_vertices:
            return False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in edge_set:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
    return True


def _two_factor_exists(adj, n, deadline, max_check_time=0.05):
    """Check if the graph has a 2-factor (cycle cover) via max flow.
    This is a necessary condition for a Hamiltonian cycle.
    Uses bipartite matching: each vertex needs degree exactly 2.
    Model: split each vertex v into v_in and v_out. Edge (u,v) becomes
    edge from u_out to v_in with capacity 1. Each v_in has demand 2,
    each v_out has supply 2. Check if max flow = 2n.
    Returns True if a 2-factor exists.
    """
    start = time.time()
    # Build flow network
    # Nodes: 0 = source, 1 = sink
    # v_out = 2 + v, v_in = 2 + n + v
    N = 2 + 2 * n
    src = 0
    sink = 1
    # Adjacency for flow (list of [to, cap, rev_index])
    flow_adj = [[] for _ in range(N)]

    def add_edge(u, v, cap):
        flow_adj[u].append([v, cap, len(flow_adj[v])])
        flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

    # Source to each v_out: capacity 2
    for v in range(n):
        add_edge(src, 2 + v, 2)
    # Each v_in to sink: capacity 2
    for v in range(n):
        add_edge(2 + n + v, sink, 2)
    # For each graph edge (u,v), add u_out -> v_in and v_out -> u_in with cap 1
    for u in range(n):
        for v in adj[u]:
            if u < v:
                add_edge(2 + u, 2 + n + v, 1)
                add_edge(2 + v, 2 + n + u, 1)

    # Max flow via Edmonds-Karp (BFS)
    def max_flow():
        total = 0
        while True:
            if time.time() - start > max_check_time:
                return None  # timeout
            # BFS to find augmenting path
            parent = [-1] * N
            parent_edge = [-1] * N
            parent[src] = src
            q = deque([src])
            while q:
                u = q.popleft()
                for i, (v, cap, _) in enumerate(flow_adj[u]):
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        parent_edge[v] = i
                        if v == sink:
                            break
                        q.append(v)
                if parent[sink] != -1:
                    break
            if parent[sink] == -1:
                break
            # Find bottleneck
            bottleneck = float('inf')
            v = sink
            while v != src:
                u = parent[v]
                idx = parent_edge[v]
                bottleneck = min(bottleneck, flow_adj[u][idx][1])
                v = u
            # Update residual
            v = sink
            while v != src:
                u = parent[v]
                idx = parent_edge[v]
                flow_adj[u][idx][1] -= bottleneck
                rev = flow_adj[u][idx][2]
                flow_adj[v][rev][1] += bottleneck
                v = u
            total += bottleneck
        return total

    result = max_flow()
    if result is None:
        return True  # assume exists on timeout (don't prune)
    return result == 2 * n


def _find_hamiltonian_cycle(adj, n, rng, deadline, start=0, max_nodes=500000):
    """Find a Hamiltonian cycle using backtracking with Warnsdorff,
    degree pruning, and forced moves."""
    # Connectivity check
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != n:
        return None

    # Precompute sorted neighbor lists
    nbrs = [sorted(adj[v]) for v in range(n)]
    adj_set = [set(adj[v]) for v in range(n)]

    path = [start]
    visited = [False] * n
    visited[start] = True
    nodes_explored = 0

    def warnsdorff_order(cur):
        """Return unvisited neighbors of cur sorted by onward degree."""
        unvisited = [nb for nb in nbrs[cur] if not visited[nb]]
        if not unvisited:
            return []
        scored = []
        for nb in unvisited:
            # Count onward options for nb (excluding cur and visited)
            onward = 0
            for x in nbrs[nb]:
                if not visited[x] and x != cur:
                    onward += 1
            # If nb is the last node to place, it must connect to start
            if len(path) == n - 1:
                if start in adj_set[nb]:
                    onward = 0  # force this if possible
                else:
                    onward = 999  # invalid
            scored.append((onward, nb))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [nb for _, nb in scored]

    def check_degree_pruning():
        """Check that every unvisited vertex has at least 2 available edges
        to unvisited vertices or to the path endpoints (start and current end)."""
        for v in range(n):
            if visited[v]:
                continue
            # Count available edges: to unvisited vertices, plus to start and current end
            avail = 0
            cur_end = path[-1]
            for nb in nbrs[v]:
                if not visited[nb]:
                    avail += 1
                elif nb == start:
                    avail += 1
                elif nb == cur_end:
                    avail += 1
            # For a Hamiltonian cycle, each vertex needs exactly 2 edges.
            # Unvisited vertices need at least 2 available connections.
            if avail < 2:
                return False
        return True

    def find_forced_move():
        """Find an unvisited vertex with exactly 1 available neighbor among
        unvisited vertices. If such a vertex exists and its forced neighbor
        is the current end, we must go there."""
        cur_end = path[-1]
        forced = None
        for v in range(n):
            if visited[v]:
                continue
            # Count unvisited neighbors
            unv_nbrs = [nb for nb in nbrs[v] if not visited[nb]]
            if len(unv_nbrs) == 1:
                forced = unv_nbrs[0]
                break
        if forced is not None:
            # The forced vertex must be reachable from cur_end
            if forced in adj_set[cur_end]:
                return forced
        return None

    # Iterative backtracking
    # Stack entries: (node, candidate_list, index)
    stack = [(start, warnsdorff_order(start), 0)]

    while stack:
        if time.time() > deadline:
            return None
        nodes_explored += 1
        if nodes_explored > max_nodes:
            return None

        node, candidates, idx = stack[-1]

        if idx >= len(candidates):
            # Backtrack
            stack.pop()
            if not stack:
                return None
            # Remove node from path
            path.pop()
            visited[node] = False
            # Re-compute candidates for new top
            if stack:
                parent_node = stack[-1][0]
                new_cands = warnsdorff_order(parent_node)
                stack[-1] = (parent_node, new_cands, 0)
            continue

        next_node = candidates[idx]
        # Advance index
        stack[-1] = (node, candidates, idx + 1)

        if visited[next_node]:
            continue

        # Check if this completes the cycle
        if len(path) == n - 1:
            if start in adj_set[next_node]:
                return path + [next_node]
            continue

        # Try the move
        visited[next_node] = True
        path.append(next_node)

        # Degree pruning check
        if not check_degree_pruning():
            visited[next_node] = False
            path.pop()
            continue

        # Forced move check
        forced = find_forced_move()
        if forced is not None and forced != next_node:
            # We must go to forced from next_node
            if forced in adj_set[next_node]:
                if not visited[forced]:
                    visited[forced] = True
                    path.append(forced)
                    # Check degree pruning again
                    if not check_degree_pruning():
                        visited[forced] = False
                        path.pop()
                        visited[next_node] = False
                        path.pop()
                        continue
                    # Continue with forced node
                    cands = warnsdorff_order(forced)
                    stack.append((forced, cands, 0))
                    continue
                else:
                    # Forced node already visited - contradiction
                    visited[next_node] = False
                    path.pop()
                    continue
            else:
                # Can't reach forced from here - dead end
                visited[next_node] = False
                path.pop()
                continue

        # Normal case: continue from next_node
        cands = warnsdorff_order(next_node)
        stack.append((next_node, cands, 0))

    return None


def _find_hc_multi(adj, n, rng, deadline):
    """Try multiple starting points."""
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts:
        if time.time() > deadline:
            return None
        result = _find_hamiltonian_cycle(adj, n, rng, deadline, start=start)
        if result is not None:
            return result
    return None


def _search_cycles(edges, n, rng, deadline, target=3):
    """Search for up to `target` edge-disjoint Hamiltonian cycles."""
    original_edges = set()
    for u, v in edges:
        original_edges.add(_edge_key(u, v))

    cycles = []
    remaining = set(original_edges)

    for cidx in range(target):
        if time.time() > deadline:
            break

        edge_list = list(remaining)
        adj = _build_adj(edge_list, n)

        # Check min degree
        min_deg = min(len(adj[v]) for v in range(n)) if adj else 0
        if min_deg < 2:
            break

        # Quick 2-factor feasibility check
        if not _two_factor_exists(adj, n, deadline, max_check_time=0.03):
            break

        found = None
        # Multiple attempts with different randomizations
        max_attempts = 300
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            # Shuffle neighbor orderings for diversity
            shuffled_adj = [set(adj[v]) for v in range(n)]
            # Perturb ordering by re-sorting with random tie-breaking
            for v in range(n):
                lst = list(shuffled_adj[v])
                rng.shuffle(lst)
                shuffled_adj[v] = set(lst)

            # Try with a random start
            start = rng.randint(0, n - 1)
            cycle = _find_hamiltonian_cycle(shuffled_adj, n, rng, deadline, start=start)
            if cycle is not None:
                found = cycle
                break

        if found is None:
            # Try multi-start with original adj
            found = _find_hc_multi(adj, n, rng, deadline)

        if found is None:
            break

        # Validate the cycle
        if len(found) != n or len(set(found)) != n:
            break
        cycle_ek = _cycle_edges(found)
        if not cycle_ek.issubset(remaining):
            break

        cycles.append(found)

        # Remove used edges
        for ek in cycle_ek:
            remaining.discard(ek)

    return cycles


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))

    best = []

    # Try many random seeds
    for seed in range(5000):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        cycles = _search_cycles(edges, n, rng, deadline, target=3)
        if len(cycles) > len(best):
            best = [c[:] for c in cycles]
            if len(best) == 3 and _validate(best, n, edge_set):
                return [[int(v) for v in c] for c in best]

    if len(best) == 3 and _validate(best, n, edge_set):
        return [[int(v) for v in c] for c in best]

    if best:
        return [[int(v) for v in c] for c in best]

    # Fallback
    return [list(range(n))]
# EVOLVE-BLOCK-END
