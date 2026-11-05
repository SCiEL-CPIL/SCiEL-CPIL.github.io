import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(1000000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _eulerian_subgraph(adj, n):
    """
    Find a large connected subgraph with all even degrees.
    Uses parity matching: pair up odd-degree vertices with shortest paths.
    Returns (edge_set, vertices) or (None, None) if no edges.
    """
    # Find connected components
    visited = set()
    components = []
    for v in range(n):
        if v in visited or not adj[v]:
            continue
        # BFS
        comp = []
        stack = [v]
        visited.add(v)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
        components.append(comp)

    best_edges = set()
    best_size = 0

    for comp in components:
        if len(comp) < 3:
            continue
        comp_set = set(comp)

        # Compute degrees within this component
        deg = {v: 0 for v in comp}
        comp_edges = set()
        for v in comp:
            for w in adj[v]:
                if w in comp_set and v < w:
                    deg[v] += 1
                    deg[w] += 1
                    comp_edges.add((v, w))

        if not comp_edges:
            continue

        # Find odd-degree vertices
        odd = [v for v in comp if deg[v] % 2 == 1]

        if not odd:
            # Already Eulerian
            if len(comp_edges) > best_size:
                best_size = len(comp_edges)
                best_edges = set(comp_edges)
            continue

        # Pair up odd vertices with shortest paths
        # Build distance matrix for odd vertices
        odd_set = set(odd)
        # Multi-source BFS from each odd vertex
        dist = {}
        parent = {}
        for src in odd:
            d = {src: 0}
            par = {src: None}
            q = deque([src])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if w in comp_set and w not in d:
                        d[w] = d[u] + 1
                        par[w] = u
                        q.append(w)
            dist[src] = d
            parent[src] = par

        # Greedy pairing: pair closest odd vertices
        # Try multiple random orderings
        rng_local = random.Random(42)
        best_removed = None
        best_removed_size = 0

        for attempt in range(20):
            order = odd[:]
            rng_local.shuffle(order)
            removed_edges = set()
            used = set()
            success = True

            for i in range(0, len(order), 2):
                if i + 1 >= len(order):
                    break
                a, b = order[i], order[i + 1]
                if a in used or b in used:
                    continue
                # Find shortest path between a and b
                if b not in dist[a]:
                    success = False
                    break
                # Reconstruct path
                path = [b]
                cur = b
                while cur != a:
                    cur = parent[a][cur]
                    path.append(cur)
                path.reverse()
                # Add edges along path
                for k in range(len(path) - 1):
                    e = tuple(sorted((path[k], path[k + 1])))
                    removed_edges.add(e)
                used.add(a)
                used.add(b)

            if not success:
                continue

            # Result: comp_edges - removed_edges should have all even degrees
            result_edges = comp_edges - removed_edges
            # Verify
            d2 = {v: 0 for v in comp}
            for u, v in result_edges:
                d2[u] += 1
                d2[v] += 1
            if all(d2[v] % 2 == 0 for v in comp):
                if len(result_edges) > best_removed_size:
                    best_removed_size = len(result_edges)
                    best_removed = set(result_edges)

        if best_removed and len(best_removed) > best_size:
            best_size = len(best_removed)
            best_edges = best_removed

    return best_edges


def _hierholzer_cycles(edge_set, n):
    """
    Decompose an Eulerian graph (all even degrees) into cycles.
    Uses Hierholzer's algorithm.
    Returns list of cycles (each a list of vertices).
    """
    if not edge_set:
        return []

    # Build adjacency with edge tracking
    adj = defaultdict(list)
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)

    # Edge used tracking
    used_edges = set()
    cycles = []

    for start in range(n):
        if start not in adj:
            continue
        # Find unused edges from start
        while any(e not in used_edges for e in adj[start]):
            # Hierholzer
            stack = [(start, None)]
            cycle = []
            while stack:
                v, prev = stack.pop()
                # Find an unused edge from v
                next_v = None
                for w in adj[v]:
                    e = tuple(sorted((v, w)))
                    if e not in used_edges:
                        next_v = w
                        break
                if next_v is not None:
                    e = tuple(sorted((v, next_v)))
                    used_edges.add(e)
                    stack.append((v, next_v))
                    stack.append((next_v, v))
                else:
                    cycle.append(v)

            # Reverse to get proper order
            cycle.reverse()
            # Remove duplicates from the cycle (keep unique vertices in order)
            if len(cycle) >= 2:
                # The cycle is: cycle[0], cycle[1], ..., cycle[-1], cycle[0]
                # But Hierholzer may produce repeated vertices for non-simple cycles
                # We need to extract simple cycles
                # Actually, for an Eulerian decomposition, we get closed walks
                # Let's extract simple cycles from the closed walk
                if len(cycle) >= 2:
                    # Find unique cycles
                    seen = set()
                    simple_cycles = []
                    # The closed walk may traverse edges, let's just record it
                    # For scoring, we need simple cycles
                    # Let's extract simple cycles properly
                    # Actually, let me redo this with a proper approach
                    pass

    # Let me use a cleaner Hierholzer that produces simple cycles
    # Actually, for an Eulerian graph, Hierholzer produces edge-disjoint closed walks
    # that partition the edges. Each closed walk may not be a simple cycle.
    # We need to decompose each closed walk into simple cycles.

    # Let me redo: use a different approach
    # For each connected Eulerian component, find cycles by repeatedly
    # finding a cycle and removing it.

    adj2 = defaultdict(set)
    for u, v in edge_set:
        adj2[u].add(v)
        adj2[v].add(u)

    cycles = []
    remaining = set(edge_set)

    while remaining:
        # Find a cycle in the remaining graph
        # Start from any vertex with edges
        start = None
        for v in range(n):
            if adj2[v]:
                start = v
                break
        if start is None:
            break

        # DFS to find a cycle back to start
        stack = [(start, [start])]
        found = None
        visited_in_dfs = {start}

        while stack and not found:
            node, path = stack.pop()
            for neighbor in list(adj2[node]):
                if neighbor == start and len(path) >= 3:
                    found = path[:]
                    break
                elif neighbor not in visited_in_dfs:
                    visited_in_dfs.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))

        if found:
            m = len(found)
            for i in range(m):
                a = found[i]
                b = found[(i + 1) % m]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
                adj2[a].discard(b)
                adj2[b].discard(a)
            cycles.append(found)
        else:
            # Shouldn't happen for Eulerian, but fallback
            break

    return cycles


def _decompose_eulerian(edges, n, deadline):
    """
    Decompose graph into cycles using Eulerian subgraph approach.
    1. Find large Eulerian subgraphs
    2. Decompose each into cycles
    3. Remaining edges become single-edge pieces
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []

    while remaining:
        if time.time() > deadline:
            break

        # Find a large Eulerian subgraph
        euler_edges = _eulerian_subgraph(adj, n)

        if euler_edges and euler_edges <= remaining:
            # Decompose into cycles
            cycles = _hierholzer_cycles(euler_edges, n)
            for cycle in cycles:
                m = len(cycle)
                for i in range(m):
                    a = cycle[i]
                    b = cycle[(i + 1) % m]
                    e = tuple(sorted((a, b)))
                    remaining.discard(e)
                    adj[a].discard(b)
                    adj[b].discard(a)
                pieces.append(cycle)
        else:
            # No good Eulerian subgraph, try to find any cycle
            found = None
            for start in range(n):
                if not adj[start]:
                    continue
                stack = [(start, [start])]
                visited_in_dfs = {start}
                while stack and not found:
                    node, path = stack.pop()
                    for neighbor in list(adj[node]):
                        if neighbor == start and len(path) >= 3:
                            found = path[:]
                            break
                        elif neighbor not in visited_in_dfs:
                            visited_in_dfs.add(neighbor)
                            stack.append((neighbor, path + [neighbor]))
                if found:
                    break

            if found:
                m = len(found)
                for i in range(m):
                    a = found[i]
                    b = found[(i + 1) % m]
                    e = tuple(sorted((a, b)))
                    remaining.discard(e)
                    adj[a].discard(b)
                    adj[b].discard(a)
                pieces.append(found)
            else:
                # Take a single edge
                u, v = next(iter(remaining))
                remaining.discard((u, v))
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])

    # Add any remaining edges
    for e in remaining:
        pieces.append([e[0], e[1]])

    return pieces


def _decompose_greedy(edges, n, deadline, rng=None):
    """
    Greedy decomposition: repeatedly find a cycle and remove it.
    """
    if rng is None:
        rng = random.Random(12345)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    while remaining > 0:
        if time.time() > deadline:
            break

        # Try to find a long cycle
        best_cycle = None
        best_len = 0

        for start in range(n):
            if not adj[start]:
                continue
            stack = [(start, [start])]
            visited_in_dfs = {start}
            while stack:
                node, path = stack.pop()
                for neighbor in list(adj[node]):
                    if neighbor == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break
                    elif neighbor not in visited_in_dfs:
                        visited_in_dfs.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))

        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # Try Eulerian decomposition
    euler_deadline = deadline - 1.0
    euler_pieces = _decompose_eulerian(edges, n, euler_deadline)
    best_count = len(euler_pieces)
    best = euler_pieces

    # Try greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 0.5:
            break
        rng = random.Random(seed)
        pieces = _decompose_greedy(edges, n, deadline - 0.5, rng)
        if len(pieces) < best_count:
            best_count = len(pieces)
            best = pieces

    return best if best else [[u, v] for u, v in edges]


def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
