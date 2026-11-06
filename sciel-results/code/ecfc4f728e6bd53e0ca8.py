"""
Decomposition of three fixed graphs into few cycles and edges (Erdős-Gallai).

Strategy:
1. Multi-start greedy longest-path cycle extraction (from inspiration 1, score 8.27).
   Each restart uses a different random seed to explore different cycle orderings.
2. After greedy extraction, attempt to promote remaining single edges into cycles
   by searching for triangles and 4-cycles among the leftover edges.
3. Keep the best decomposition across all restarts.

The key insight from the top-scoring inspiration is that randomized greedy
longest-path heuristics with many restarts find significantly better cycle
decompositions than deterministic DFS-based approaches. The current program's
triangle/4-cycle promotion is a useful refinement that captures short cycles
the greedy finder might miss.
"""

import time
import random
from collections import defaultdict


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_longest_cycle_greedy(adj, remaining, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not remaining:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    rng.shuffle(vertices)

    # Try multiple starting vertices (top by degree + random)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:5] + vertices[:5]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Greedy longest path from start: always extend to unvisited neighbor
        # with highest degree (keeps options open for closing the cycle)
        path = [start]
        visited = {start}

        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u]
                          if v not in visited
                          and (min(u, v), max(u, v)) in remaining]
            if not candidates:
                break
            candidates.sort(reverse=True)
            # Randomize among top-3 to add diversity
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        # Check if we can close the cycle directly
        if len(path) >= 3:
            last = path[-1]
            if (min(start, last), max(start, last)) in remaining and start in adj[last]:
                if len(path) > len(best_cycle):
                    best_cycle = list(path)

        # Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
        # Take the longest such cycle
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                u_i, v_j = path[i], path[j]
                e = (min(u_i, v_j), max(u_i, v_j))
                if e in remaining:
                    cycle = path[i:j + 1]
                    if len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Only the longest j for this i

    return best_cycle


def _try_promote_cycles(remaining, adj, deadline):
    """
    Try to find triangles and 4-cycles among remaining edges to reduce piece count.
    Returns a list of cycles found and updates remaining/adj in place.
    """
    promoted = []
    
    # Build edge lookup for fast access
    edge_set = set(remaining)
    
    # Try to find triangles
    for e1 in list(edge_set):
        if e1 not in edge_set:
            continue
        u1, v1 = e1
        # Find edges sharing u1
        for e2 in list(edge_set):
            if e2 == e1 or e2 not in edge_set:
                continue
            u2, v2 = e2
            # Check if e1 and e2 share a vertex
            shared = None
            other1 = None
            other2 = None
            if u2 == u1:
                shared = u1
                other1 = v1
                other2 = v2
            elif v2 == u1:
                shared = u1
                other1 = v1
                other2 = u2
            elif u2 == v1:
                shared = v1
                other1 = u1
                other2 = v2
            elif v2 == v1:
                shared = v1
                other1 = u1
                other2 = u2
            else:
                continue
            
            if other1 == other2:
                continue
            
            e3 = (min(other1, other2), max(other1, other2))
            if e3 in edge_set and e3 != e1 and e3 != e2:
                # Found a triangle
                promoted.append([shared, other1, other2])
                edge_set.discard(e1)
                edge_set.discard(e2)
                edge_set.discard(e3)
                # Update adj
                for a, b in [(shared, other1), (other1, other2), (other2, shared)]:
                    if b in adj[a]:
                        adj[a].discard(b)
                    if a in adj[b]:
                        adj[b].discard(a)
                    if not adj[a]:
                        del adj[a]
                    if not adj[b]:
                        del adj[b]
                break
        if e1 not in edge_set:
            continue
        
        # Try to find 4-cycles
        u1, v1 = e1
        for e2 in list(edge_set):
            if e2 == e1 or e2 not in edge_set:
                continue
            u2, v2 = e2
            shared = None
            other1 = None
            other2 = None
            if u2 == u1:
                shared = u1
                other1 = v1
                other2 = v2
            elif v2 == u1:
                shared = u1
                other1 = v1
                other2 = u2
            elif u2 == v1:
                shared = v1
                other1 = u1
                other2 = v2
            elif v2 == v1:
                shared = v1
                other1 = u1
                other2 = u2
            else:
                continue
            
            if other1 == other2:
                continue
            
            # Look for a path of length 2 from other1 to other2 through some vertex w
            for w in list(adj.get(other1, set())):
                if w == shared or w == other2:
                    continue
                e3 = (min(other1, w), max(other1, w))
                e4 = (min(w, other2), max(w, other2))
                if e3 in edge_set and e4 in edge_set and e3 != e1 and e4 != e2:
                    # Found a 4-cycle: shared - other1 - w - other2 - shared
                    promoted.append([shared, other1, w, other2])
                    edge_set.discard(e1)
                    edge_set.discard(e2)
                    edge_set.discard(e3)
                    edge_set.discard(e4)
                    for a, b in [(shared, other1), (other1, w), (w, other2), (other2, shared)]:
                        if b in adj[a]:
                            adj[a].discard(b)
                        if a in adj[b]:
                            adj[b].discard(a)
                        if not adj[a]:
                            del adj[a]
                        if not adj[b]:
                            del adj[b]
                    break
            if e1 not in edge_set:
                break
        if e1 not in edge_set:
            continue
    
    # Update remaining
    remaining.clear()
    remaining.update(edge_set)
    return promoted


def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    deadline = start_time + time_limit

    def greedy_decompose(remaining_edges, seed_offset):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        cycle_counter = 0
        while remaining:
            # Find a long cycle
            cycle = _find_longest_cycle_greedy(adj, remaining, seed + seed_offset * 1000 + cycle_counter, deadline)
            if cycle and len(cycle) >= 3:
                # Extract cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)

                # Remove from remaining and adj
                for e in cycle_edges:
                    if e in remaining:
                        remaining.discard(e)
                        u, v = e
                        adj[u].discard(v)
                        adj[v].discard(u)
                        if not adj[u]:
                            del adj[u]
                        if not adj[v]:
                            del adj[v]

                pieces.append(cycle)
                cycle_counter += 1
            else:
                # No cycle found; remove a single edge
                # Prefer edges where both endpoints have low degree
                best_edge = None
                best_priority = -1
                for u, v in remaining:
                    deg_u = len(adj[u])
                    deg_v = len(adj[v])
                    priority = -(deg_u + deg_v)
                    if priority > best_priority:
                        best_priority = priority
                        best_edge = (u, v)

                if best_edge is None:
                    best_edge = next(iter(remaining))

                u, v = best_edge
                remaining.discard(best_edge)
                adj[u].discard(v)
                adj[v].discard(u)
                if not adj[u]:
                    del adj[u]
                if not adj[v]:
                    del adj[v]
                pieces.append([u, v])

        # Try to promote remaining singles into cycles
        if remaining and time.time() < deadline:
            promoted = _try_promote_cycles(remaining, adj, deadline)
            for cyc in promoted:
                pieces.append(cyc)
            # Add remaining as singles
            for e in sorted(remaining):
                pieces.append([e[0], e[1]])

        return pieces

    # Multi-start search with diverse strategies
    num_restarts = 300
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        pieces = greedy_decompose(edges, restart)
        current_count = len(pieces)

        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces

    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_deadline = time.time() + 140.0  # safety margin before 150s kill
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Allocate time proportionally, with more time for larger graphs
        remaining_time = total_deadline - time.time()
        graphs_left = 3 - i
        time_limit = max(5.0, remaining_time / graphs_left - 5.0)
        pieces = _decompose(edges, n, seed, time_limit=time_limit)
        results.append(pieces)
    
    return results
