"""Improved decomposition into cycles and edges."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

DEADLINE_MARGIN = 12  # seconds before 150s hard limit

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def cycle_edges(cycle):
    """Given a list of vertices forming a cycle (>=3 distinct), return set of edges."""
    if len(cycle) < 3:
        return set()
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        e = (u, v) if u < v else (v, u)
        edges.add(e)
    return edges

def find_cycle_dfs(remaining, adj, start, max_len, rng):
    """Find a cycle starting and ending at start using DFS. Returns list of vertices or None."""
    if start not in adj or not adj[start]:
        return None
    visited = {start}
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if len(path) > max_len:
            continue
        neighbors = list(adj.get(node, set()))
        rng.shuffle(neighbors)
        for nb in neighbors:
            e = (node, nb) if node < nb else (nb, node)
            if e not in remaining:
                continue
            if nb == start and len(path) >= 3:
                return path
            if nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb]))
    return None

def decompose_greedy(edges, max_time, num_restarts=20):
    """
    Decompose edges into fewest pieces using randomized greedy cycle extraction.
    Each piece is either a cycle (>=3 distinct vertices) or a single edge.
    """
    start_time = time.time()
    all_edges_set = set(tuple(sorted(e)) for e in edges)
    best_pieces = None
    best_count = len(edges) + 1

    def try_decomposition(seed_idx, max_len):
        nonlocal best_pieces, best_count
        if time.time() - start_time > max_time:
            return False

        rng = Random(seed_idx)
        remaining = set(all_edges_set)
        adj = build_adj(list(remaining))
        pieces = []

        while remaining:
            if time.time() - start_time > max_time:
                return False

            # Try to find a cycle
            found_cycle = None
            nodes = list(set([u for (u, v) in remaining] + [v for (u, v) in remaining]))
            rng.shuffle(nodes)

            # Try multiple starting nodes and attempts
            for _attempt in range(8):
                if not remaining:
                    break
                start_node = rng.choice(nodes)
                cyc = find_cycle_dfs(remaining, adj, start_node, max_len, rng)
                if cyc and len(cyc) >= 3:
                    found_cycle = cyc
                    break

            if found_cycle:
                cyc_edges = cycle_edges(found_cycle)
                if cyc_edges <= remaining:
                    pieces.append(found_cycle[:])
                    for e in cyc_edges:
                        remaining.discard(e)
                    adj = build_adj(list(remaining))
                else:
                    # Fallback: take a single edge
                    e = rng.choice(list(remaining))
                    pieces.append(list(e))
                    remaining.discard(e)
                    adj = build_adj(list(remaining))
            else:
                # No cycle found; take a single edge
                if remaining:
                    e = rng.choice(list(remaining))
                    pieces.append(list(e))
                    remaining.discard(e)
                    adj = build_adj(list(remaining))

        if len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = [p[:] for p in pieces]
        return time.time() - start_time < max_time

    # Try different max_len values and multiple seeds
    max_lens = [8, 12, 16, 20, 30]
    for ml in max_lens:
        if time.time() - start_time > max_time:
            break
        n_seeds = max(1, num_restarts // len(max_lens))
        for s in range(n_seeds):
            if time.time() - start_time > max_time:
                break
            try_decomposition(ml * 1000 + s, ml)

    if best_pieces is None:
        best_pieces = [list(e) for e in edges]

    return best_pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    per_graph_time = max(1.0, (DEADLINE_MARGIN - 3) / 3)

    for i, spec in enumerate(specs):
        edges = instance(*spec)
        cur_time = time.time()
        remaining_time = min(per_graph_time, DEADLINE_MARGIN - (cur_time - start))
        if remaining_time < 1.0:
            results.append([list(e) for e in edges])
            continue

        pieces = decompose_greedy(edges, max_time=remaining_time, num_restarts=25)
        results.append(pieces)

    # Ensure exactly 3 results
    while len(results) < 3:
        results.append([])

    return results[:3]

# EVOLVE-BLOCK-END
