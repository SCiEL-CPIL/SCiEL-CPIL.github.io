import time
import random
import sys

def construct():
    deadline = time.time() + 140.0
    n = 16
    target_edges = 65
    rng = random.Random(12345)

    # Start from K_{8,8} plus one intra-part edge
    A = set(range(8))
    B = set(range(8, 16))

    def initial_edges():
        e = []
        for u in A:
            for v in B:
                e.append((u, v))
        e.append((0, 1))
        return e

    edges = initial_edges()

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deg = [len(adj[i]) for i in range(n)]

    def find_triangles():
        tris = []
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[u] & adj[v]:
                    if w > v:
                        tris.append((u, v, w))
        return tris

    def compute_score(tris):
        if not tris:
            return 0
        return max(deg[u] + deg[v] + deg[w] for u, v, w in tris)

    tris = find_triangles()
    best_score = compute_score(tris)
    best_edges = list(edges)

    # Precompute: for each edge, the triangles containing it
    def edge_to_triangles(tris):
        emap = {}
        for (u, v, w) in tris:
            for e in ((u, v), (u, w), (v, w)):
                key = (min(e), max(e))
                emap.setdefault(key, []).append((u, v, w))
        return emap

    etri = edge_to_triangles(tris)

    def is_edge(u, v):
        return v in adj[u]

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1

    # Simulated-annealing-like local search with edge swaps
    # Move: remove one edge, add another (keeping edge count)
    # Or: swap an edge for another

    def score_after_swap(tris, etri, rem_u, rem_v, add_u, add_v):
        # Estimate new score: triangles affected are those involving rem or add edges
        # Triangles removed: those in etri[(rem_u,rem_v)]
        # Triangles added: new triangles formed by add edge
        # For speed, we do a quick estimate
        # Affected triangles: those containing rem edge, plus new ones from add edge
        # New triangles from add edge: common neighbors of add_u and add_v
        common = adj[add_u] & adj[add_v]
        new_tris = []
        for w in common:
            new_tris.append((min(add_u, add_v, w), min(max(add_u, add_v), w), max(add_u, add_v, w)))
        removed_tris = etri.get((min(rem_u, rem_v), max(rem_u, rem_v)), [])
        # The new max could only come from new_tris or remain from others
        # For a fast bound: if any new tri has sum > current best, it's bad
        # We'll do full recompute only for promising moves
        return new_tris, removed_tris

    # Simpler: just do random swaps and recompute score periodically
    # But recompute is O(#triangles) which is small for n=16

    # Let's do a proper local search: try all possible single-edge swaps
    # Number of non-edges: 120 - 65 = 55
    # Number of edges: 65
    # Total swaps: 55 * 65 = 3575, each recompute is fast

    def all_non_edges():
        ne = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    ne.append((u, v))
        return ne

    def all_edges_list():
        el = []
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    el.append((u, v))
        return el

    # Local search: hill climbing with random restarts
    # Try to find a swap that reduces the max triangle sum

    def try_improve():
        nonlocal best_score, best_edges
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            el = all_edges_list()
            ne = all_non_edges()
            # For each swap, check if it reduces the score
            # To be fast, only check swaps that affect triangles with high sum
            current_tris = find_triangles()
            current_score = compute_score(current_tris)
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edges)
            for rem in el:
                for add in ne:
                    # Skip if add is same as rem
                    if rem == add:
                        continue
                    # Do the swap
                    remove_edge(rem[0], rem[1])
                    add_edge(add[0], add[1])
                    # Recompute score
                    new_tris = find_triangles()
                    new_score = compute_score(new_tris)
                    # Undo
                    remove_edge(add[0], add[1])
                    add_edge(rem[0], rem[1])
                    if new_score < current_score:
                        # Accept
                        remove_edge(rem[0], rem[1])
                        add_edge(add[0], add[1])
                        improved = True
                        break
                if improved:
                    break
        return best_score

    # Run local search
    try_improve()

    # If we have time, try random restarts with perturbation
    while time.time() < deadline - 5.0:
        # Perturb: do a few random swaps
        for _ in range(3):
            el = all_edges_list()
            ne = all_non_edges()
            if not el or not ne:
                break
            rem = rng.choice(el)
            add = rng.choice(ne)
            if rem == add:
                continue
            remove_edge(rem[0], rem[1])
            add_edge(add[0], add[1])
        # Local search from perturbed state
        try_improve()

    # Return best found
    return {'n': n, 'edges': [[u, v] for u, v in best_edges]}
