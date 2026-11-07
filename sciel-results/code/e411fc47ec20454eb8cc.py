import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)

        # Precompute all K-edges
        k_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                k_edges.add((u, v))

        # For each v in I, compute K-edges within N(v)
        def k_edges_in(neigh):
            s = set(neigh)
            edges = []
            for u in neigh:
                for w in neigh:
                    if u < w:
                        edges.append((u, w))
            return edges

        v_kedges = {}
        for v in range(k, n):
            v_kedges[v] = k_edges_in(neighbors[v])

        # Greedy maximal packing:
        # Process v's in order of decreasing |N(v)| (most beneficial first)
        # A v can "form a clique" N(v)∪{v} only if none of its K-edges are already used
        # When we form a clique for v, we mark all K-edges in N(v) as used
        # This covers |N(v)| cross edges + |K-edges in N(v)| K-edges in 1 clique
        # vs. using |N(v)| 2-cliques + separate K-edge coverage

        # Sort by |N(v)| descending
        v_order = sorted(range(k, n), key=lambda v: -len(neighbors[v]))

        used_kedges = set()
        cliques = []
        v_clique = {}  # v -> clique list if formed
        skipped = []

        for v in v_order:
            kedges = v_kedges[v]
            # Check if any K-edge in N(v) is already used
            can_form = True
            for e in kedges:
                if e in used_kedges:
                    can_form = False
                    break
            if can_form and len(neighbors[v]) >= 2:
                # Form clique N(v) ∪ {v}
                clique = list(neighbors[v]) + [v]
                cliques.append(clique)
                v_clique[v] = clique
                # Mark K-edges as used
                for e in kedges:
                    used_kedges.add(e)
            else:
                skipped.append(v)

        # Now cover remaining edges:
        # 1. Cross edges for skipped v's: each (u, v) where u in N(v)
        # 2. K-edges not in used_kedges

        # Remaining cross edges
        remaining_cross = []
        for v in skipped:
            for u in neighbors[v]:
                remaining_cross.append((u, v))

        # Remaining K-edges
        remaining_kedges = list(k_edges - used_kedges)

        # Greedy maximal clique cover for remaining edges
        # Build adjacency for the remaining subgraph
        # Vertices involved: 0..n-1
        # Edge set: remaining_cross + remaining_kedges
        all_remaining_edges = remaining_cross + remaining_kedges

        if not all_remaining_edges:
            return [[int(x) for x in c] for c in cliques]

        # Build adjacency
        adj = {v: set() for v in range(n)}
        edge_set = set()
        for (a, b) in all_remaining_edges:
            adj[a].add(b)
            adj[b].add(a)
            edge_set.add((min(a, b), max(a, b)))

        # Greedy: repeatedly find a maximal clique in the remaining graph
        # Use a simple greedy: pick vertex with highest degree, expand to clique
        remaining_edges = set(edge_set)
        remaining_adj = {v: set() for v in adj}
        for v in remaining_adj:
            remaining_adj[v] = set(adj[v])

        # We'll iteratively remove edges as we form cliques
        # For efficiency, track remaining edges per vertex
        rem_edges = {v: set() for v in range(n)}
        for (a, b) in remaining_edges:
            rem_edges[a].add(b)
            rem_edges[b].add(a)

        active_vertices = set(v for v in range(n) if rem_edges[v])

        while active_vertices:
            # Pick vertex with highest remaining degree
            best_v = max(active_vertices, key=lambda v: len(rem_edges[v]))
            # Greedy clique expansion
            clique = [best_v]
            candidates = set(rem_edges[best_v])
            # Repeatedly add a candidate that is connected to all current clique members
            changed = True
            while changed:
                changed = False
                for c in list(candidates):
                    # Check c is connected to all in clique
                    ok = True
                    for cv in clique:
                        if c not in rem_edges[cv]:
                            ok = False
                            break
                    if ok:
                        clique.append(c)
                        candidates.discard(c)
                        changed = True
                        break

            if len(clique) < 2:
                # Shouldn't happen if active_vertices is correct, but safety
                active_vertices.discard(best_v)
                continue

            cliques.append(clique)

            # Remove all edges within this clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (min(a, b), max(a, b))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        rem_edges[a].discard(b)
                        rem_edges[b].discard(a)

            # Update active_vertices
            for v in clique:
                if not rem_edges[v]:
                    active_vertices.discard(v)

            # Also remove vertices that lost all edges
            for v in list(active_vertices):
                if not rem_edges[v]:
                    active_vertices.discard(v)

        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            # Safety: return what we have
            break
        cliques = solve_instance(*spec)
        results.append(cliques)

    # If we ran out of time, fill remaining with trivial solution
    while len(results) < 3:
        spec = specs[len(results)]
        n, k, p, seed = spec
        neighbors = instance(n, k, p, seed)
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        for v in range(k, n):
            for u in neighbors[v]:
                cliques.append([u, v])
        results.append([[int(x) for x in c] for c in cliques])

    return results
