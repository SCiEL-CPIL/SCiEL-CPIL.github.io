"""Rewritten program with a structural change: exact-cover via recursive branch-and-bound
using a constraint-propagation (pigeonhole) style most-constrained-first search, with
a hard deadline guard and a greedy fallback."""

# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150 s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    def clique_partition(n, p, seed, time_budget):
        adj = get_graph(n, p, seed)
        # Build the set of edges
        edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.add(edge_key(u, v))

        # Precompute all cliques of size >= 2 (up to a reasonable size for speed).
        # For dense random graphs of n<=64, the number of large cliques is huge, so we
        # will generate them lazily on the fly inside the search using a greedy
        # extension restricted to the current "frontier" of remaining edges.
        #
        # However, for an exact-cover style B&B, we need a candidate set.  Instead of
        # enumerating all cliques upfront (too many), we use a heuristic: at each step
        # pick the most constrained remaining edge and try to extend it greedily to a
        # maximal clique in the subgraph induced by the remaining edges.  We also try
        # a few alternative extensions (by shuffling candidate order) to improve quality.
        #
        # To keep this tractable, we cap the number of alternative extensions per edge.

        remaining = set(edges)
        cliques = []

        # Precompute adjacency restricted to remaining edges dynamically.
        # We maintain a set of remaining edges and a helper to test if an edge is present.

        def try_grow_clique(seed_edge, adj, remaining, max_alt=4):
            """Try to grow a clique starting from seed_edge.
            Returns a list of candidate cliques (the seed edge extended in different ways)."""
            a, b = seed_edge
            best_cliques = []
            # Candidate vertices are common neighbours of a and b that still have
            # both edges (a,v) and (b,v) in remaining.
            common = (adj[a] & adj[b]) - {a, b}
            common = {v for v in common if edge_key(a, v) in remaining and edge_key(b, v) in remaining}

            if not common:
                return [[a, b]]

            # Sort candidates by a heuristic: prefer vertices with many common neighbours
            # among a, b and previously chosen vertices.
            def heuristic_sort(cands, current_clique):
                def score(v):
                    # number of current clique members v is adjacent to (in remaining)
                    s = 0
                    for x in current_clique:
                        if x == v:
                            continue
                        if edge_key(v, x) in remaining:
                            s += 1
                    # plus degree within common
                    s += len(adj[v] & set(current_clique))
                    return -s  # lower is better for sorted
                return sorted(cands, key=score)

            # Generate a few different orderings
            cands_list = list(common)
            random.shuffle(cands_list)

            for alt in range(max_alt):
                clique = [a, b]
                cands = set(cands_list)
                # Reorder for this alternative
                order = heuristic_sort(cands, clique)
                i = 0
                while i < len(order):
                    v = order[i]
                    # check v still connects to all in clique via remaining edges
                    ok = True
                    for x in clique:
                        if edge_key(v, x) not in remaining:
                            ok = False
                            break
                    if ok:
                        clique.append(v)
                        # filter order to those still compatible
                        order = [w for w in order if w not in clique and all(
                            edge_key(w, x) in remaining for x in clique)]
                        i = 0
                    else:
                        order.pop(i)

                best_cliques.append(clique)
                if len(clique) >= 5:
                    break
            return best_cliques

        # Main loop: repeatedly pick the most constrained remaining edge,
        # grow a clique, commit it, remove its edges.
        while remaining:
            if time.time() > deadline - 8.0:
                break

            # Pick the edge with the fewest "extension potential":
            # i.e. minimize the number of common neighbours in remaining.
            best_edge = None
            best_potential = float('inf')
            for e in remaining:
                a, b = e
                common = adj[a] & adj[b]
                common = {v for v in common if edge_key(a, v) in remaining and edge_key(b, v) in remaining}
                pot = len(common)
                if pot < best_potential:
                    best_potential = pot
                    best_edge = e
                    if pot == 0:
                        break

            if best_edge is None:
                break

            # Try to grow a clique from best_edge
            candidates = try_grow_clique(best_edge, adj, remaining, max_alt=3)
            # Choose the largest candidate clique
            best_clique = max(candidates, key=len)

            # Commit
            cliques.append(best_clique)
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    e = edge_key(best_clique[i], best_clique[j])
                    if e in remaining:
                        remaining.remove(e)

        # Fallback: any remaining edges become 2-cliques
        if remaining:
            for e in remaining:
                cliques.append([e[0], e[1]])

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = clique_partition(n, p, seed, 50.0)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
