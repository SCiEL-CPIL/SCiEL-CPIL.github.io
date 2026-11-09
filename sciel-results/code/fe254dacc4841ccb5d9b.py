import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    # ---- DSATUR clique cover of a graph (complement-colored) ----
    # adj: list of frozensets of int (the graph whose cliques we want)
    # Returns list of cliques (each a list of ints), covering every edge of adj exactly once.
    def dsatur_clique_cover(adj, k, rng, seed):
        m = k
        # comp: dynamic complement adjacency (mutable sets), starts as complement of adj
        comp = [set() for _ in range(m)]
        for i in range(m):
            ai = adj[i]
            for j in range(i + 1, m):
                if j not in ai:
                    comp[i].add(j)
                    comp[j].add(i)
        # precompute for each vertex the list of its neighbors in the original graph (potential clique mates)
        neigh = [list(adj[i]) for i in range(m)]

        uncolored = list(range(m))
        # saturation: number of distinct colors among colored comp-neighbors
        color_of = [-1] * m
        sat = [0] * m
        sat_colors = [set() for _ in range(m)]  # colors present among colored comp-neighbors
        color_count = 0
        # degree in comp (for tie-break)
        comp_deg = [len(comp[i]) for i in range(m)]

        cliques = []  # list of lists of vertices per color

        def remove_edge(a, b):
            # remove a-b from comp (they become same-clique mates)
            comp[a].discard(b)
            comp[b].discard(a)
            # update saturation for neighbors of a and b (their comp-neighbor set changed)
            for x in list(comp[a]):
                sat_colors[x].discard(color_of[a])
                sat[x] = len(sat_colors[x])
                comp_deg[x] = len(comp[x])
            for x in list(comp[b]):
                sat_colors[x].discard(color_of[b])
                sat[x] = len(sat_colors[x])
                comp_deg[x] = len(comp[x])

        # We color vertices; each color class becomes a clique in the original graph.
        # A vertex can only be added to a color class if it is adjacent in adj to ALL
        # existing members of that class, i.e. NOT in comp to any of them.
        while uncolored:
            # choose vertex: max sat, then max comp_deg, then random tie
            best = None
            best_key = None
            for v in uncolored:
                key = (sat[v], comp_deg[v], rng.random())
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            v = best
            uncolored.remove(v)

            # find best color: prefer color class with max size where v is compatible
            # (v not in comp to any member). Among compatible, pick largest class.
            best_color = None
            best_size = -1
            for c in range(color_count):
                cls = cliques[c]
                ok = True
                for w in cls:
                    if w in comp[v]:
                        ok = False
                        break
                if ok:
                    if len(cls) > best_size:
                        best_size = len(cls)
                        best_color = c
            if best_color is None:
                best_color = color_count
                cliques.append([])
                color_count += 1
                best_size = 0

            cliques[best_color].append(v)
            color_of[v] = best_color

            # update saturation of uncolored comp-neighbors of v
            for x in list(comp[v]):
                if x in uncolored:
                    if color_of[v] not in sat_colors[x]:
                        sat_colors[x].add(color_of[v])
                        sat[x] = len(sat_colors[x])

            # remove edges from v to all members of its color class (they are clique mates)
            for w in cliques[best_color]:
                if w != v:
                    remove_edge(v, w)

        # Now each clique (color class) is a clique in the original graph.
        # Build the edge sets consumed: for each clique, all pairs.
        result = []
        for cls in cliques:
            if len(cls) >= 2:
                result.append(sorted(cls))
        return result

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # cross adjacency: for each v in I, its neighbors in C
        cross = {}
        for v in I:
            cross[v] = adj[v] & set(C)

        # We'll do multiple randomized restarts.
        # Each restart: order I vertices, run dynamic DSATUR on cross complement,
        # then cover remaining C edges with DSATUR.
        best_partition = None
        best_count = None

        # Precompute all pairs in C
        all_cpairs = [(u, v) for u in range(k) for v in range(u + 1, k)]

        max_restarts = 40

        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            rng = random.Random(seed * 1000003 + r * 7919 + 13)

            # dynamic complement on C: starts with ALL pairs available (C is complete)
            # compC[u] = set of C-vertices v such that pair (u,v) is NOT yet used
            compC = [set(range(k)) - {u} for u in range(k)]
            used_pairs = set()  # (min,max)

            # order I vertices by (descending |cross|, then random)
            I_order = sorted(I, key=lambda v: (-len(cross[v]), rng.random()))

            partition = []

            # For each v in I_order: find a clique S in the current compC-graph
            # containing as many of cross[v] as possible, such that S is a clique in compC
            # (i.e., all pairs in S are available). Then add clique {v}+S, consume pairs in S.
            # We want S to be a maximal clique in compC restricted to cross[v], chosen greedily.
            # To minimize total cliques, we want S as large as possible (greedy max clique in compC).
            # But we also must ensure that using S doesn't hurt future vertices too much.
            # Heuristic: greedy maximal clique in compC induced by cross[v], ordered by degree.

            for v in I_order:
                N = cross[v]
                if not N:
                    continue
                # Build induced subgraph on N using compC (available pairs)
                # Greedy maximal clique: pick vertices one by one, each adjacent (in compC) to all picked.
                # Order candidates by compC-degree within N (descending) for larger cliques.
                cand = list(N)
                # degree within N in compC
                def deg_in_N(x):
                    return len(compC[x] & N)
                cand.sort(key=lambda x: (-deg_in_N(x), rng.random()))
                S = []
                for x in cand:
                    if x in S:
                        continue
                    ok = True
                    for y in S:
                        if y not in compC[x]:
                            ok = False
                            break
                    if ok:
                        S.append(x)
                if not S:
                    continue
                # Add clique {v} + S
                clique = S + [v]
                partition.append(sorted(clique))
                # Consume all pairs in S
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = S[i], S[j]
                        if a > b:
                            a, b = b, a
                        used_pairs.add((a, b))
                        compC[a].discard(b)
                        compC[b].discard(a)

            # Remaining C edges: pairs in all_cpairs not in used_pairs
            rem = [e for e in all_cpairs if e not in used_pairs]
            if rem:
                # Build adj for remaining C-edge graph (only C vertices)
                rem_adj = [set() for _ in range(k)]
                for u, v in rem:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                # DSATUR clique cover of rem_adj
                rem_cliques = dsatur_clique_cover(rem_adj, k, rng, seed)
                partition.extend(rem_cliques)

            count = len(partition)
            # Validate quickly
            valid = True
            edge_seen = {}
            for cl in partition:
                if len(cl) < 2:
                    valid = False
                    break
                if len(set(cl)) != len(cl):
                    valid = False
                    break
                # check clique
                for i in range(len(cl)):
                    for j in range(i + 1, len(cl)):
                        a, b = cl[i], cl[j]
                        if a > b:
                            a, b = b, a
                        if (a, b) not in edge_seen:
                            edge_seen[(a, b)] = True
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            # check all edges covered
            if len(edge_seen) != len(edges):
                valid = False
            if not valid:
                continue
            if best_count is None or count < best_count:
                best_count = count
                best_partition = partition

        # Fallback: always-valid edge-based partition (each edge as its own 2-clique)
        if best_partition is None:
            best_partition = [[u, v] for u, v in edges]

        # Ensure Python int types
        cleaned = []
        for cl in best_partition:
            cleaned.append([int(x) for x in cl])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline - 2.0:
            # Not enough time; use minimal valid partition for remaining specs
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        res = solve_instance(*spec)
        results.append(res)

    return results
