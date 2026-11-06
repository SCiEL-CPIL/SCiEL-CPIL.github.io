"""Strong edge colourings via maximum-clique extraction on the induced-matching graph."""

import time
import sys

# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def max_clique(n, adj, deadline):
        """Tomita maximum clique with greedy coloring bound. adj: list of frozensets."""
        best = [0]
        best_set = [()]
        # ordering by degree (ascending) for the coloring heuristic
        order = sorted(range(n), key=lambda i: len(adj[i]))
        # remap to new indices
        pos = {old: new for new, old in enumerate(order)}
        new_adj = [set() for _ in range(n)]
        for i in range(n):
            for j in adj[i]:
                if pos[j] > pos[i]:
                    new_adj[pos[i]].add(pos[j])
        adj2 = [frozenset(s) for s in new_adj]

        # greedy color sort on a candidate set (list of vertices) -> returns list sorted by color (ascending color = later in list)
        def color_sort(cand):
            # cand: list of vertex indices; returns (ordered_list, colors) where ordered_list[i] has color colors[i]
            uncolored = list(cand)
            color_of = {}
            color = 0
            ordered = []
            while uncolored:
                color += 1
                new_uncolored = []
                for v in uncolored:
                    # can use current color if no neighbor in this color class already placed
                    conflict = False
                    for u in ordered:
                        if color_of[u] == color and u in adj2[v]:
                            conflict = True
                            break
                    if conflict:
                        new_uncolored.append(v)
                    else:
                        color_of[v] = color
                        ordered.append(v)
                uncolored = new_uncolored
            return ordered, color_of, color

        def expand(R, P):
            if time.time() > deadline:
                return
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                    best_set[0] = tuple(R)
                return
            ordered, color_of, maxc = color_sort(P)
            # iterate in reverse color order (highest color first)
            for i in range(len(ordered) - 1, -1, -1):
                if len(R) + color_of[ordered[i]] <= best[0]:
                    return
                v = ordered[i]
                newP = [u for u in ordered[:i] if u in adj2[v]]
                R.append(v)
                expand(R, newP)
                R.pop()

        expand([], list(range(n)))
        return best_set[0]

    def solve_instance(edges, deadline):
        n_e = len(edges)
        if n_e == 0:
            return []
        adj_map = {}
        for u, v in edges:
            adj_map.setdefault(u, set()).add(v)
            adj_map.setdefault(v, set()).add(u)
        # complement adjacency: edge i-j if they form an induced matching pair
        comp = [set() for _ in range(n_e)]
        for i in range(n_e):
            a, b = edges[i]
            near = {a, b}
            for x in (a, b):
                near |= adj_map.get(x, set())
            for j in range(i + 1, n_e):
                c, d = edges[j]
                if not ({c, d} & near):
                    comp[i].add(j)
                    comp[j].add(i)
        comp_f = [frozenset(s) for s in comp]

        remaining = list(range(n_e))
        colour = [None] * n_e
        color_idx = 0
        while remaining:
            if time.time() > deadline:
                break
            clique = max_clique(len(remaining),
                                 [comp_f[v] for v in remaining] if False else
                                 # pass adj restricted: we need adj within remaining
                                 [frozenset({remaining.index(w) for w in comp_f[v] if w in set(remaining)}) for v in remaining],
                                 deadline)
            # clique indices are in the remapped space of `remaining`
            for idx in clique:
                v = remaining[idx]
                colour[v] = color_idx
            color_idx += 1
            # remove used
            used = set(remaining[idx] for idx in clique)
            remaining = [v for v in remaining if v not in used]
        # if we ran out of time, fill rest with greedy to keep valid
        if any(c is None for c in colour):
            # greedy assignment
            for v in range(n_e):
                if colour[v] is None:
                    near = set()
                    a, b = edges[v]
                    for x in (a, b):
                        near |= adj_map.get(x, set())
                    near |= {a, b}
                    taken = set()
                    for w in range(n_e):
                        if colour[w] is not None:
                            c, d = edges[w]
                            if {c, d} & near:
                                taken.add(colour[w])
                    c = 0
                    while c in taken:
                        c += 1
                    colour[v] = c
        return colour

    deadline = time.time() + 140.0
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        out.append(solve_instance(edges, deadline))
    return out
# EVOLVE-BLOCK-END
