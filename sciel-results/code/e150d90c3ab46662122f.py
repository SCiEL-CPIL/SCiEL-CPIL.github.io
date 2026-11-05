"""Strong edge colourings via multi-ordering greedy + local merging."""
import time
import random
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adj(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_conflict_set(e, adj):
        """Return set of vertices that conflict with edge e (endpoints + neighbors)."""
        u, v = e
        return {u, v} | adj[u] | adj[v]

    def greedy_color(edges, adj, order):
        """Greedy color edges in given order. Returns dict edge->color."""
        colour = {}
        for e in order:
            near = edge_conflict_set(e, adj)
            taken = set()
            for f, c in colour.items():
                fu, fv = f
                if fu in near or fv in near:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            colour[e] = c
        return colour

    def merge_color_classes(edges, adj, colour, order):
        """Try to merge color classes to reduce total colors. Returns improved colour dict."""
        # Build list of edges per color
        max_c = max(colour.values()) if colour else 0
        for _ in range(50):  # limited passes
            # Group edges by color
            groups = defaultdict(list)
            for e in edges:
                groups[colour[e]].append(e)
            # Try to merge each pair of color classes
            merged = False
            for c1 in range(max_c):
                if c1 not in groups or not groups[c1]:
                    continue
                for c2 in range(c1 + 1, max_c + 1):
                    if c2 not in groups or not groups[c2]:
                        continue
                    # Check if all edges in group c2 can be recolored to c1
                    can_merge = True
                    for e2 in groups[c2]:
                        near2 = edge_conflict_set(e2, adj)
                        for e1 in groups[c1]:
                            near1 = edge_conflict_set(e1, adj)
                            if near1 & near2:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        for e2 in groups[c2]:
                            colour[e2] = c1
                        merged = True
            if not merged:
                break
        return colour

    def try_ordering(edges, adj, order, time_budget):
        """Run greedy + merge, return (num_colors, colour_dict)."""
        colour = greedy_color(edges, adj, order)
        colour = merge_color_classes(edges, adj, colour, order)
        max_c = max(colour.values()) if colour else 0
        return max_c + 1, colour

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        if not edges:
            return []
        adj = build_adj(edges, n)
        m = len(edges)

        # Generate several orderings
        orderings = []

        # 1. Conflict-degree ordering: sort by number of conflicting edges (descending)
        edge_conflict_count = {}
        for e in edges:
            near = edge_conflict_set(e, adj)
            count = sum(1 for f in edges if (f[0] in near or f[1] in near))
            edge_conflict_count[e] = count
        orderings.append(sorted(edges, key=lambda e: -edge_conflict_count[e]))

        # 2. Reverse conflict-degree
        orderings.append(sorted(edges, key=lambda e: edge_conflict_count[e]))

        # 3. Random orderings
        rng = random.Random(seed + 999)
        for _ in range(10):
            order = edges[:]
            rng.shuffle(order)
            orderings.append(order)

        # 4. Weighted random by degree
        for _ in range(5):
            weights = [edge_conflict_count[e] + 1 for e in edges]
            order = []
            remaining = list(edges)
            w_remaining = list(weights)
            while remaining:
                total = sum(w_remaining)
                r = rng.random() * total
                cum = 0
                for i, w in enumerate(w_remaining):
                    cum += w
                    if r <= cum:
                        order.append(remaining[i])
                        remaining.pop(i)
                        w_remaining.pop(i)
                        break
            orderings.append(order)

        best_count = len(edges)
        best_colour = None

        for order in orderings:
            if time.time() > deadline:
                break
            count, colour = try_ordering(edges, adj, order, deadline)
            if count < best_count:
                best_count = count
                best_colour = dict(colour)

        if best_colour is None:
            # Fallback: simple greedy
            colour = greedy_color(edges, adj, edges)
            best_colour = colour
            best_count = max(colour.values()) + 1

        return [best_colour[e] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 5:
            # Not enough time; use simple greedy for remaining
            n, d, seed = spec
            edges = instance(n, d, seed)
            adj = build_adj(edges, n)
            colour = greedy_color(edges, adj, edges)
            out.append([colour[e] for e in edges])
        else:
            out.append(solve_instance(*spec))

    return out
