"""Strong edge colouring: conflict-graph greedy with multiple orderings + local search."""
import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        """Build the strong edge-colouring conflict graph as adjacency lists."""
        m = len(edges)
        # vertex -> list of edge indices incident to it
        v2e = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v2e[u].append(i)
            v2e[v].append(i)

        # Two edges conflict if they share a vertex OR are joined by an edge
        # i.e., for edge e_i=(u,v) and edge e_j=(x,y), they conflict if:
        #   {u,v} & {x,y} != empty  (share a vertex)
        #   OR any of u,v is adjacent to any of x,y (joined by an edge)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check if they share a vertex
                if x in (u, v) or y in (u, v):
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if joined by an edge
                if (u in adj[x] or u in adj[y] or v in adj[x] or v in adj[y]):
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def greedy_color(edges, conflict, order):
        """Assign colours using greedy first-fit in the given edge order."""
        m = len(edges)
        colour = [0] * m  # 0 means unassigned
        # Track which colours are used
        used_colours = [0] * (m + 1)  # max number of colours used so far
        max_colour = 0

        for idx in order:
            # Find the smallest colour not used by any conflicting edge
            forbidden = set()
            for nb in conflict[idx]:
                if colour[nb] > 0:
                    forbidden.add(colour[nb])
            c = 1
            while c in forbidden:
                c += 1
            colour[idx] = c
            if c > max_colour:
                max_colour = c
        return colour, max_colour

    def try_local_improve(edges, conflict, colour, max_colour, rng, time_limit=5.0):
        """Try to reduce colour count by re-colouring or merging colour classes."""
        m = len(edges)
        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            # Try to remove the highest colour if possible
            for target_c in range(max_colour, 1, -1):
                if time.time() > DEADLINE - 5.0:
                    return colour, max_colour
                # Find all edges with this colour
                edges_with_c = [i for i in range(m) if colour[i] == target_c]
                if not edges_with_c:
                    continue
                # Try to re-colour one of them to a lower colour
                for i in edges_with_c:
                    forbidden = set()
                    for nb in conflict[i]:
                        if colour[nb] > 0:
                            forbidden.add(colour[nb])
                    # Try colours from 1 up to target_c - 1
                    for c in range(1, target_c):
                        if c not in forbidden:
                            colour[i] = c
                            # Check if target_c is still used
                            if not any(colour[j] == target_c for j in range(m)):
                                max_colour -= 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return colour, max_colour

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        conflict = build_conflict(edges)

        # Build vertex degrees for edge ordering
        vdeg = [0] * n
        for u, v in edges:
            vdeg[u] += 1
            vdeg[v] += 1

        best_colour = None
        best_max = m + 1

        # Try multiple orderings
        orderings = []
        # 1. Original sorted order
        orderings.append(list(range(m)))
        # 2. Descending by max endpoint degree
        orderings.append(sorted(range(m), key=lambda i: max(vdeg[edges[i][0]], vdeg[edges[i][1]]), reverse=True))
        # 3. Descending by sum of endpoint degrees
        orderings.append(sorted(range(m), key=lambda i: vdeg[edges[i][0]] + vdeg[edges[i][1]], reverse=True))
        # 4. Random shuffles
        rng = random.Random(42)
        for _ in range(20):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)

        for order in orderings:
            if time.time() > DEADLINE - 10.0:
                break
            colour, max_c = greedy_color(edges, conflict, order)
            if max_c < best_max:
                best_max = max_c
                best_colour = colour[:]
            # Try local improvement
            if time.time() < DEADLINE - 15.0:
                colour_imp, max_imp = try_local_improve(edges, conflict, colour, max_c, rng, time_limit=3.0)
                if max_imp < best_max:
                    best_max = max_imp
                    best_colour = colour_imp[:]

        # More random restarts if time permits
        if best_colour is not None and time.time() < DEADLINE - 20.0:
            for _ in range(50):
                if time.time() > DEADLINE - 10.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                colour, max_c = greedy_color(edges, conflict, order)
                if max_c < best_max:
                    best_max = max_c
                    best_colour = colour[:]
                if time.time() < DEADLINE - 15.0:
                    colour_imp, max_imp = try_local_improve(edges, conflict, colour, max_c, rng, time_limit=2.0)
                    if max_imp < best_max:
                        best_max = max_imp
                        best_colour = colour_imp[:]

        if best_colour is None:
            # Fallback: just use greedy with original order
            colour, _ = greedy_color(edges, conflict, list(range(m)))
            return colour
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5.0:
            # Not enough time, return empty lists (shouldn't happen)
            out.append([])
            continue
        result = solve_instance(*spec)
        out.append(result)

    return out
