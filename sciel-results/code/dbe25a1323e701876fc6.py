"""Structural change: backtracking constraint solver with forward checking for strong edge coloring."""
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    sys.setrecursionlimit(10000)

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

    def build_conflict(edges, adj):
        n_edges = len(edges)
        conflict = [0] * n_edges
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                elif ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def greedy_color(conflict, n_edges, order):
        """Greedy coloring with given order. Returns list of colors."""
        colour = [0] * n_edges
        for idx in order:
            taken = 0
            cm = conflict[idx]
            while cm:
                j = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                taken |= (1 << colour[j])
            c = 0
            while (taken >> c) & 1:
                c += 1
            colour[idx] = c
        return colour

    def color_count(colour):
        """Return the number of distinct colors used."""
        if not colour:
            return 0
        return max(colour) + 1

    def backtracking_color(conflict, n_edges, order, k_max, time_limit, start_time):
        """
        Backtracking solver with forward checking.
        Tries to color all edges using at most k_max colors.
        Returns a valid coloring or None.
        """
        colour = [-1] * n_edges
        # Precompute: for each edge in order, its neighbors that come before it in order
        # We process edges in the given order
        positions = [0] * n_edges
        for pos, idx in enumerate(order):
            positions[idx] = pos

        def solve(pos):
            if time.time() - start_time > time_limit:
                return False
            if pos == n_edges:
                return True

            idx = order[pos]
            # Forward checking: determine which colors are available for this edge
            # A color is unavailable if any already-colored neighbor uses it
            used_colors = 0
            cm = conflict[idx]
            while cm:
                j = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                if colour[j] >= 0:
                    used_colors |= (1 << colour[j])

            # Try colors in order: prefer reusing existing colors, then new
            # First try colors 0..k_max-1 that are not in used_colors
            for c in range(k_max):
                if not (used_colors >> c) & 1:
                    colour[idx] = c
                    if solve(pos + 1):
                        return True
                    colour[idx] = -1

            return False

        if solve(0):
            return colour
        return None

    def local_improve(conflict, n_edges, colour, time_limit, start_time):
        """
        Try to reduce the number of colors by reassigning individual edges
        to lower colors when possible, or by merging color classes.
        """
        k = color_count(colour)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to reduce each edge's color
            for i in range(n_edges):
                if time.time() - start_time > time_limit:
                    return colour
                current = colour[i]
                # Try colors 0..current-1
                for c in range(current):
                    # Check if color c is available for edge i
                    ok = True
                    cm = conflict[i]
                    while cm:
                        j = (cm & -cm).bit_length() - 1
                        cm &= cm - 1
                        if colour[j] == c:
                            ok = False
                            break
                    if ok:
                        colour[i] = c
                        improved = True
                        break
        # Recompute k
        k = color_count(colour)
        # Try to eliminate the highest color class by moving its edges
        for c in range(k - 1, -1, -1):
            if time.time() - start_time > time_limit:
                return colour
            # Find all edges with color c
            edges_in_color = [i for i in range(n_edges) if colour[i] == c]
            if not edges_in_color:
                continue
            # Try to reassign each edge to a lower color
            success = True
            for i in edges_in_color:
                reassigned = False
                for nc in range(c):
                    ok = True
                    cm = conflict[i]
                    while cm:
                        j = (cm & -cm).bit_length() - 1
                        cm &= cm - 1
                        if colour[j] == nc:
                            ok = False
                            break
                    if ok:
                        colour[i] = nc
                        reassigned = True
                        break
                if not reassigned:
                    success = False
                    break
            if success:
                # Color c is now empty, we can reduce k
                # Renumber colors to be contiguous
                # Find which colors are still used
                used = set(colour)
                mapping = {}
                new_k = 0
                for c2 in sorted(used):
                    mapping[c2] = new_k
                    new_k += 1
                for i in range(n_edges):
                    colour[i] = mapping[colour[i]]
                k = new_k
                improved = True
        return colour

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        n_edges = len(edges)
        conflict = build_conflict(edges, adj)
        degs = [conflict[i].bit_count() for i in range(n_edges)]

        # Multiple greedy orderings to get a good initial coloring
        best_colour = None
        best_k = float('inf')

        # Ordering 1: descending degree
        order1 = sorted(range(n_edges), key=lambda i: -degs[i])
        c1 = greedy_color(conflict, n_edges, order1)
        k1 = color_count(c1)
        if k1 < best_k:
            best_k = k1
            best_colour = c1[:]

        # Ordering 2: ascending degree
        order2 = sorted(range(n_edges), key=lambda i: degs[i])
        c2 = greedy_color(conflict, n_edges, order2)
        k2 = color_count(c2)
        if k2 < best_k:
            best_k = k2
            best_colour = c2[:]

        # Ordering 3: random
        rng = random.Random(42)
        order3 = list(range(n_edges))
        rng.shuffle(order3)
        c3 = greedy_color(conflict, n_edges, order3)
        k3 = color_count(c3)
        if k3 < best_k:
            best_k = k3
            best_colour = c3[:]

        # Ordering 4: another random
        rng2 = random.Random(123)
        order4 = list(range(n_edges))
        rng2.shuffle(order4)
        c4 = greedy_color(conflict, n_edges, order4)
        k4 = color_count(c4)
        if k4 < best_k:
            best_k = k4
            best_colour = c4[:]

        # Local improvement
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 5:
            best_colour = local_improve(conflict, n_edges, best_colour,
                                        time_limit=min(10, remaining_time - 5),
                                        start_time=time.time())
            best_k = color_count(best_colour)

        # Backtracking: try to prove we can do with best_k - 1 colors
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 10:
            # Try k = best_k - 1
            k_try = best_k - 1
            if k_try >= 1:
                # Use a good ordering for backtracking
                order_bt = sorted(range(n_edges), key=lambda i: -degs[i])
                bt_start = time.time()
                bt_limit = min(20, remaining_time - 5)
                result = backtracking_color(conflict, n_edges, order_bt, k_try, bt_limit, bt_start)
                if result is not None:
                    best_colour = result
                    best_k = k_try
                    # Try to go even lower
                    remaining_time = time_budget - (time.time() - start_time)
                    while best_k > 1 and remaining_time > 10:
                        k_try = best_k - 1
                        bt_start = time.time()
                        bt_limit = min(15, remaining_time - 5)
                        result = backtracking_color(conflict, n_edges, order_bt, k_try, bt_limit, bt_start)
                        if result is not None:
                            best_colour = result
                            best_k = k_try
                        else:
                            break
                        remaining_time = time_budget - (time.time() - start_time)

        # Final local improvement
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 3:
            best_colour = local_improve(conflict, n_edges, best_colour,
                                        time_limit=min(5, remaining_time - 1),
                                        start_time=time.time())

        return best_colour

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    # Allocate time budget per instance
    total_budget = deadline - start_time
    per_instance = total_budget / 3.0

    for spec in specs:
        if time.time() > deadline - 3:
            break
        remaining_time = min(per_instance, deadline - time.time() - 2)
        if remaining_time < 5:
            remaining_time = 5
        colour = solve_instance(spec[0], spec[1], spec[2], remaining_time)
        out.append(colour)

    # Fallback for any missing
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        conflict = build_conflict(edges, adj)
        n_edges = len(edges)
        order = list(range(n_edges))
        colour = greedy_color(conflict, n_edges, order)
        out.append(colour)

    return out
