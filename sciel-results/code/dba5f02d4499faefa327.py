"""Rewritten program for few-colors-without-bicolored-cycle task."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin

    n = P['n']
    edges = P['edges']

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Precompute edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    def find_bicolored_cycle(coloring, c1, c2):
        """Find a bicolored cycle using DFS. Returns list of vertices in cycle or None."""
        vertices = [i for i in range(n) if coloring[i] in (c1, c2)]
        if len(vertices) < 3:
            return None

        # Build subgraph adjacency
        sub_adj = {v: [] for v in vertices}
        for u in vertices:
            for v in adj[u]:
                if v > u and coloring[v] in (c1, c2):
                    sub_adj[u].append(v)
                    sub_adj[v].append(u)

        # DFS to find cycle
        visited = set()
        parent = {}

        def dfs(node, path):
            visited.add(node)
            for nb in sub_adj[node]:
                if nb == parent.get(node, -1):
                    continue
                if nb in visited:
                    # Found a cycle
                    cycle_start = path.index(nb)
                    return path[cycle_start:] + [nb]
                if nb not in visited:
                    parent[nb] = node
                    result = dfs(nb, path + [nb])
                    if result:
                        return result
            return None

        for v in vertices:
            if v not in visited:
                result = dfs(v, [v])
                if result:
                    return result
        return None

    def is_valid(coloring):
        """Check proper coloring and no bicolored cycles."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False

        colors_used = sorted(set(coloring))
        for i in range(len(colors_used)):
            for j in range(i + 1, len(colors_used)):
                if find_bicolored_cycle(coloring, colors_used[i], colors_used[j]):
                    return False
        return True

    def count_violations(coloring):
        """Count total violations: edge conflicts + bicolored cycles."""
        violations = 0
        for u, v in edges:
            if coloring[u] == coloring[v]:
                violations += 1

        colors_used = sorted(set(coloring))
        for i in range(len(colors_used)):
            for j in range(i + 1, len(colors_used)):
                if find_bicolored_cycle(coloring, colors_used[i], colors_used[j]):
                    violations += 1
        return violations

    def break_bicolored_cycle(coloring, num_colors, cycle, c1, c2):
        """
        Try to break a bicolored cycle by recoloring one vertex in the cycle
        to a third color. Returns True if successful.
        """
        # Try each vertex in the cycle (excluding the last which is duplicate of first)
        for idx in range(len(cycle) - 1):
            v = cycle[idx]
            old_color = coloring[v]

            # Try colors other than c1, c2, and colors used by neighbors
            neighbor_colors = set(coloring[nb] for nb in adj[v])
            for nc in range(num_colors):
                if nc in (c1, c2):
                    continue
                if nc in neighbor_colors:
                    continue

                # Try this recoloring
                coloring[v] = nc
                # Check if this broke the cycle and didn't create new issues
                # Quick check: verify no bicolored cycle for (c1, c2) anymore
                cycle2 = find_bicolored_cycle(coloring, c1, c2)
                if cycle2 is None:
                    # Also check we didn't create a new bicolored cycle with nc
                    new_colors = sorted(set(coloring))
                    created_new = False
                    for i in range(len(new_colors)):
                        for j in range(i + 1, len(new_colors)):
                            if new_colors[i] == nc or new_colors[j] == nc:
                                if find_bicolored_cycle(coloring, new_colors[i], new_colors[j]):
                                    created_new = True
                                    break
                        if created_new:
                            break

                    if not created_new:
                        return True
                    else:
                        coloring[v] = old_color
                else:
                    coloring[v] = old_color

        return False

    def local_search_break_cycles(coloring, num_colors, max_iters=5000):
        """
        Local search that specifically targets and breaks bicolored cycles.
        """
        current_violations = count_violations(coloring)
        best = coloring[:]
        best_violations = current_violations

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_violations == 0:
                break

            colors_used = sorted(set(coloring))

            # Find all bicolored cycles
            cycles_found = []
            for i in range(len(colors_used)):
                for j in range(i + 1, len(colors_used)):
                    cycle = find_bicolored_cycle(coloring, colors_used[i], colors_used[j])
                    if cycle:
                        cycles_found.append((cycle, colors_used[i], colors_used[j]))

            if cycles_found:
                # Pick a random cycle to break
                cycle, c1, c2 = random.choice(cycles_found)

                if break_bicolored_cycle(coloring, num_colors, cycle, c1, c2):
                    new_violations = count_violations(coloring)
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best = coloring[:]
                    current_violations = new_violations
                else:
                    # If can't break with existing colors, try random recoloring
                    v = random.choice(cycle[:-1])  # exclude duplicate endpoint
                    old_color = coloring[v]
                    neighbor_colors = set(coloring[nb] for nb in adj[v])
                    available = [c for c in range(num_colors) if c not in neighbor_colors]
                    if available:
                        nc = random.choice(available)
                        coloring[v] = nc
                        new_violations = count_violations(coloring)
                        if new_violations < current_violations:
                            current_violations = new_violations
                            if new_violations < best_violations:
                                best_violations = new_violations
                                best = coloring[:]
                        else:
                            coloring[v] = old_color
            else:
                # No bicolored cycles, but might have edge conflicts
                # Try to fix edge conflicts
                conflict_found = False
                for u, v in edges:
                    if coloring[u] == coloring[v]:
                        # Recolor v
                        neighbor_colors = set(coloring[nb] for nb in adj[v] if nb != u)
                        available = [c for c in range(num_colors) if c not in neighbor_colors]
                        if available:
                            nc = random.choice(available)
                            coloring[v] = nc
                            new_violations = count_violations(coloring)
                            if new_violations < current_violations:
                                current_violations = new_violations
                                if new_violations < best_violations:
                                    best_violations = new_violations
                                    best = coloring[:]
                            else:
                                coloring[v] = coloring[u]  # revert
                        break

            # Occasionally do a random perturbation to escape local minima
            if it % 100 == 99 and current_violations > 0:
                v = random.randint(0, n - 1)
                old_color = coloring[v]
                neighbor_colors = set(coloring[nb] for nb in adj[v])
                available = [c for c in range(num_colors) if c not in neighbor_colors]
                if available:
                    nc = random.choice(available)
                    coloring[v] = nc
                    new_violations = count_violations(coloring)
                    if new_violations < current_violations:
                        current_violations = new_violations
                        if new_violations < best_violations:
                            best_violations = new_violations
                            best = coloring[:]
                    else:
                        coloring[v] = old_color

        return best, best_violations

    def greedy_init(num_colors, seed=None):
        """Greedy coloring with deterministic or random ordering."""
        coloring = [-1] * n
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        order = list(range(n))
        rng.shuffle(order)

        for v in order:
            used = set(coloring[nb] for nb in adj[v] if coloring[nb] != -1)
            available = [c for c in range(num_colors) if c not in used]
            if available:
                coloring[v] = rng.choice(available)
            else:
                coloring[v] = 0
        return coloring

    # Strategy: Try 3 colors first, then 4, 5, 6
    for nc in [3, 4, 5, 6]:
        if time.time() > deadline:
            break

        # Multiple restarts with local search that breaks bicolored cycles
        for restart in range(50):
            if time.time() > deadline:
                break
            coloring = greedy_init(nc, seed=restart * 7919 + 42)
            coloring, violations = local_search_break_cycles(coloring, nc, max_iters=3000)
            if violations == 0 and is_valid(coloring):
                return [int(c) for c in coloring]

    # Fallback: all distinct colors
    return [int(c) for c in range(n)]
# EVOLVE-BLOCK-END
