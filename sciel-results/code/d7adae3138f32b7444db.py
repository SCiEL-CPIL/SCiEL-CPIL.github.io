import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def generate_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, n):
        """
        Build conflict structure using bitmasks for efficient checking.
        Returns:
        - m: number of edges
        - conflict[i]: bitmask of edges that conflict with edge i
        - adj: adjacency list of original graph
        """
        m = len(edges)
        orig_adj = defaultdict(set)
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)

        # Build conflict bitmasks
        conflict = [0] * m

        for i in range(m):
            u, v = edges[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                x, y = edges[j]
                # Check conflict: share vertex or connected by edge
                if u == x or u == y or v == x or v == y:
                    mask |= (1 << j)
                elif (u in orig_adj[x] or u in orig_adj[y] or
                      v in orig_adj[x] or v in orig_adj[y]):
                    mask |= (1 << j)
            conflict[i] = mask

        return m, conflict, orig_adj

    def greedy_first_fit(conflict, m, max_colors=None):
        """
        Greedy coloring: assign each edge the smallest available color.
        Returns list of colors for each edge.
        """
        if max_colors is None:
            max_colors = m
        colors = [-1] * m
        used_colors = set()

        # Process edges in order of decreasing degree (conflict count)
        order = sorted(range(m), key=lambda i: bin(conflict[i]).count('1'), reverse=True)

        for i in order:
            # Find colors used by conflicting edges
            forbidden = set()
            mask = conflict[i]
            for j in range(m):
                if (mask >> j) & 1:
                    if colors[j] >= 0:
                        forbidden.add(colors[j])

            # Assign smallest available color
            c = 0
            while c in forbidden:
                c += 1
            if c >= max_colors:
                return None  # Failed
            colors[i] = c
            used_colors.add(c)

        return colors

    def can_recolor_to_color(v, c, colors, conflict, m):
        """Check if vertex v can be recolored to color c."""
        # No neighbor of v should have color c
        mask = conflict[v]
        for u in range(m):
            if (mask >> u) & 1:
                if colors[u] == c:
                    return False
        return True

    def local_search_improve(colors, conflict, m, max_colors, time_limit=10.0):
        """
        Try to reduce number of colors using local search.
        Returns improved colors or None if no improvement.
        """
        if colors is None:
            return None

        current_colors = colors[:]
        current_num_colors = max(current_colors) + 1 if m > 0 else 0

        # Try to eliminate colors from highest to lowest
        for target_color in range(current_num_colors - 1, -1, -1):
            if time.time() > start_time + time_limit:
                break
            if target_color >= current_num_colors:
                continue

            # Find all vertices with this color
            vertices_with_color = [v for v in range(m) if current_colors[v] == target_color]

            if not vertices_with_color:
                continue

            # Try to move each vertex to a lower color
            moved_any = False
            for v in vertices_with_color:
                # Try all colors lower than target_color
                for c in range(target_color):
                    if can_recolor_to_color(v, c, current_colors, conflict, m):
                        current_colors[v] = c
                        moved_any = True
                        break

            if moved_any:
                # Recalculate number of colors
                current_num_colors = max(current_colors) + 1 if m > 0 else 0
                # Restart from highest color
                break
            else:
                # Can't eliminate this color, continue
                continue

        return current_colors

    def branch_and_bound_coloring(conflict, m, max_colors, time_limit=5.0):
        """
        Branch and bound to find coloring with at most max_colors colors.
        Returns colors list or None.
        """
        if time.time() > start_time + time_limit:
            return None

        colors = [-1] * m
        color_count = [0] * max_colors  # Count of vertices using each color

        # Sort vertices by degree (conflict count) descending
        order = sorted(range(m), key=lambda i: bin(conflict[i]).count('1'), reverse=True)

        def is_valid(v, c):
            """Check if vertex v can be assigned color c."""
            # No neighbor of v should have color c
            mask = conflict[v]
            for u in range(m):
                if (mask >> u) & 1:
                    if colors[u] == c:
                        return False
            return True

        def backtrack(idx):
            """Recursive backtracking."""
            if time.time() > start_time + time_limit:
                return False

            if idx == m:
                return True

            v = order[idx]

            # Try each color
            for c in range(max_colors):
                if is_valid(v, c):
                    colors[v] = c
                    color_count[c] += 1

                    if backtrack(idx + 1):
                        return True

                    colors[v] = -1
                    color_count[c] -= 1

            return False

        if backtrack(0):
            return colors
        return None

    def mrw_backtracking(conflict, m, max_colors, time_limit=5.0):
        """
        MRV (Minimum Remaining Values) heuristic backtracking.
        Orders uncolored edges by number of available colors (ascending).
        Uses forward checking to prune.
        """
        if time.time() > start_time + time_limit:
            return None

        colors = [-1] * m
        # For each edge, track which colors are forbidden
        forbidden = [set() for _ in range(m)]

        def count_available(v):
            """Count available colors for edge v."""
            return max_colors - len(forbidden[v])

        def is_valid(v, c):
            """Check if edge v can be assigned color c."""
            return c not in forbidden[v]

        def forward_check(v, c, uncolored_set):
            """
            After assigning color c to edge v, update forbidden sets
            for uncolored neighbors. Returns False if any uncolored
            neighbor has no available colors (dead end).
            """
            mask = conflict[v]
            for u in uncolored_set:
                if (mask >> u) & 1:
                    forbidden[u].add(c)
                    if len(forbidden[u]) >= max_colors:
                        return False
            return True

        def backtrack(uncolored_set):
            """Recursive backtracking with MRV."""
            if time.time() > start_time + time_limit:
                return False

            if not uncolored_set:
                return True

            # MRV: pick uncolored edge with fewest available colors
            best_v = None
            best_avail = max_colors + 1
            for v in uncolored_set:
                avail = max_colors - len(forbidden[v])
                if avail < best_avail:
                    best_avail = avail
                    best_v = v
                if avail <= 0:
                    return False  # Dead end

            v = best_v
            new_set = uncolored_set - {v}

            # Try each available color
            for c in range(max_colors):
                if c not in forbidden[v]:
                    colors[v] = c
                    if forward_check(v, c, new_set):
                        if backtrack(new_set):
                            return True
                    # Undo forward checking
                    mask = conflict[v]
                    for u in new_set:
                        if (mask >> u) & 1:
                            forbidden[u].discard(c)
                    colors[v] = -1

            return False

        if backtrack(set(range(m))):
            return colors
        return None

    def solve_instance(edges, n, time_budget):
        """
        Solve one instance: find minimum number of colors for strong edge coloring.
        """
        m, conflict, orig_adj = build_conflict_bitmasks(edges, n)

        if m == 0:
            return []

        # Start with greedy to get an upper bound
        greedy_colors = greedy_first_fit(conflict, m)
        if greedy_colors is None:
            # Fallback: simple sequential coloring
            greedy_colors = list(range(m))

        current_best = greedy_colors[:]
        current_num_colors = max(current_best) + 1 if m > 0 else 0

        # Try to improve with local search
        improved = local_search_improve(current_best, conflict, m, current_num_colors, time_limit=min(5.0, time_budget * 0.3))
        if improved is not None:
            new_num_colors = max(improved) + 1 if m > 0 else 0
            if new_num_colors < current_num_colors:
                current_best = improved
                current_num_colors = new_num_colors

        # Try MRV backtracking with decreasing max_colors
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 2.0:
            for max_c in range(current_num_colors - 1, 0, -1):
                if time.time() > start_time + time_budget - 1.0:
                    break
                mrv_colors = mrw_backtracking(conflict, m, max_c, time_limit=min(3.0, remaining_time * 0.5))
                if mrv_colors is not None:
                    current_best = mrv_colors
                    current_num_colors = max_c
                    break

        return current_best

    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    # Allocate time budget: more time for larger instances
    total_budget = deadline - start_time - 5.0  # Reserve 5s for safety
    time_per_instance = [total_budget * 0.25, total_budget * 0.35, total_budget * 0.40]

    for i, (n, d, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            # Not enough time, use simple greedy
            edges = generate_instance(n, d, seed)
            m, conflict, _ = build_conflict_bitmasks(edges, n)
            colors = greedy_first_fit(conflict, m) if m > 0 else []
            if colors is None:
                colors = list(range(m))
            results.append(colors)
            continue

        edges = generate_instance(n, d, seed)
        time_budget = min(time_per_instance[i], deadline - start_time - 2.0)
        colors = solve_instance(edges, n, time_budget)
        results.append(colors)

    return results
