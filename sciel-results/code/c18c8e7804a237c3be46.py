"""Strong edge colouring of bounded-degree graphs via DSATUR + greedy + swap-based local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 110  # seconds to finish before 120s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
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
        return sorted(edges), degree

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def precompute_conflicts(edges, adj):
        """Precompute conflict sets using vertex-edge indexing for speed."""
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)

        for i, (u1, v1) in enumerate(edges):
            # Edges sharing a vertex
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)

            # Distance-2 edges: neighbours of u1 or v1 (excluding the edge itself)
            for neighbor in adj[u1]:
                if neighbor == v1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)
            for neighbor in adj[v1]:
                if neighbor == u1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)

        return conflicts

    def dsatur_coloring(edges, conflicts):
        """DSATUR heuristic: colour the most-saturated edge first."""
        n_edges = len(edges)
        color = [-1] * n_edges
        neighbor_colors = [set() for _ in range(n_edges)]

        for _ in range(n_edges):
            if time.time() > deadline:
                break
            best_idx = -1
            best_sat = -1
            best_deg = -1
            for i in range(n_edges):
                if color[i] != -1:
                    continue
                sat = len(neighbor_colors[i])
                deg = len(conflicts[i])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_idx = i

            if best_idx == -1:
                break

            c = 0
            while c in neighbor_colors[best_idx]:
                c += 1
            color[best_idx] = c

            for j in conflicts[best_idx]:
                if color[j] == -1:
                    neighbor_colors[j].add(c)

        return color

    def greedy_coloring(edges, conflicts, order):
        """Greedy coloring in given order."""
        n_edges = len(edges)
        color = [-1] * n_edges

        for e_idx in order:
            taken = set()
            for c_idx in conflicts[e_idx]:
                if color[c_idx] != -1:
                    taken.add(color[c_idx])
            c = 0
            while c in taken:
                c += 1
            color[e_idx] = c

        return color

    def local_search(edges, conflicts, color, time_budget):
        """Local search with move-to-lower-color and 2-swap moves."""
        n_edges = len(edges)
        color = list(color)
        color_count = max(color) + 1 if color else 0
        rng = random.Random(42)

        # Build color class sets for fast lookup
        color_classes = defaultdict(set)
        for i, c in enumerate(color):
            color_classes[c].add(i)

        start = time.time()
        max_iterations = 10000

        for iteration in range(max_iterations):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break

            improved = False
            order = list(range(n_edges))
            rng.shuffle(order)

            # Phase 1: try to move each edge to a lower colour
            for e_idx in order:
                c_current = color[e_idx]
                if c_current == 0:
                    continue

                for new_c in range(c_current):
                    can_move = True
                    for c_idx in conflicts[e_idx]:
                        if color[c_idx] == new_c:
                            can_move = False
                            break
                    if can_move:
                        color_classes[c_current].discard(e_idx)
                        color_classes[new_c].add(e_idx)
                        color[e_idx] = new_c
                        improved = True
                        if c_current == color_count - 1 and not color_classes[c_current]:
                            color_count -= 1
                        break

            # Phase 2: try 2-swap moves to eliminate the highest colour
            if color_count > 0:
                top_color = color_count - 1
                top_edges = list(color_classes[top_color])
                if top_edges:
                    for e1 in top_edges:
                        if time.time() - start > time_budget:
                            break
                        c1 = color[e1]
                        if c1 != top_color:
                            continue
                        # Try swapping e1 with an edge e2 in a lower colour
                        for e2 in order:
                            c2 = color[e2]
                            if c2 == c1 or c2 == 0:
                                continue
                            # Check if e1 can move to c2
                            can1 = True
                            for c_idx in conflicts[e1]:
                                if color[c_idx] == c2 and c_idx != e2:
                                    can1 = False
                                    break
                            if not can1:
                                continue
                            # Check if e2 can move to c1
                            can2 = True
                            for c_idx in conflicts[e2]:
                                if color[c_idx] == c1 and c_idx != e1:
                                    can2 = False
                                    break
                            if not can2:
                                continue
                            # Perform swap
                            color_classes[c1].discard(e1)
                            color_classes[c2].discard(e2)
                            color_classes[c2].add(e1)
                            color_classes[c1].add(e2)
                            color[e1] = c2
                            color[e2] = c1
                            improved = True
                            if not color_classes[top_color]:
                                color_count -= 1
                            break

            if not improved:
                break

        return color, color_count

    def optimize_instance(n, d, seed, time_budget):
        """Optimize coloring for a single instance within time budget."""
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        n_edges = len(edges)

        best_color = None
        best_count = float('inf')

        elapsed_start = time.time()

        # DSATUR initial coloring
        dsatur_color = dsatur_coloring(edges, conflicts)
        if -1 in dsatur_color:
            dsatur_count = float('inf')
        else:
            dsatur_count = max(dsatur_color) + 1 if dsatur_color else 0
        if dsatur_count < best_count:
            best_color = list(dsatur_color)
            best_count = dsatur_count

        # Local search on DSATUR
        remaining = time_budget - (time.time() - elapsed_start)
        if remaining > 0.5:
            ls_color, ls_count = local_search(edges, conflicts, dsatur_color, remaining * 0.5)
            if ls_count < best_count:
                best_color = list(ls_color)
                best_count = ls_count

        # Try multiple greedy orderings with local search
        remaining = time_budget - (time.time() - elapsed_start)
        n_restarts = max(3, int(remaining * 4))

        # Precompute orderings
        orderings = []
        # High-conflict-first
        conflict_counts = [len(conflicts[i]) for i in range(n_edges)]
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True))
        # Degree-based
        edge_degrees = [degree[u] + degree[v] for u, v in edges]
        orderings.append(sorted(range(n_edges), key=lambda i: edge_degrees[i], reverse=True))
        # Random shuffles
        for s in range(n_restarts):
            rng = random.Random(seed + 1000 + s)
            shuffled = list(range(n_edges))
            rng.shuffle(shuffled)
            orderings.append(shuffled)

        for ordering in orderings:
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break

            g_color = greedy_coloring(edges, conflicts, ordering)
            g_count = max(g_color) + 1 if g_color else 0

            if g_count < best_count:
                best_color = list(g_color)
                best_count = g_count

            # Local search on this greedy solution
            remaining = time_budget - (time.time() - elapsed_start)
            if remaining > 0.3 and best_count > 1:
                ls_color, ls_count = local_search(edges, conflicts, g_color, remaining * 0.3)
                if ls_count < best_count:
                    best_color = list(ls_color)
                    best_count = ls_count

        if best_color is None or -1 in best_color:
            best_color = greedy_coloring(edges, conflicts, list(range(n_edges)))
            best_count = max(best_color) + 1 if best_color else 0

        return best_color, best_count

    results = []
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()

    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            # Emergency fallback
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            color = greedy_coloring(edges, conflicts, list(range(len(edges))))
            results.append(color)
            continue

        time_budget = remaining_time / (len(instances) - idx)
        coloring, _ = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        remaining_time = deadline - time.time()

    return results

# EVOLVE-BLOCK-END
