"""Strong edge colourings via Local Search on Conflict Graph with bitmask acceleration."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, d, seed):
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

    def build_conflict_bitmasks(edges, n):
        """Build conflict graph as list of integer bitmasks."""
        m = len(edges)
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)

        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)

        conflict = [0] * m

        # 1. Edges sharing a vertex
        for u in range(n):
            inc = vertex_to_edges[u]
            for i in range(len(inc)):
                for j in range(i + 1, len(inc)):
                    conflict[inc[i]] |= (1 << inc[j])
                    conflict[inc[j]] |= (1 << inc[i])

        # 2. Edges joined by an edge (distance-2 in line graph)
        for i, (u, v) in enumerate(edges):
            mask = 0
            for nu in vertex_adj[u]:
                if nu != v:
                    for j in vertex_to_edges[nu]:
                        if j != i:
                            mask |= (1 << j)
            for nv in vertex_adj[v]:
                if nv != u:
                    for j in vertex_to_edges[nv]:
                        if j != i:
                            mask |= (1 << j)
            conflict[i] |= mask

        # Symmetrize (should already be symmetric but ensure)
        for i in range(m):
            for j in range(i + 1, m):
                if (conflict[i] >> j) & 1:
                    conflict[j] |= (1 << i)

        return conflict

    def greedy_color_bitmask(m, conflict, order):
        """Greedy coloring using bitmask availability."""
        colours = [0] * m
        colour_masks = []  # colour_masks[c] = bitmask of edges with colour c

        for idx in order:
            # Find available colors: a color c is available if conflict[idx] & colour_masks[c] == 0
            c = 0
            while c < len(colour_masks) and (conflict[idx] & colour_masks[c]):
                c += 1
            if c == len(colour_masks):
                colour_masks.append(0)
            colour_masks[c] |= (1 << idx)
            colours[idx] = c

        return colours, colour_masks

    def try_remove_color_bitmask(colours, colour_masks, conflict, c_remove, m, rng):
        """Try to remove color c_remove by reassigning its edges to lower colors."""
        edges_with_c = [i for i in range(m) if colours[i] == c_remove]
        if not edges_with_c:
            return True

        old_colours = colours[:]
        old_masks = colour_masks[:]
        rng.shuffle(edges_with_c)

        # Temporarily remove this color class
        colour_masks[c_remove] = 0

        for idx in edges_with_c:
            available = []
            for c_new in range(c_remove):
                if colour_masks[c_new] == 0 or not (conflict[idx] & colour_masks[c_new]):
                    available.append(c_new)

            if not available:
                colours[:] = old_colours
                colour_masks[:] = old_masks
                return False
            else:
                # Prefer lowest available
                idx_choice = rng.choice(available) if rng.random() < 0.3 else available[0]
                colours[idx] = idx_choice
                colour_masks[idx_choice] |= (1 << idx)

        return True

    def local_search_bitmask(colours, colour_masks, conflict, m, rng, time_limit):
        """Local search: repeatedly try to remove highest color."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            max_c = max(colours) if m > 0 else -1

            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    break

                if not any(colours[i] == c_remove for i in range(m)):
                    continue

                # Try multiple random orderings for reassignment
                for attempt in range(30):
                    if try_remove_color_bitmask(colours, colour_masks, conflict, c_remove, m, rng):
                        improved = True
                        break

                if improved:
                    break

                # If direct removal fails, try swap-based approach
                edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                if not edges_with_c:
                    continue

                success = False
                for idx in edges_with_c:
                    if time.time() > time_limit or success:
                        break

                    # Find blocked colors
                    blocked = set()
                    cm = conflict[idx]
                    for c_t in range(c_remove):
                        if colour_masks[c_t] and (cm & colour_masks[c_t]):
                            blocked.add(c_t)

                    for c_t in blocked:
                        if success:
                            break
                        # Find an edge j with colour c_t that conflicts with idx
                        mask_c_t = colour_masks[c_t]
                        candidates = mask_c_t & cm
                        if candidates == 0:
                            continue

                        # Try each candidate j
                        j = candidates.bit_length() - 1
                        while j >= 0:
                            if time.time() > time_limit or success:
                                break
                            # Try to move j to another color
                            old_c = colours[:]
                            old_m = [x for x in colour_masks]
                            for c_new in range(c_remove):
                                if c_new == c_t:
                                    continue
                                if not (conflict[j] & colour_masks[c_new]):
                                    # Move j to c_new
                                    colour_masks[c_t] &= ~(1 << j)
                                    colour_masks[c_new] |= (1 << j)
                                    colours[j] = c_new

                                    # Now check if idx can be reassigned
                                    avail = []
                                    for c_a in range(c_remove):
                                        if not (conflict[idx] & colour_masks[c_a]):
                                            avail.append(c_a)
                                    if avail:
                                        colours[idx] = avail[0]
                                        colour_masks[avail[0]] |= (1 << idx)
                                        success = True
                                        break
                                    else:
                                        # Undo
                                        colour_masks[c_new] &= ~(1 << j)
                                        colour_masks[c_t] |= (1 << j)
                                        colours[j] = c_t

                            if success:
                                break
                            j = (candidates & ~(1 << j)).bit_length() - 1

        return colours, colour_masks

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []

        conflict = build_conflict_bitmasks(edges, n)
        rng = random.Random(seed)

        best_colours = None
        best_count = float('inf')
        max_conflict_degree = max(conflict[i].bit_count() for i in range(m)) if m > 0 else 0
        lower_bound = max_conflict_degree + 1

        max_restarts = 200
        restart_count = 0

        while restart_count < max_restarts and time.time() < time_limit:
            # Generate diverse orderings
            r = rng.random()
            if r < 0.3:
                # Degree-based (highest conflict degree first)
                order = sorted(range(m), key=lambda i: -conflict[i].bit_count())
            elif r < 0.6:
                # Random
                order = list(range(m))
                rng.shuffle(order)
            else:
                # Degree-based with random perturbation
                order = list(range(m))
                rng.shuffle(order)
                order.sort(key=lambda i: -conflict[i].bit_count() + rng.randint(-5, 5))

            colours, colour_masks = greedy_color_bitmask(m, conflict, order)

            # Local search
            colours, colour_masks = local_search_bitmask(colours, colour_masks, conflict, m, rng, time_limit)

            current_count = max(colours) + 1 if m > 0 else 0

            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]

            restart_count += 1

            if best_count <= lower_bound:
                break

        if best_colours is None:
            order = sorted(range(m), key=lambda i: -conflict[i].bit_count())
            best_colours, _ = greedy_color_bitmask(m, conflict, order)

        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)

        inst_start = time.time()
        remaining = deadline - time.time()
        inst_limit = min(inst_start + 45.0, deadline - 5.0)

        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)

    return results
# EVOLVE-BLOCK-END
