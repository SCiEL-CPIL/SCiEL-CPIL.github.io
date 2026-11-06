import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n = 20
    n_colors = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)

    # Precompute all 4-cycles as tuples of 4 edge indices
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    all_4cycles = []
    for quad in combinations(range(n), 4):
        w, x, y, z = quad
        cycles = [
            [(w, x), (x, y), (y, z), (z, w)],
            [(w, x), (x, z), (z, y), (y, w)],
            [(w, y), (y, x), (x, z), (z, w)],
        ]
        for cyc in cycles:
            try:
                edge_idxs = tuple(sorted(edge_to_idx[e] for e in cyc))
                all_4cycles.append(edge_idxs)
            except KeyError:
                pass

    # For fast checking: for each edge, list of 4-cycles containing it
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    def can_add_edge(ei, c, edges_set, colors):
        """Check if adding edge ei with color c is valid."""
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            if all(e2 in edges_set for e2 in cyc if e2 != ei):
                cs = [colors.get(e2) for e2 in cyc if e2 != ei] + [c]
                if len(set(cs)) != 4:
                    return False
        return True

    def count_conflicts(ei, c, edges_set, colors):
        """Count how many 4-cycles would be violated by adding edge ei with color c."""
        conflicts = 0
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            if all(e2 in edges_set for e2 in cyc if e2 != ei):
                cs = [colors.get(e2) for e2 in cyc if e2 != ei] + [c]
                if len(set(cs)) != 4:
                    conflicts += 1
        return conflicts

    def try_add_edge(ei, edges_set, colors):
        """Try to add edge ei with any valid color. Returns True if added."""
        valid_colors = list(range(n_colors))
        random.shuffle(valid_colors)
        for c in valid_colors:
            if can_add_edge(ei, c, edges_set, colors):
                edges_set.add(ei)
                colors[ei] = c
                return True
        return False

    def greedy_build():
        """Greedy edge addition with random edge ordering."""
        edges_set = set()
        colors = {}
        remaining = list(range(num_edges))
        random.shuffle(remaining)

        for ei in remaining:
            if time.time() > deadline:
                break
            try_add_edge(ei, edges_set, colors)
        return edges_set, colors

    def local_search(edges_set, colors):
        """
        Local search: try to add missing edges.
        If an edge cannot be added, try to re-color existing edges to make it possible.
        """
        missing = [ei for ei in range(num_edges) if ei not in edges_set]
        random.shuffle(missing)

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            random.shuffle(missing)

            for ei in missing:
                if time.time() > deadline - 2.0:
                    break
                if ei in edges_set:
                    continue

                # First, try simple addition
                if try_add_edge(ei, edges_set, colors):
                    improved = True
                    missing.remove(ei)
                    continue

                # If simple addition failed, try to fix conflicts by re-coloring
                conflicting_cycles = []
                for ci in edge_cycles[ei]:
                    cyc = all_4cycles[ci]
                    if all(e2 in edges_set for e2 in cyc if e2 != ei):
                        conflicting_cycles.append(cyc)

                if not conflicting_cycles:
                    continue

                random.shuffle(conflicting_cycles)
                target_cyc = conflicting_cycles[0]
                other_edges = [e2 for e2 in target_cyc if e2 != ei]
                random.shuffle(other_edges)

                for e_change in other_edges:
                    old_color = colors[e_change]
                    possible_colors = [c for c in range(n_colors) if c != old_color]
                    random.shuffle(possible_colors)

                    for new_c in possible_colors:
                        valid = True
                        for ci in edge_cycles[e_change]:
                            cyc = all_4cycles[ci]
                            if all(e2 in edges_set for e2 in cyc):
                                cs = [colors.get(e2) for e2 in cyc if e2 != e_change] + [new_c]
                                if len(set(cs)) != 4:
                                    valid = False
                                    break

                        if not valid:
                            continue

                        colors[e_change] = new_c

                        added = False
                        valid_colors_for_ei = list(range(n_colors))
                        random.shuffle(valid_colors_for_ei)
                        for c_ei in valid_colors_for_ei:
                            if can_add_edge(ei, c_ei, edges_set, colors):
                                edges_set.add(ei)
                                colors[ei] = c_ei
                                added = True
                                break

                        if added:
                            improved = True
                            missing.remove(ei)
                            break
                        else:
                            colors[e_change] = old_color

                    if improved:
                        break

    def polish_pass(edges_set, colors):
        """
        High-precision polish: for each missing edge, find the color that minimizes
        conflicts (smooth surrogate), then try to resolve via targeted recoloring.
        """
        missing = [ei for ei in range(num_edges) if ei not in edges_set]
        random.shuffle(missing)

        for ei in missing:
            if time.time() > deadline - 1.0:
                break
            if ei in edges_set:
                continue

            # Find the color with minimum conflicts
            best_c = None
            best_conflicts = float('inf')
            for c in range(n_colors):
                conf = count_conflicts(ei, c, edges_set, colors)
                if conf < best_conflicts:
                    best_conflicts = conf
                    best_c = c

            if best_conflicts == 0:
                # No conflicts, just add it
                edges_set.add(ei)
                colors[ei] = best_c
                continue

            # Try to reduce conflicts by recoloring edges in the conflicting cycles
            # Get all conflicting cycles
            conflict_cycs = []
            for ci in edge_cycles[ei]:
                cyc = all_4cycles[ci]
                if all(e2 in edges_set for e2 in cyc if e2 != ei):
                    cs = [colors.get(e2) for e2 in cyc if e2 != ei] + [best_c]
                    if len(set(cs)) != 4:
                        conflict_cycs.append(cyc)

            if not conflict_cycs:
                continue

            # Try recoloring one edge per conflicting cycle
            for cyc in conflict_cycs[:3]:  # Limit to avoid too much work
                other_edges = [e2 for e2 in cyc if e2 != ei]
                random.shuffle(other_edges)

                for e_change in other_edges:
                    old_color = colors[e_change]
                    # Find a new color that doesn't create new conflicts
                    for new_c in range(n_colors):
                        if new_c == old_color:
                            continue

                        # Check if recoloring e_change is valid
                        valid = True
                        for ci in edge_cycles[e_change]:
                            c2 = all_4cycles[ci]
                            if all(e2 in edges_set for e2 in c2):
                                cs = [colors.get(e2) for e2 in c2 if e2 != e_change] + [new_c]
                                if len(set(cs)) != 4:
                                    valid = False
                                    break

                        if not valid:
                            continue

                        # Check if this reduces conflicts for ei
                        colors[e_change] = new_c
                        new_conflicts = count_conflicts(ei, best_c, edges_set, colors)
                        if new_conflicts < best_conflicts:
                            # Good improvement, keep it and try to add ei
                            for c_ei in range(n_colors):
                                if can_add_edge(ei, c_ei, edges_set, colors):
                                    edges_set.add(ei)
                                    colors[ei] = c_ei
                                    break
                            else:
                                # Couldn't add ei yet, but keep the recoloring
                                pass
                            break
                        else:
                            colors[e_change] = old_color

    # Main search loop
    best_edges = None
    best_count = 0

    for restart in range(200):
        if time.time() > deadline - 10:
            break

        edges_set, colors = greedy_build()
        local_search(edges_set, colors)
        polish_pass(edges_set, colors)

        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = [[u, v, colors[ei]] for ei, (u, v) in enumerate(all_edges) if ei in edges_set]

    if best_edges is None:
        # Fallback: star
        best_edges = [[0, v, 0] for v in range(1, n)]

    return best_edges
