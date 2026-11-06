import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill limit

    n = 20
    num_colors = 20

    # Precompute all 4-cycles and edge-to-cycle incidence
    all_cycles = []
    edge_to_cycles = defaultdict(list)

    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = tuple(sorted((min(e), max(e)) for e in cycle))
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}

    def can_add_edge(e, c, edge_colors):
        """Check if adding edge e with color c keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            all_present = True
            other_colors = []
            for oe in cycle:
                if oe == e:
                    continue
                if oe not in edge_colors:
                    all_present = False
                    break
                other_colors.append(edge_colors[oe])
            if not all_present:
                continue
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True

    def find_valid_color(e, edge_colors):
        """Find a valid color for edge e, prioritizing colors not used in adjacent cycles."""
        # Collect colors used in cycles containing e
        used_in_cycles = set()
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            for oe in cycle:
                if oe != e and oe in edge_colors:
                    used_in_cycles.add(edge_colors[oe])

        unused = [c for c in range(num_colors) if c not in used_in_cycles]
        used = [c for c in range(num_colors) if c in used_in_cycles]
        random.shuffle(unused)
        random.shuffle(used)
        for c in unused + used:
            if can_add_edge(e, c, edge_colors):
                return c
        return None

    def get_conflict_edges(edge_colors):
        conflict = set()
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict.add(e)
        return conflict

    # Phase 1: Structured initial solution using a near-complete graph
    # Use a systematic coloring: color = (u * 7 + v * 3) % num_colors for edge (u,v)
    edge_colors = {}
    for u in range(n):
        for v in range(u + 1, n):
            c = (u * 7 + v * 3) % num_colors
            edge_colors[(u, v)] = c

    # Repair: remove edges causing conflicts, prioritizing high-degree edges
    # Try recoloring before removing
    max_repair = 3000
    for _ in range(max_repair):
        if time.time() > deadline:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        # Pick the edge in the most conflict cycles
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        # Try recoloring first
        recolored = False
        for c in range(num_colors):
            if c == edge_colors[e_to_remove]:
                continue
            old_c = edge_colors[e_to_remove]
            del edge_colors[e_to_remove]
            if can_add_edge(e_to_remove, c, edge_colors):
                edge_colors[e_to_remove] = c
                recolored = True
                break
            else:
                edge_colors[e_to_remove] = old_c
        if not recolored:
            del edge_colors[e_to_remove]

    # Phase 2: Local search with incremental missing set
    best_count = len(edge_colors)
    best_state = dict(edge_colors)

    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)

    missing_set = set(e for e in all_edges if e not in edge_colors)

    # Iterative improvement loop
    while time.time() < deadline:
        # Move type 1: Add a missing edge (prioritize edges with more available colors)
        if missing_set:
            # Score a small sample of missing edges
            sample_size = min(len(missing_set), 15)
            sample = random.sample(list(missing_set), sample_size)
            scored = []
            for e in sample:
                valid_count = 0
                for c in range(num_colors):
                    if can_add_edge(e, c, edge_colors):
                        valid_count += 1
                        if valid_count >= 3:
                            break
                if valid_count > 0:
                    scored.append((valid_count, e))
            if scored:
                scored.sort(reverse=True)
                top_edges = [e for _, e in scored[:5]]
                e_add = random.choice(top_edges)
                c = find_valid_color(e_add, edge_colors)
                if c is not None:
                    edge_colors[e_add] = c
                    missing_set.discard(e_add)
                    save_if_better()
                    continue

        # Move type 2: Recolor an existing edge
        if edge_colors:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_c = edge_colors[e_recolor]
            for _ in range(3):
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c and can_add_edge(e_recolor, new_c, edge_colors):
                    edge_colors[e_recolor] = new_c
                    break

        # Move type 3: Swap - remove one edge, add another
        if edge_colors and missing_set and random.random() < 0.4:
            e_remove = random.choice(list(edge_colors.keys()))
            e_add = random.choice(list(missing_set))
            old_c = edge_colors[e_remove]
            del edge_colors[e_remove]
            missing_set.add(e_remove)
            c_add = find_valid_color(e_add, edge_colors)
            if c_add is not None:
                edge_colors[e_add] = c_add
                missing_set.discard(e_add)
                save_if_better()
                # Try to re-add e_remove
                c_rem = find_valid_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    missing_set.discard(e_remove)
                    save_if_better()
            else:
                # Revert
                c_rem = find_valid_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    missing_set.discard(e_remove)
                else:
                    edge_colors[e_remove] = old_c

        # Move type 4: Double swap - remove 2, add 2
        if random.random() < 0.15 and len(edge_colors) >= 2 and len(missing_set) >= 2:
            e1, e2 = random.sample(list(edge_colors.keys()), 2)
            e3, e4 = random.sample(list(missing_set), 2)
            c1 = edge_colors[e1]
            c2 = edge_colors[e2]
            del edge_colors[e1]
            del edge_colors[e2]
            missing_set.add(e1)
            missing_set.add(e2)
            c3 = find_valid_color(e3, edge_colors)
            c4 = find_valid_color(e4, edge_colors)
            if c3 is not None and c4 is not None:
                edge_colors[e3] = c3
                edge_colors[e4] = c4
                missing_set.discard(e3)
                missing_set.discard(e4)
                save_if_better()
                # Try to re-add e1, e2
                c1_new = find_valid_color(e1, edge_colors)
                c2_new = find_valid_color(e2, edge_colors)
                if c1_new is not None:
                    edge_colors[e1] = c1_new
                    missing_set.discard(e1)
                if c2_new is not None:
                    edge_colors[e2] = c2_new
                    missing_set.discard(e2)
                save_if_better()
            else:
                # Revert
                edge_colors[e1] = c1
                edge_colors[e2] = c2
                missing_set.discard(e1)
                missing_set.discard(e2)

    # Use the best state found
    edge_colors = best_state
    missing_set = set(e for e in all_edges if e not in edge_colors)

    # Phase 3: Polish - systematic pass over all missing edges
    for e in list(missing_set):
        if time.time() > deadline:
            break
        c = find_valid_color(e, edge_colors)
        if c is not None:
            edge_colors[e] = c
            missing_set.discard(e)

    # Final cleanup: ensure validity
    for _ in range(500):
        if time.time() > deadline:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        del edge_colors[e_to_remove]

    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
