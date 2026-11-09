"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n = 20
    C = 20
    deadline = time.time() + 130

    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_to_idx[(u, v)] = len(all_edges)
    num_edges = len(all_edges)

    # Enumerate all 4-cycles as frozensets of 4 edge-tuples
    all_4cycles = set()
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        for cyc in [
            frozenset([(a, b), (b, c), (c, d), (d, a)]),
            frozenset([(a, b), (b, d), (d, c), (c, a)]),
            frozenset([(a, c), (c, b), (b, d), (d, a)]),
        ]:
            all_4cycles.add(cyc)
    all_4cycles = list(all_4cycles)
    num_cycles = len(all_4cycles)

    # Map each cycle to its edge indices
    cycle_edge_indices = []
    for cyc in all_4cycles:
        idxs = [edge_to_idx[e] for e in cyc]
        cycle_edge_indices.append(idxs)

    # Map each edge to the list of cycles it belongs to
    edge_to_cycles = [[] for _ in range(num_edges)]
    for ci, idxs in enumerate(cycle_edge_indices):
        for ei in idxs:
            edge_to_cycles[ei].append(ci)

    random.seed(123)
    colors = [random.randint(0, C - 1) for _ in range(num_edges)]
    present = [True] * num_edges

    # Precompute violation status for each cycle
    def cycle_violates(ci):
        idxs = cycle_edge_indices[ci]
        if not all(present[i] for i in idxs):
            return False
        c0, c1, c2, c3 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
        return not (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3)

    # Build initial violation set
    violated = set(ci for ci in range(num_cycles) if cycle_violates(ci))
    num_violations = len(violated)

    best_violations = num_violations
    best_colors = colors[:]
    best_present = present[:]

    def update_cycle(ci):
        """Update violation status of cycle ci, return old and new status."""
        was_violated = ci in violated
        now_violates = cycle_violates(ci)
        if was_violated and not now_violates:
            violated.discard(ci)
        elif not was_violated and now_violates:
            violated.add(ci)
        return was_violated, now_violates

    T = 10.0
    T_min = 0.05
    cooling = 0.9998

    while T > T_min and time.time() < deadline:
        move_type = random.random()
        old_num_violations = len(violated)

        if move_type < 0.8:
            # Recolor
            present_edges = [i for i in range(num_edges) if present[i]]
            if not present_edges:
                break
            ei = random.choice(present_edges)
            old_color = colors[ei]
            new_color = random.randint(0, C - 1)
            if new_color == old_color:
                continue
            colors[ei] = new_color

            # Update affected cycles
            affected = edge_to_cycles[ei]
            for ci in affected:
                update_cycle(ci)

            new_num_violations = len(violated)
            if new_num_violations <= best_violations:
                best_violations = new_num_violations
                best_colors = colors[:]
                best_present = present[:]
            elif random.random() < pow(2.718281828, -(new_num_violations - old_num_violations) / T):
                pass
            else:
                colors[ei] = old_color
                for ci in affected:
                    update_cycle(ci)
        else:
            # Toggle edge
            ei = random.randint(0, num_edges - 1)
            old_present = present[ei]
            present[ei] = not old_present

            affected = edge_to_cycles[ei]
            for ci in affected:
                update_cycle(ci)

            new_num_violations = len(violated)
            new_edge_count = sum(present)
            old_edge_count = sum(best_present)

            # Objective: minimize violations, then maximize edges
            old_obj = best_violations * 1000 - old_edge_count
            new_obj = new_num_violations * 1000 - new_edge_count

            if new_obj <= old_obj:
                best_violations = new_num_violations
                best_colors = colors[:]
                best_present = present[:]
            elif random.random() < pow(2.718281828, -(new_obj - old_obj) / T):
                pass
            else:
                present[ei] = old_present
                for ci in affected:
                    update_cycle(ci)

        T *= cooling

    # Use best state
    colors = best_colors[:]
    present = best_present[:]

    # Greedy repair: remove edges to fix remaining violations
    for _ in range(10000):
        if time.time() > deadline - 3:
            break
        if not violated:
            break
        ci = random.choice(list(violated))
        idxs = cycle_edge_indices[ci]
        # Pick a random present edge in this cycle to remove
        candidates = [i for i in idxs if present[i]]
        if not candidates:
            violated.discard(ci)
            continue
        ei = random.choice(candidates)
        present[ei] = False
        for cj in edge_to_cycles[ei]:
            update_cycle(cj)

    result = []
    for i, (u, v) in enumerate(all_edges):
        if present[i]:
            result.append([u, v, colors[i]])

    return result
# EVOLVE-BLOCK-END
