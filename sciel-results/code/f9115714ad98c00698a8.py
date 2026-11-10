"""Edge-coloring of K_16 to maximize distinct colors with no rainbow C4."""
import time
import random
import sys
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute edge indices
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1

    # Precompute all 4-cycles
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles.append((edge_idx[(a, b)], edge_idx[(b, c)], edge_idx[(c, d)], edge_idx[(a, d)]))
                    # Cycle 2: a-b, b-d, d-c, c-a
                    cycles.append((edge_idx[(a, b)], edge_idx[(b, d)], edge_idx[(c, d)], edge_idx[(a, c)]))
                    # Cycle 3: a-c, c-b, b-d, d-a
                    cycles.append((edge_idx[(a, c)], edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(a, d)]))

    num_cycles = len(cycles)

    # Precompute for each edge, which cycles it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)

    # Precompute for each edge, the other edges in each cycle it belongs to
    edge_to_cycle_others = [dict() for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            others = tuple(e for e in cycle if e != ei)
            edge_to_cycle_others[ei][ci] = others

    def is_rainbow(c0, c1, c2, c3):
        return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

    def count_violations(colors):
        count = 0
        for cycle in cycles:
            e0, e1, e2, e3 = cycle
            if is_rainbow(colors[e0], colors[e1], colors[e2], colors[e3]):
                count += 1
        return count

    def check_edge_change(colors, edge_i, new_color):
        old_color = colors[edge_i]
        if old_color == new_color:
            return 0, 0

        num_new = 0
        num_removed = 0
        for ci in edge_to_cycles[edge_i]:
            others = edge_to_cycle_others[edge_i][ci]
            o0, o1, o2 = others
            c0, c1, c2 = colors[o0], colors[o1], colors[o2]

            was_violation = is_rainbow(old_color, c0, c1, c2)
            is_violation = is_rainbow(new_color, c0, c1, c2)

            if is_violation and not was_violation:
                num_new += 1
            elif was_violation and not is_violation:
                num_removed += 1

        return num_new, num_removed

    def count_distinct_colors(colors):
        return len(set(colors))

    def simulated_annealing(colors, deadline, rng, max_steps=80000):
        """SA with swap moves to escape local optima and maximize distinct colors."""
        colors = list(colors)
        current_violations = count_violations(colors)
        best_colors = list(colors)
        best_distinct = count_distinct_colors(colors)
        best_violations = current_violations

        T_start = 5.0
        T_end = 0.01
        step = 0

        while step < max_steps:
            if time.time() > deadline:
                break

            step += 1
            T = T_start * (T_end / T_start) ** (step / max_steps)

            move_type = rng.random()
            if current_violations > 0:
                # Focus on reducing violations
                if move_type < 0.5:
                    # Recolor a violating edge
                    violating_edges = []
                    for ci, cycle in enumerate(cycles):
                        e0, e1, e2, e3 = cycle
                        if is_rainbow(colors[e0], colors[e1], colors[e2], colors[e3]):
                            violating_edges.extend([e0, e1, e2, e3])
                    if not violating_edges:
                        break
                    e = rng.choice(violating_edges)
                    old_color = colors[e]
                    new_color = rng.randint(0, MAX_COLOR - 1)
                    if new_color == old_color:
                        continue
                    num_new, num_removed = check_edge_change(colors, e, new_color)
                    delta_v = num_new - num_removed
                    if delta_v < 0 or (delta_v == 0 and rng.random() < 0.3) or rng.random() < np.exp(-delta_v / max(T, 0.01)):
                        colors[e] = new_color
                        current_violations += delta_v
                else:
                    # Swap two violating edges
                    violating_edges = []
                    for ci, cycle in enumerate(cycles):
                        e0, e1, e2, e3 = cycle
                        if is_rainbow(colors[e0], colors[e1], colors[e2], colors[e3]):
                            violating_edges.extend([e0, e1, e2, e3])
                    if len(violating_edges) >= 2:
                        e1, e2 = rng.sample(violating_edges, 2)
                        c1, c2 = colors[e1], colors[e2]
                        if c1 != c2:
                            # Check swap
                            colors[e1], colors[e2] = c2, c1
                            new_violations = count_violations(colors)
                            delta_v = new_violations - current_violations
                            if delta_v <= 0 or rng.random() < np.exp(-delta_v / max(T, 0.01)):
                                current_violations = new_violations
                            else:
                                colors[e1], colors[e2] = c1, c2
            else:
                # No violations: maximize distinct colors
                if move_type < 0.4:
                    # Try to introduce a new color
                    used = set(colors)
                    available = [c for c in range(MAX_COLOR) if c not in used]
                    if available:
                        e = rng.randint(0, NUM_EDGES - 1)
                        new_color = rng.choice(available)
                        num_new, num_removed = check_edge_change(colors, e, new_color)
                        if num_new == 0:
                            colors[e] = new_color
                        elif num_new == 1 and rng.random() < np.exp(-1.0 / max(T, 0.01)):
                            colors[e] = new_color
                elif move_type < 0.7:
                    # Recolor to a random color
                    e = rng.randint(0, NUM_EDGES - 1)
                    new_color = rng.randint(0, MAX_COLOR - 1)
                    if new_color == colors[e]:
                        continue
                    num_new, num_removed = check_edge_change(colors, e, new_color)
                    if num_new == 0:
                        colors[e] = new_color
                    elif num_new == 1 and rng.random() < np.exp(-1.5 / max(T, 0.01)):
                        colors[e] = new_color
                else:
                    # Swap two edges (key move for escaping local optima)
                    e1, e2 = rng.sample(range(NUM_EDGES), 2)
                    c1, c2 = colors[e1], colors[e2]
                    if c1 == c2:
                        continue
                    colors[e1], colors[e2] = c2, c1
                    new_violations = count_violations(colors)
                    if new_violations == 0:
                        current_violations = 0
                        distinct = count_distinct_colors(colors)
                        if distinct > best_distinct:
                            best_distinct = distinct
                            best_colors = list(colors)
                    else:
                        # Revert
                        colors[e1], colors[e2] = c1, c2

            # Track best
            if current_violations == 0:
                distinct = count_distinct_colors(colors)
                if distinct > best_distinct:
                    best_distinct = distinct
                    best_colors = list(colors)

        return best_colors, best_distinct

    def local_search(colors, max_iter, rng, deadline):
        violations = count_violations(colors)

        # Phase 1: Reduce violations to zero
        for it in range(max_iter):
            if time.time() > deadline:
                break

            if violations == 0:
                break

            changed = False
            for _ in range(50):
                ei = rng.randint(0, NUM_EDGES - 1)
                if not edge_to_cycles[ei]:
                    continue

                in_violation = False
                for ci in edge_to_cycles[ei]:
                    others = edge_to_cycle_others[ei][ci]
                    o0, o1, o2 = others
                    if is_rainbow(colors[ei], colors[o0], colors[o1], colors[o2]):
                        in_violation = True
                        break

                if not in_violation:
                    continue

                old_color = colors[ei]
                best_new_color = None
                best_delta = float('inf')

                candidate_colors = set()
                for ci in edge_to_cycles[ei]:
                    others = edge_to_cycle_others[ei][ci]
                    for e in others:
                        candidate_colors.add(colors[e])

                for _ in range(3):
                    candidate_colors.add(rng.randint(0, MAX_COLOR - 1))

                for nc in candidate_colors:
                    if nc == old_color:
                        continue
                    num_new, num_removed = check_edge_change(colors, ei, nc)
                    delta = num_new - num_removed
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = nc

                if best_new_color is not None and best_delta <= 0:
                    colors[ei] = best_new_color
                    violations += best_delta
                    changed = True
                    break

            if not changed:
                for _ in range(5):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    colors[ei] = rng.randint(0, MAX_COLOR - 1)
                violations = count_violations(colors)

        # Phase 2: Maximize distinct colors while maintaining validity
        if violations == 0:
            used_colors = set(colors)
            for it in range(max_iter):
                if time.time() > deadline:
                    break

                improved = False
                for _ in range(20):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    old_color = colors[ei]

                    available = [c for c in range(MAX_COLOR) if c not in used_colors]
                    if not available:
                        break

                    nc = rng.choice(available)
                    num_new, num_removed = check_edge_change(colors, ei, nc)

                    if num_new == 0:
                        colors[ei] = nc
                        used_colors.add(nc)
                        improved = True
                        break

                if not improved:
                    any_change = False
                    for _ in range(10):
                        ei = rng.randint(0, NUM_EDGES - 1)
                        old_color = colors[ei]
                        nc = rng.randint(0, MAX_COLOR - 1)
                        if nc == old_color:
                            continue
                        num_new, num_removed = check_edge_change(colors, ei, nc)
                        if num_new == 0:
                            colors[ei] = nc
                            used_colors.discard(old_color)
                            used_colors.add(nc)
                            any_change = True
                            break

                    if not any_change:
                        break

        return colors

    # Main search
    best_colors = None
    best_score = -1

    rng = random.Random(42)

    # Strategy 1: SA with random restarts (key improvement)
    for restart in range(15):
        if time.time() - start_time > DEADLINE:
            break

        colors = [rng.randint(0, MAX_COLOR - 1) for _ in range(NUM_EDGES)]
        sa_colors, sa_score = simulated_annealing(colors, DEADLINE - 2.0, rng, max_steps=60000)

        if sa_score > best_score:
            best_score = sa_score
            best_colors = sa_colors[:]

    # Strategy 2: SA with structured initial colorings
    for restart in range(10):
        if time.time() - start_time > DEADLINE:
            break

        mod = rng.randint(2, 10)
        offset = rng.randint(0, MAX_COLOR - 1)
        colors = []
        for i, (u, v) in enumerate(edges):
            base = (u + v) % mod
            color = (base * 13 + offset + i * 7) % MAX_COLOR
            colors.append(color)

        sa_colors, sa_score = simulated_annealing(colors, DEADLINE - 2.0, rng, max_steps=60000)

        if sa_score > best_score:
            best_score = sa_score
            best_colors = sa_colors[:]

    # Strategy 3: Local search on best SA result
    if best_colors is not None:
        if time.time() - start_time < DEADLINE - 5.0:
            improved = local_search(best_colors[:], 5000, rng, DEADLINE - 2.0)
            violations = count_violations(improved)
            if violations == 0:
                score = count_distinct_colors(improved)
                if score > best_score:
                    best_score = score
                    best_colors = improved

    # Strategy 4: More SA restarts if time permits
    for restart in range(10):
        if time.time() - start_time > DEADLINE:
            break

        # Use best_colors as starting point with perturbation
        if best_colors is not None:
            colors = best_colors[:]
            for _ in range(rng.randint(5, 20)):
                ei = rng.randint(0, NUM_EDGES - 1)
                colors[ei] = rng.randint(0, MAX_COLOR - 1)
        else:
            colors = [rng.randint(0, MAX_COLOR - 1) for _ in range(NUM_EDGES)]

        sa_colors, sa_score = simulated_annealing(colors, DEADLINE - 2.0, rng, max_steps=40000)

        if sa_score > best_score:
            best_score = sa_score
            best_colors = sa_colors[:]

    # Final fallback
    if best_colors is None:
        colors = [0] * NUM_EDGES
        for i, (u, v) in enumerate(edges):
            colors[i] = (u + v) % 15

        if count_violations(colors) == 0:
            best_colors = colors
        else:
            best_colors = [0] * NUM_EDGES

    return best_colors

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Distinct colors: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")
