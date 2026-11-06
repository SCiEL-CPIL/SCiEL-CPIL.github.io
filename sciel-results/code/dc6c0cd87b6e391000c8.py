import random
import time
import itertools
import numpy as np

N = 16
E = N*(N-1)//2  # 120 edges

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i*(2*N-1-i)//2 + j - i - 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin

    # Precompute all unique 4-cycles (as sets of 4 edge indices)
    unique_cycles = []
    seen = set()
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        for cyc in (e1, e2, e3):
            key = frozenset(cyc)
            if key not in seen:
                seen.add(key)
                unique_cycles.append(cyc)

    num_cycles = len(unique_cycles)
    cycles_arr = np.array(unique_cycles, dtype=np.int32)  # shape (num_cycles, 4)

    # Build: for each edge, list of cycle indices it belongs to
    cycles_for_edge = [[] for _ in range(E)]
    for i, cyc in enumerate(unique_cycles):
        for e in cyc:
            cycles_for_edge[e].append(i)

    # Precompute: for each edge, the other 3 edges in each cycle it's in
    others_list = [None] * E
    for e in range(E):
        cyc_indices = cycles_for_edge[e]
        if not cyc_indices:
            others_list[e] = np.empty((0,3), dtype=np.int32)
        else:
            others = []
            for ci in cyc_indices:
                cyc = unique_cycles[ci]
                others.append([cyc[k] for k in range(4) if cyc[k] != e])
            others_list[e] = np.array(others, dtype=np.int32)

    def is_valid(colors):
        """Check if no 4-cycle has 4 distinct colors. Returns True if valid."""
        if num_cycles == 0:
            return True
        # colors is a numpy array of shape (E,)
        cycle_colors = colors[cycles_arr]  # shape (num_cycles, 4)
        # A cycle is bad if all 4 colors are distinct
        c0 = cycle_colors[:, 0]
        c1 = cycle_colors[:, 1]
        c2 = cycle_colors[:, 2]
        c3 = cycle_colors[:, 3]
        bad = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return not np.any(bad)

    def count_distinct(colors):
        freq = np.bincount(colors, minlength=120)
        return int(np.sum(freq > 0))

    best_result = None
    best_score = 0

    def try_local_search(colors_init, time_budget):
        """Run local search from initial coloring, return (colors, distinct)."""
        colors = colors_init.copy()
        freq = np.bincount(colors, minlength=120)
        distinct = int(np.sum(freq > 0))

        # Compute initial badness
        def compute_badness(colors):
            if num_cycles == 0:
                return np.zeros(num_cycles, dtype=np.int8), 0
            cycle_colors = colors[cycles_arr]
            c0 = cycle_colors[:, 0]
            c1 = cycle_colors[:, 1]
            c2 = cycle_colors[:, 2]
            c3 = cycle_colors[:, 3]
            bad = ((c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)).astype(np.int8)
            return bad, int(np.sum(bad))

        bad, total_bad = compute_badness(colors)

        local_best = colors.copy()
        local_best_distinct = distinct

        T = 3.0
        cooling = 0.9998
        min_T = 0.1

        while time.time() < time_budget:
            e = random.randint(0, E - 1)
            old_color = colors[e]
            # Bias toward introducing new colors
            if random.random() < 0.3:
                unused = [c for c in range(120) if freq[c] == 0]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, 119)
            else:
                new_color = random.randint(0, 119)

            if old_color == new_color:
                continue

            others = others_list[e]
            if others.shape[0] == 0:
                delta_bad = 0
            else:
                oc = colors[others]
                o0, o1, o2 = oc[:, 0], oc[:, 1], oc[:, 2]
                distinct_3 = (o0 != o1) & (o1 != o2) & (o0 != o2)
                in_new = (o0 == new_color) | (o1 == new_color) | (o2 == new_color)
                in_old = (o0 == old_color) | (o1 == old_color) | (o2 == old_color)
                new_bad = distinct_3 & ~in_new
                old_bad = distinct_3 & ~in_old
                delta_bad = int(np.sum(new_bad)) - int(np.sum(old_bad))

            delta_distinct = 0
            if freq[old_color] == 1:
                delta_distinct -= 1
            if freq[new_color] == 0:
                delta_distinct += 1

            new_total_bad = total_bad + delta_bad
            new_distinct = distinct + delta_distinct

            accept = False
            if new_total_bad == 0:
                if new_distinct > distinct:
                    accept = True
                elif delta_bad < 0:
                    accept = True
                elif random.random() < 0.03:
                    accept = True
            elif total_bad == 0:
                # Don't leave valid state unless it helps distinct count
                if new_distinct > distinct and new_total_bad == 0:
                    accept = True
            else:
                if delta_bad < 0:
                    accept = True
                elif random.random() < np.exp(-max(0, delta_bad) / max(T, 0.01)):
                    accept = True

            if accept:
                colors[e] = new_color
                freq[old_color] -= 1
                freq[new_color] += 1
                distinct = new_distinct
                total_bad = new_total_bad

                # Update badness for affected cycles
                for ci in cycles_for_edge[e]:
                    cyc = unique_cycles[ci]
                    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                    bad[ci] = 1 if len({c0, c1, c2, c3}) == 4 else 0

                if total_bad == 0 and distinct > local_best_distinct:
                    local_best_distinct = distinct
                    local_best = colors.copy()

            T *= cooling

        # Final check
        if is_valid(local_best):
            return local_best, local_best_distinct
        elif is_valid(colors):
            return colors, distinct
        return local_best, local_best_distinct

    # Multiple restarts with different initial colorings
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_budget = min(remaining - 1, 20.0)  # each restart gets up to 20s

        # Generate initial coloring
        if num_restarts == 1:
            # Star coloring: color(i,j) = i
            colors_init = np.zeros(E, dtype=np.int32)
            for i in range(N):
                for j in range(i + 1, N):
                    colors_init[edge_idx(i, j)] = i
        elif num_restarts == 2:
            # Random coloring with few colors
            colors_init = np.random.randint(0, 8, size=E, dtype=np.int32)
        elif num_restarts == 3:
            # Block coloring: partition vertices into groups
            colors_init = np.zeros(E, dtype=np.int32)
            # Assign vertex groups
            groups = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
            for i in range(N):
                for j in range(i + 1, N):
                    colors_init[edge_idx(i, j)] = groups[i] + groups[j]
        elif num_restarts == 4:
            # Another random with moderate colors
            colors_init = np.random.randint(0, 12, size=E, dtype=np.int32)
        else:
            # Random with increasing diversity
            max_c = min(15 + num_restarts, 40)
            colors_init = np.random.randint(0, max_c, size=E, dtype=np.int32)

        result_colors, result_distinct = try_local_search(colors_init, time_budget)

        if result_distinct > best_score:
            best_score = result_distinct
            best_result = result_colors

        # If we've achieved a high score, try to improve further
        if best_score >= 30 and time.time() < deadline - 10:
            # Continue with more targeted search
            pass

    if best_result is None:
        # Fallback: star coloring
        colors = np.zeros(E, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = i
        return colors.tolist()

    return best_result.tolist()
