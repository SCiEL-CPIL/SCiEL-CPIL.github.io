import time
import random
import math

def construct():
    """
    Finds a 5-coloring of K_n with no monochromatic C4 for the largest possible n.
    Uses a structural change: a constraint-based local search that works directly
    with adjacency bitmasks and counts C4s, with a strict global deadline guard.
    """
    start_wall = time.time()
    HARD_LIMIT = 140.0  # wall-clock limit (evaluator kills at 150s)

    def edge_index(u, v, n):
        if u > v:
            u, v = v, u
        return u * n - (u * (u + 1)) // 2 + v - u - 1

    def decode_edge(idx, n):
        for u in range(n):
            base = u * n - (u * (u + 1)) // 2
            if idx < base + (n - 1 - u):
                v = u + 1 + (idx - base)
                return u, v
        return 0, 1

    def count_c4(colors, n):
        """Count total number of monochromatic C4s. A C4 is counted 4 times
        (once per edge pair) so we divide appropriately. Actually, we count
        pairs of edges sharing two vertices in same color, which is C4 * 4?
        Let's just count pairs of common neighbors: for each color c, for each
        pair (i,j), count C(common_neighbors(c,i,j), 2). Each C4 in color c
        has 4 such pairs, so total C4 count = cost / 4. But we only need cost==0."""
        masks = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_index(i, j, n)
                c = colors[idx]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
        cost = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = masks[c][i] & masks[c][j]
                    cnt = common.bit_count()
                    cost += cnt * (cnt - 1) // 2
        return cost

    def local_search(n, time_budget):
        """Simulated annealing with fast delta computation using bitmasks."""
        total_edges = n * (n - 1) // 2
        best_cost = float('inf')
        best_colors = None

        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() - start_wall > HARD_LIMIT - 5.0:
                break

            # Random initialization
            colors = [random.randint(0, 4) for _ in range(total_edges)]

            # Build masks
            masks = [[0] * n for _ in range(5)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = edge_index(i, j, n)
                    c = colors[idx]
                    masks[c][i] |= (1 << j)
                    masks[c][j] |= (1 << i)

            # Initial cost
            cost = count_c4(colors, n)
            if cost == 0:
                return colors

            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]

            T = 5.0
            T_min = 0.05
            step = 0

            while time.time() - start_wall < HARD_LIMIT - 5.0:
                step += 1

                # Pick random edge
                idx = random.randint(0, total_edges - 1)
                u, v = decode_edge(idx, n)

                c_old = colors[idx]
                c_new = random.randint(0, 4)
                if c_old == c_new:
                    continue

                # Compute delta efficiently
                common_old = masks[c_old][u] & masks[c_old][v]
                cnt_old = common_old.bit_count()
                delta_old = cnt_old * (cnt_old - 1) // 2

                common_new = masks[c_new][u] & masks[c_new][v]
                cnt_new = common_new.bit_count()
                delta_new = cnt_new * (cnt_new - 1) // 2

                delta = delta_new - delta_old

                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept move
                    masks[c_old][u] &= ~(1 << v)
                    masks[c_old][v] &= ~(1 << u)
                    masks[c_new][u] |= (1 << v)
                    masks[c_new][v] |= (1 << u)
                    colors[idx] = c_new
                    cost += delta

                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors[:]
                        if cost == 0:
                            return colors
                else:
                    pass

                # Cool down
                T *= 0.9998
                if T < T_min:
                    T = T_min

        return best_colors if best_cost < float('inf') else None

    # Try n from 27 down to 12
    # Allocate time proportionally to difficulty
    for n in range(27, 11, -1):
        if time.time() - start_wall > HARD_LIMIT - 10.0:
            break

        # Give more time to larger n
        remaining = HARD_LIMIT - 10.0 - (time.time() - start_wall)
        time_budget = min(remaining * 0.7, 80.0)
        if time_budget < 5.0:
            break

        res = local_search(n, time_budget)
        if res is not None:
            if count_c4(res, n) == 0:
                return {'n': n, 'colors': res}

    # Fallback: n=11 with known valid coloring
    n = 11
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(min(j - i, n - j + i) - 1)
    if count_c4(colors, n) == 0:
        return {'n': n, 'colors': colors}

    return {'n': 1, 'colors': []}
