"""Strong edge colouring: fixed-k simulated annealing with bitmask conflicts."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 5

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
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)

        for i, (u1, v1) in enumerate(edges):
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
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

    def conflicts_to_bitmask(conflicts, n_edges):
        masks = [0] * n_edges
        for i in range(n_edges):
            mask = 0
            for j in conflicts[i]:
                mask |= (1 << j)
            masks[i] = mask
        return masks

    def greedy_coloring(edges, conflicts, order, masks):
        n_edges = len(edges)
        color = [-1] * n_edges
        color_masks = [0] * (n_edges + 1)
        for e_idx in order:
            c = 0
            while (color_masks[c] & masks[e_idx]):
                c += 1
            color[e_idx] = c
            color_masks[c] |= (1 << e_idx)
        return color

    def sa_fixed_k(edges, masks, k, n_edges, time_budget, seed_val):
        """
        Simulated annealing to find a valid k-coloring.
        State: each edge assigned to color 0..k-1.
        Energy: number of conflicting pairs within same color.
        Move: reassign one edge to a different color.
        """
        if k <= 0:
            return None
        rng = random.Random(seed_val)

        # Initialize: greedy into k colors (assign to first valid, else random)
        colors = [-1] * n_edges
        color_masks = [0] * k
        for i in range(n_edges):
            placed = False
            for c in range(k):
                if not (color_masks[c] & masks[i]):
                    colors[i] = c
                    color_masks[c] |= (1 << i)
                    placed = True
                    break
            if not placed:
                c = rng.randrange(k)
                colors[i] = c
                color_masks[c] |= (1 << i)

        # Compute initial energy: count conflicts within each color class
        def compute_energy():
            energy = 0
            for c in range(k):
                cm = color_masks[c]
                # For each edge in color c, count conflicts also in color c
                m = cm
                while m:
                    low = m & (-m)
                    i = low.bit_length() - 1
                    m ^= low
                    # Count bits in (masks[i] & cm)
                    energy += (masks[i] & cm).bit_count()
            return energy // 2  # each pair counted twice

        current_energy = compute_energy()
        if current_energy == 0:
            return colors  # already valid

        T_start = 5.0
        T_end = 0.01
        start = time.time()
        best_colors = None
        best_energy = current_energy

        # Track best valid solution
        while time.time() - start < time_budget:
            # Pick a random edge and a random new color
            i = rng.randrange(n_edges)
            old_c = colors[i]
            new_c = rng.randrange(k)
            if new_c == old_c:
                continue

            # Compute delta energy
            # Remove i from old_c, add to new_c
            old_mask = color_masks[old_c]
            new_mask = color_masks[new_c]

            # Conflicts of i with other edges in old_c (excluding i)
            old_conf = (masks[i] & (old_mask & ~(1 << i))).bit_count()
            # Conflicts of i with other edges in new_c
            new_conf = (masks[i] & new_mask).bit_count()

            delta = new_conf - old_conf

            # Accept if improving or with probability exp(-delta/T)
            accept = False
            if delta <= 0:
                accept = True
            else:
                # Temperature schedule
                elapsed = time.time() - start
                frac = min(1.0, elapsed / time_budget)
                T = T_start * (T_end / T_start) ** frac
                if T > 0.001:
                    import math
                    if rng.random() < math.exp(-delta / T):
                        accept = True

            if accept:
                color_masks[old_c] &= ~(1 << i)
                color_masks[new_c] |= (1 << i)
                colors[i] = new_c
                current_energy += delta

                if current_energy < best_energy:
                    best_energy = current_energy
                    if current_energy == 0:
                        return list(colors)  # valid!
                    best_colors = list(colors)

        if best_energy == 0 and best_colors is not None:
            return best_colors
        return None  # no valid coloring found

    def local_search(edges, conflicts, color, time_budget, masks):
        """Local search with move-to-lower-color and 2-swap."""
        n_edges = len(edges)
        color = list(color)
        color_count = max(color) + 1 if color else 0
        rng = random.Random(42)

        color_class_masks = [0] * (n_edges + 1)
        for i, c in enumerate(color):
            color_class_masks[c] |= (1 << i)

        start = time.time()
        max_iterations = 500

        for iteration in range(max_iterations):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break

            improved = False
            order = list(range(n_edges))
            rng.shuffle(order)

            # Phase 1: move to lower color
            for e_idx in order:
                c_current = color[e_idx]
                if c_current == 0:
                    continue
                for new_c in range(c_current):
                    if not (color_class_masks[new_c] & masks[e_idx]):
                        color_class_masks[c_current] &= ~(1 << e_idx)
                        color_class_masks[new_c] |= (1 << e_idx)
                        color[e_idx] = new_c
                        improved = True
                        if c_current == color_count - 1 and color_class_masks[c_current] == 0:
                            color_count -= 1
                        break

            # Phase 2: 2-swap to eliminate top color
            if color_count > 1:
                top_color = color_count - 1
                top_mask = color_class_masks[top_color]
                if top_mask:
                    top_edges = []
                    m = top_mask
                    while m:
                        low = m & (-m)
                        e1 = low.bit_length() - 1
                        top_edges.append(e1)
                        m ^= low

                    for e1 in top_edges:
                        if time.time() - start > time_budget or time.time() > deadline:
                            break
                        c1 = color[e1]
                        if c1 != top_color:
                            continue
                        for c2 in range(top_color):
                            if c2 == 0:
                                continue
                            c2_mask = color_class_masks[c2]
                            if not c2_mask:
                                continue
                            m2 = c2_mask
                            while m2:
                                low2 = m2 & (-m2)
                                e2 = low2.bit_length() - 1
                                m2 ^= low2
                                if not ((color_class_masks[c2] & ~(1 << e2)) & masks[e1]):
                                    if not ((color_class_masks[c1] & ~(1 << e1)) & masks[e2]):
                                        color_class_masks[c1] &= ~(1 << e1)
                                        color_class_masks[c2] &= ~(1 << e2)
                                        color_class_masks[c2] |= (1 << e1)
                                        color_class_masks[c1] |= (1 << e2)
                                        color[e1] = c2
                                        color[e2] = c1
                                        improved = True
                                        if color_class_masks[top_color] == 0:
                                            color_count -= 1
                                        break
                            if color_class_masks[top_color] == 0:
                                break

            if not improved:
                break

        return color, color_count

    def optimize_instance(n, d, seed, time_budget):
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        masks = conflicts_to_bitmask(conflicts, len(edges))
        n_edges = len(edges)

        if n_edges == 0:
            return [], 0

        elapsed_start = time.time()
        best_color = None
        best_count = float('inf')

        # Step 1: Get an initial upper bound via greedy
        conflict_counts = [masks[i].bit_count() for i in range(n_edges)]
        order_desc = sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True)
        g_color = greedy_coloring(edges, conflicts, order_desc, masks)
        g_count = max(g_color) + 1 if g_color else 0
        best_color = list(g_color)
        best_count = g_count

        # Local search to improve initial
        remaining = time_budget - (time.time() - elapsed_start)
        if remaining > 0.5:
            ls_color, ls_count = local_search(edges, conflicts, best_color, remaining * 0.2, masks)
            if ls_count < best_count:
                best_color = list(ls_color)
                best_count = ls_count

        # Step 2: Try fixed-k SA from lower bound up to best_count - 1
        # Lower bound: max conflict degree + 1 is not a valid LB for strong coloring.
        # Use a heuristic: for degree d, conjecture is 5d^2/4.
        # But we want to try small k first.
        # A simple LB: the size of the largest clique in the conflict graph.
        # Hard to compute, so use max(conflict_counts) as a weak LB? No, that's an upper bound on chi.
        # Actually for strong edge coloring, a valid LB is: for any vertex v of degree d,
        # all d edges incident to v must have distinct colors, AND edges at distance 2 from v
        # also constrain. A simple LB is d+1 (for standard) but for strong it's higher.
        # Let's just try from a reasonable starting point.
        # The conjecture gives 5d^2/4. Let's start from max(2, int(0.6 * best_count)).
        LB = max(2, int(0.5 * best_count))
        if LB < 2:
            LB = 2

        remaining = time_budget - (time.time() - elapsed_start)
        if remaining > 1.0 and best_count > LB:
            num_k = best_count - LB
            time_per_k = remaining * 0.7 / max(1, num_k)
            # Cap time_per_k to avoid spending too long on one k
            time_per_k = min(time_per_k, remaining * 0.3)

            for k in range(LB, best_count):
                if time.time() > deadline:
                    break
                if time.time() - elapsed_start > time_budget:
                    break

                # Try multiple restarts for this k
                sa_found = False
                for restart in range(3):
                    t_left = time_budget - (time.time() - elapsed_start)
                    if t_left < 0.2:
                        break
                    t_alloc = min(time_per_k / 3.0, t_left * 0.9)
                    if t_alloc < 0.1:
                        break
                    sa_color = sa_fixed_k(edges, masks, k, n_edges, t_alloc, seed + k * 100 + restart)
                    if sa_color is not None:
                        # Verify validity
                        valid = True
                        for i in range(n_edges):
                            for j in range(i + 1, n_edges):
                                if sa_color[i] == sa_color[j] and (masks[i] >> j) & 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            best_color = sa_color
                            best_count = k
                            sa_found = True
                            break

        # Step 3: Final local search polish
        remaining = time_budget - (time.time() - elapsed_start)
        if remaining > 0.3 and best_color is not None:
            ls_color, ls_count = local_search(edges, conflicts, best_color, remaining * 0.8, masks)
            if ls_count < best_count:
                best_color = list(ls_color)
                best_count = ls_count

        if best_color is None:
            best_color = greedy_coloring(edges, conflicts, list(range(n_edges)), masks)
            best_count = max(best_color) + 1 if best_color else 0

        return best_color, best_count

    results = []
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()

    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            masks = conflicts_to_bitmask(conflicts, len(edges))
            color = greedy_coloring(edges, conflicts, list(range(len(edges))), masks)
            results.append(color)
            continue

        time_budget = remaining_time / (len(instances) - idx)
        coloring, _ = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        remaining_time = deadline - time.time()

    return results

# EVOLVE-BLOCK-END
