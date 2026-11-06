"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 85
    n = 128

    # Precompute all nonconstant 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    num_aps = len(aps)

    pos_aps = [[] for _ in range(n)]
    for idx, (i, j, k, l) in enumerate(aps):
        pos_aps[i].append(idx)
        pos_aps[j].append(idx)
        pos_aps[k].append(idx)
        pos_aps[l].append(idx)
    ap_pos = [aps[idx] for idx in range(num_aps)]

    def full_violation_count(colors):
        v = 0
        for idx in range(num_aps):
            i, j, k, l = ap_pos[idx]
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                v += 1
        return v

    def density_greedy(max_colors, seed, order_mode='position'):
        """Density-incremental greedy with incremental risk tracking."""
        rng = random.Random(seed)
        colors = [None] * n
        assigned = [False] * n

        # Determine order
        if order_mode == 'position':
            order = list(range(n))
        elif order_mode == 'degree':
            order = sorted(range(n), key=lambda p: -len(pos_aps[p]))
        else:
            order = list(range(n))
            rng.shuffle(order)

        # Track per-AP state: distinct color count among assigned, and whether any unassigned
        ap_distinct = [0] * num_aps
        ap_unassigned = [4] * num_aps  # number of unassigned positions in AP

        for pos in order:
            best_color = None
            best_key = None
            used_colors = sorted({colors[p] for p in range(n) if assigned[p]})

            candidate_colors = list(used_colors)
            for c in range(len(used_colors), min(len(used_colors) + 2, max_colors)):
                candidate_colors.append(c)
            if len(candidate_colors) < max_colors:
                for c in range(max_colors):
                    if c not in candidate_colors:
                        candidate_colors.append(c)
                candidate_colors = candidate_colors[:max_colors]

            for c in candidate_colors:
                # Tentatively assign
                colors[pos] = c
                assigned[pos] = True

                # Compute key: (hard_violations, risk, new_color_penalty)
                hard = 0
                risk = 0
                for ap_idx in pos_aps[pos]:
                    i, j, k, l = ap_pos[ap_idx]
                    present = []
                    any_un = False
                    for p in (i, j, k, l):
                        if assigned[p]:
                            present.append(colors[p])
                        else:
                            any_un = True
                    distinct = len(set(present))
                    if distinct < 3:
                        if any_un:
                            risk += 1
                        else:
                            hard += 1

                colors[pos] = None
                assigned[pos] = False

                new_color_pen = 1 if c not in used_colors else 0
                key = (hard, risk, new_color_pen)
                if best_key is None or key < best_key:
                    best_key = key
                    best_color = c

            colors[pos] = best_color
            assigned[pos] = True

        return colors

    def local_search(colors, max_colors, deadline, seed=0):
        rng = random.Random(seed)
        colors = colors[:]
        cur_viol = full_violation_count(colors)
        best = colors[:]
        best_viol = cur_viol
        if cur_viol == 0:
            return best
        no_improve = 0
        for step in range(8000):
            if time.time() > deadline - 3:
                break
            if cur_viol == 0:
                return best
            pos = rng.randint(0, n - 1)
            old = colors[pos]
            new = rng.randint(0, max_colors - 1)
            if new == old:
                continue
            delta = 0
            for ap_idx in pos_aps[pos]:
                i, j, k, l = ap_pos[ap_idx]
                was = len({colors[i], colors[j], colors[k], colors[l]}) < 3
                colors[pos] = new
                is = len({colors[i], colors[j], colors[k], colors[l]}) < 3
                colors[pos] = old
                if was and not is:
                    delta -= 1
                elif not was and is:
                    delta += 1
            if delta <= 0:
                colors[pos] = new
                cur_viol += delta
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            if no_improve > 2000:
                break
        return best if best_viol < cur_viol else colors

    def try_merge_colors(colors, deadline):
        current = colors[:]
        distinct = len(set(current))
        if distinct <= 4:
            return current, distinct
        color_list = sorted(set(current))
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(color_list)):
                for j in range(i + 1, len(color_list)):
                    if time.time() > deadline - 5:
                        return current, len(set(current))
                    c1, c2 = color_list[i], color_list[j]
                    test = current[:]
                    for idx in range(n):
                        if test[idx] == c2:
                            test[idx] = c1
                    if full_violation_count(test) == 0:
                        current = test
                        color_list = sorted(set(current))
                        improved = True
                        break
                if improved:
                    break
        return current, len(set(current))

    best_colors = None
    best_distinct = 128

    # Phase 1: Density greedy
    for max_colors in [4, 5, 6, 7, 8]:
        if time.time() > deadline - 20:
            break
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            for order_mode in ['position', 'degree', 'random']:
                if time.time() > deadline - 12:
                    break
                colors = density_greedy(max_colors, seed * 100 + max_colors, order_mode)
                viol = full_violation_count(colors)
                if viol == 0:
                    merged, distinct = try_merge_colors(colors, deadline)
                    if distinct < best_distinct:
                        best_distinct = distinct
                        best_colors = merged[:]
                else:
                    colors2 = local_search(colors, max_colors, deadline, seed)
                    if full_violation_count(colors2) == 0:
                        merged, distinct = try_merge_colors(colors2, deadline)
                        if distinct < best_distinct:
                            best_distinct = distinct
                            best_colors = merged[:]
        if best_distinct <= 4:
            break

    # Phase 2: Random + local search
    for max_colors in range(4, 12):
        if time.time() > deadline - 8:
            break
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            colors = [random.randint(0, max_colors - 1) for _ in range(n)]
            colors = local_search(colors, max_colors, deadline, trial + max_colors * 50)
            if full_violation_count(colors) == 0:
                merged, distinct = try_merge_colors(colors, deadline)
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = merged[:]
            if best_distinct <= 4:
                break
        if best_distinct <= 4:
            break

    if best_colors is not None:
        return best_colors

    # Fallback: trivially valid 128-color coloring
    return list(range(n))

# EVOLVE-BLOCK-END
