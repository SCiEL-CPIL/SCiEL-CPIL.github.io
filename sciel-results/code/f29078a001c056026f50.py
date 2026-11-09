"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    # Precompute for n=34 (target)
    target_n = 34
    triple_list = list(combinations(range(target_n), 3))
    num_triples = len(triple_list)
    triple_to_idx = {t: i for i, t in enumerate(triple_list)}

    # For each quad (4-set), the 4 triples it contains
    all_quads = list(combinations(range(target_n), 4))
    quad_triples_idx = []
    for a, b, c, d in all_quads:
        quad_triples_idx.append([
            triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
            triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
        ])
    num_quads = len(quad_triples_idx)

    # For each quint (5-set), the 10 triples it contains
    all_quints = list(combinations(range(target_n), 5))
    quint_triples_idx = []
    for q5 in all_quints:
        quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
    num_quints = len(quint_triples_idx)

    # For each triple, which quads/quints contain it (for incremental updates)
    triple_in_quads = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quad_triples_idx):
        for pos, tidx in enumerate(qt):
            triple_in_quads[tidx].append((qi, pos))

    triple_in_quints = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quint_triples_idx):
        for pos, tidx in enumerate(qt):
            triple_in_quints[tidx].append((qi, pos))

    # Convert to numpy arrays for faster access
    quad_triples_np = np.array(quad_triples_idx, dtype=np.int32)
    quint_triples_np = np.array(quint_triples_idx, dtype=np.int32)

    # Convert to lists of tuples for faster indexing in local search
    quad_triples_list = [tuple(t) for t in quad_triples_idx]
    quint_triples_list = [tuple(t) for t in quint_triples_idx]

    found = False
    max_restarts = 50

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize coloring with various strategies
        if restart == 0:
            color_arr = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart == 1:
            color_arr = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart == 2:
            color_arr = np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart == 3:
            color_arr = np.array([0 if (a+b+c) % 6 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart == 4:
            color_arr = np.array([0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart == 5:
            color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        elif restart == 6:
            color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            color_arr[np.random.random(num_triples) < 0.4] = 0
        elif restart == 7:
            color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            color_arr[np.random.random(num_triples) < 0.4] = 1
        elif restart % 2 == 0:
            base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            flip_mask = np.random.random(num_triples) < 0.15
            color_arr = base.copy()
            color_arr[flip_mask] = 1 - color_arr[flip_mask]
        else:
            base = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            flip_mask = np.random.random(num_triples) < 0.15
            color_arr = base.copy()
            color_arr[flip_mask] = 1 - color_arr[flip_mask]

        # Count violations using vectorized numpy operations
        quad_red_counts = np.sum(color_arr[quad_triples_np] == 0, axis=1)
        rk4 = int(np.sum(quad_red_counts == 4))

        quint_blue_counts = np.sum(color_arr[quint_triples_np] == 1, axis=1)
        bk5 = int(np.sum(quint_blue_counts == 10))

        if rk4 == 0 and bk5 == 0:
            found = True
            break

        # Local search with simulated annealing and constraint propagation
        max_iters = 50000
        temp_start = 3.0
        temp_end = 0.001
        it = 0
        stuck_count = 0
        best_rk4 = rk4
        best_bk5 = bk5
        best_color_copy = color_arr.copy()

        while it < max_iters:
            if time.time() > deadline - 3:
                break
            if rk4 == 0 and bk5 == 0:
                found = True
                break

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip - bias toward violating structures
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.4:
                # Find a violating red K4 and flip one of its triples
                violating_quads = np.where(quad_red_counts == 4)[0]
                if len(violating_quads) > 0:
                    qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                    qt = quad_triples_list[qi]
                    tidx = qt[random.randint(0, 3)]
            elif bk5 > 0 and r < 0.8:
                # Find a violating blue K5 and flip one of its triples
                violating_quints = np.where(quint_blue_counts == 10)[0]
                if len(violating_quints) > 0:
                    qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                    qt = quint_triples_list[qi]
                    tidx = qt[random.randint(0, 9)]
            elif rk4 > 0 and r < 0.9:
                # Near-violating red K4 (3 out of 4 red)
                near_violating = np.where(quad_red_counts == 3)[0]
                if len(near_violating) > 0:
                    qi = near_violating[random.randint(0, len(near_violating)-1)]
                    qt = quad_triples_list[qi]
                    # Pick the blue triple to flip
                    for t in qt:
                        if color_arr[t] == 1:
                            tidx = t
                            break
            elif bk5 > 0:
                # Near-violating blue K5 (9 out of 10 blue)
                near_violating = np.where(quint_blue_counts == 9)[0]
                if len(near_violating) > 0:
                    qi = near_violating[random.randint(0, len(near_violating)-1)]
                    qt = quint_triples_list[qi]
                    # Pick the red triple to flip
                    for t in qt:
                        if color_arr[t] == 0:
                            tidx = t
                            break

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = int(color_arr[tidx])

            # Compute delta_rk4 incrementally
            delta_rk4 = 0
            for (qi, pos) in triple_in_quads[tidx]:
                qt = quad_triples_list[qi]
                # Check if other 3 triples are red
                others_red = True
                for p in range(4):
                    if p != pos and color_arr[qt[p]] != 0:
                        others_red = False
                        break
                if old_color == 0 and others_red:
                    delta_rk4 -= 1
                elif old_color == 1 and others_red:
                    delta_rk4 += 1

            # Compute delta_bk5 incrementally
            delta_bk5 = 0
            for (qi, pos) in triple_in_quints[tidx]:
                qt = quint_triples_list[qi]
                # Check if other 9 triples are blue
                others_blue = True
                for p in range(10):
                    if p != pos and color_arr[qt[p]] != 1:
                        others_blue = False
                        break
                if old_color == 1 and others_blue:
                    delta_bk5 -= 1
                elif old_color == 0 and others_blue:
                    delta_bk5 += 1

            delta = delta_rk4 + delta_bk5

            # Acceptance criterion
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < np.exp(-delta / max(temp, 1e-10))

            if accept:
                # Apply the flip
                new_color = 1 - old_color
                color_arr[tidx] = new_color
                rk4 += delta_rk4
                bk5 += delta_bk5

                # Update violation counts for affected quads/quints
                for (qi, pos) in triple_in_quads[tidx]:
                    old_rc = int(quad_red_counts[qi])
                    if old_color == 0:
                        quad_red_counts[qi] = old_rc - 1
                    else:
                        quad_red_counts[qi] = old_rc + 1
                for (qi, pos) in triple_in_quints[tidx]:
                    old_bc = int(quint_blue_counts[qi])
                    if old_color == 1:
                        quint_blue_counts[qi] = old_bc - 1
                    else:
                        quint_blue_counts[qi] = old_bc + 1

                # Track best solution
                if rk4 + bk5 < best_rk4 + best_bk5:
                    best_rk4 = rk4
                    best_bk5 = bk5
                    best_color_copy = color_arr.copy()
                    stuck_count = 0
                else:
                    stuck_count += 1

                # If stuck for too long, restart from best
                if stuck_count > 5000:
                    color_arr = best_color_copy.copy()
                    quad_red_counts = np.sum(color_arr[quad_triples_np] == 0, axis=1)
                    quint_blue_counts = np.sum(color_arr[quint_triples_np] == 1, axis=1)
                    rk4 = int(np.sum(quad_red_counts == 4))
                    bk5 = int(np.sum(quint_blue_counts == 10))
                    stuck_count = 0
                    temp = max(temp_start * 0.5, temp_end)
            else:
                stuck_count += 1

            it += 1

        # After local search, check if we found a valid solution
        if rk4 == 0 and bk5 == 0:
            found = True
            break

        # If we improved but didn't solve, keep the best and try again
        if best_rk4 + best_bk5 < rk4 + bk5:
            color_arr = best_color_copy.copy()
            quad_red_counts = np.sum(color_arr[quad_triples_np] == 0, axis=1)
            quint_blue_counts = np.sum(color_arr[quint_triples_np] == 1, axis=1)
            rk4 = int(np.sum(quad_red_counts == 4))
            bk5 = int(np.sum(quint_blue_counts == 10))

    if found:
        best_n = target_n
        best_colors = color_arr.tolist()
        if is_valid(target_n, best_colors):
            pass
        else:
            # Fallback to smaller n
            for try_n in range(33, 8, -1):
                if time.time() > deadline - 1:
                    break
                triple_list_sm = list(combinations(range(try_n), 3))
                num_triples_sm = len(triple_list_sm)
                triple_to_idx_sm = {t: i for i, t in enumerate(triple_list_sm)}

                all_quads_sm = list(combinations(range(try_n), 4))
                quad_triples_idx_sm = []
                for a, b, c, d in all_quads_sm:
                    quad_triples_idx_sm.append([
                        triple_to_idx_sm[(a,b,c)], triple_to_idx_sm[(a,b,d)],
                        triple_to_idx_sm[(a,c,d)], triple_to_idx_sm[(b,c,d)]
                    ])

                all_quints_sm = list(combinations(range(try_n), 5))
                quint_triples_idx_sm = []
                for q5 in all_quints_sm:
                    quint_triples_idx_sm.append([triple_to_idx_sm[t] for t in combinations(q5, 3)])

                quad_triples_np_sm = np.array(quad_triples_idx_sm, dtype=np.int32)
                quint_triples_np_sm = np.array(quint_triples_idx_sm, dtype=np.int32)

                for restart_sm in range(20):
                    if time.time() > deadline - 1:
                        break
                    color_arr_sm = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list_sm], dtype=np.int8)
                    quad_red_counts_sm = np.sum(color_arr_sm[quad_triples_np_sm] == 0, axis=1)
                    rk4_sm = int(np.sum(quad_red_counts_sm == 4))
                    quint_blue_counts_sm = np.sum(color_arr_sm[quint_triples_np_sm] == 1, axis=1)
                    bk5_sm = int(np.sum(quint_blue_counts_sm == 10))

                    if rk4_sm == 0 and bk5_sm == 0:
                        best_n = try_n
                        best_colors = color_arr_sm.tolist()
                        break

                    # Quick local search
                    for it_sm in range(10000):
                        if time.time() > deadline - 1:
                            break
                        if rk4_sm == 0 and bk5_sm == 0:
                            break
                        tidx_sm = random.randint(0, num_triples_sm - 1)
                        old_color_sm = int(color_arr_sm[tidx_sm])
                        # Simple flip
                        color_arr_sm[tidx_sm] = 1 - old_color_sm
                        quad_red_counts_sm = np.sum(color_arr_sm[quad_triples_np_sm] == 0, axis=1)
                        rk4_sm = int(np.sum(quad_red_counts_sm == 4))
                        quint_blue_counts_sm = np.sum(color_arr_sm[quint_triples_np_sm] == 1, axis=1)
                        bk5_sm = int(np.sum(quint_blue_counts_sm == 10))

                    if rk4_sm == 0 and bk5_sm == 0:
                        best_n = try_n
                        best_colors = color_arr_sm.tolist()
                        break

                if best_n > 9:
                    break
            else:
                best_n = 9
                best_colors = mod_color(9, 3)
    else:
        best_n = 9
        best_colors = mod_color(9, 3)

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
