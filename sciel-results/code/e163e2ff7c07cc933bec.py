"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 117  # Safety margin before 120s limit

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 8:
            break

        # Precompute all quads and quints and their triple indices
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples = []
        for a, b, c, d in all_quads:
            quad_triples.append([(a,b,c), (a,b,d), (a,c,d), (b,c,d)])

        all_quints = list(combinations(range(target_n), 5))
        quint_triples = []
        for q5 in all_quints:
            quint_triples.append(list(combinations(q5, 3)))

        # Precompute: for each triple index, which quads/quints contain it
        triple_quad_map = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples):
            for pos, t in enumerate(qt):
                triple_quad_map[triple_to_idx[t]].append((qi, pos))

        triple_quint_map = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples):
            for pos, t in enumerate(qt):
                triple_quint_map[triple_to_idx[t]].append((qi, pos))

        found = False
        max_restarts = 300
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            # Initialize: use modular coloring as warm start, then randomize some
            if restart == 0:
                # Modular init: (a+b+c) mod 3 == 0 -> red, else blue
                color_flat = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 1:
                # Modular init: (a+b+c) mod 2 == 0 -> red, else blue
                color_flat = np.array([0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 2:
                # Random
                color_flat = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            else:
                # Perturb modular init
                base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.1
                color_flat = base.copy()
                color_flat[flip_mask] = 1 - color_flat[flip_mask]

            # Count initial violations
            def count_violations(cf):
                rk4 = 0
                for qt in quad_triples:
                    idxs = [triple_to_idx[t] for t in qt]
                    if all(cf[i] == 0 for i in idxs):
                        rk4 += 1
                bk5 = 0
                for qt in quint_triples:
                    idxs = [triple_to_idx[t] for t in qt]
                    if all(cf[i] == 1 for i in idxs):
                        bk5 += 1
                return rk4, bk5

            rk4, bk5 = count_violations(color_flat)

            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Simulated annealing local search
            max_iters = 30000
            temp_start = 1.0
            temp_end = 0.01
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

                temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

                # Choose a triple to flip
                if rk4 > 0 and random.random() < 0.7:
                    # Pick from a violating red K4
                    for _ in range(30):
                        qi = random.randint(0, len(quad_triples) - 1)
                        qt = quad_triples[qi]
                        if all(color_flat[triple_to_idx[t]] == 0 for t in qt):
                            pos = random.randint(0, 3)
                            tidx = triple_to_idx[qt[pos]]
                            break
                    else:
                        tidx = random.randint(0, num_triples - 1)
                elif bk5 > 0 and random.random() < 0.7:
                    # Pick from a violating blue K5
                    for _ in range(30):
                        qi = random.randint(0, len(quint_triples) - 1)
                        qt = quint_triples[qi]
                        if all(color_flat[triple_to_idx[t]] == 1 for t in qt):
                            pos = random.randint(0, 9)
                            tidx = triple_to_idx[qt[pos]]
                            break
                    else:
                        tidx = random.randint(0, num_triples - 1)
                else:
                    tidx = random.randint(0, num_triples - 1)

                old_color = color_flat[tidx]
                new_color = 1 - old_color

                # Compute delta_rk4
                delta_rk4 = 0
                for (qi, pos) in triple_quad_map[tidx]:
                    qt = quad_triples[qi]
                    others_red = True
                    for p in range(4):
                        if p == pos:
                            continue
                        if color_flat[triple_to_idx[qt[p]]] != 0:
                            others_red = False
                            break
                    if old_color == 0 and others_red:
                        delta_rk4 -= 1
                    elif old_color == 1 and others_red:
                        delta_rk4 += 1

                # Compute delta_bk5
                delta_bk5 = 0
                for (qi, pos) in triple_quint_map[tidx]:
                    qt = quint_triples[qi]
                    others_blue = True
                    for p in range(10):
                        if p == pos:
                            continue
                        if color_flat[triple_to_idx[qt[p]]] != 1:
                            others_blue = False
                            break
                    if old_color == 1 and others_blue:
                        delta_bk5 -= 1
                    elif old_color == 0 and others_blue:
                        delta_bk5 += 1

                old_total = rk4 + bk5
                new_total = (rk4 + delta_rk4) + (bk5 + delta_bk5)
                delta = new_total - old_total

                if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    color_flat[tidx] = new_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5

            if rk4 == 0 and bk5 == 0:
                found = True
                break

        if found:
            best_n = target_n
            best_colors = color_flat.tolist()

    # Final verification
    if best_n > 9:
        color_map = {}
        for i, (a, b, c) in enumerate(combinations(range(best_n), 3)):
            color_map[(a, b, c)] = best_colors[i]

        valid = True
        for quad in combinations(range(best_n), 4):
            a, b, c, d = quad
            if all(color_map[t] == 0 for t in [(a,b,c),(a,b,d),(a,c,d),(b,c,d)]):
                valid = False
                break
        if valid:
            for quint in combinations(range(best_n), 5):
                if all(color_map[t] == 1 for t in combinations(quint, 3)):
                    valid = False
                    break

        if not valid:
            best_n = 9
            best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
