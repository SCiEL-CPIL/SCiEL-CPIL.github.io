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

    def build_structures(target_n):
        """Precompute all structures needed for fast local search."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, list of quad indices containing it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                triple_in_quads[tidx].append(qi)

        # For each triple, list of quint indices containing it
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                triple_in_quints[tidx].append(qi)

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints)

    def local_search(target_n, init_colors, deadline):
        """Run local search to eliminate violations. Returns (colors, success)."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = build_structures(target_n)

        color_arr = np.array(init_colors, dtype=np.int8)
        
        # Initialize counts
        red_count_in_quad = np.zeros(num_quads, dtype=np.int32)
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                if color_arr[tidx] == 0:
                    red_count_in_quad[qi] += 1
        
        blue_count_in_quint = np.zeros(num_quints, dtype=np.int32)
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                if color_arr[tidx] == 1:
                    blue_count_in_quint[qi] += 1

        rk4 = np.sum(red_count_in_quad == 4)
        bk5 = np.sum(blue_count_in_quint == 10)

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        max_iters = 80000
        temp_start = 3.0
        temp_end = 0.005

        # Precompute which triples are in violating structures for faster sampling
        violating_quad_triples = set()
        for qi in range(num_quads):
            if red_count_in_quad[qi] == 4:
                for tidx in quad_triples_idx[qi]:
                    violating_quad_triples.add(tidx)
        
        violating_quint_triples = set()
        for qi in range(num_quints):
            if blue_count_in_quint[qi] == 10:
                for tidx in quint_triples_idx[qi]:
                    violating_quint_triples.add(tidx)

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip with bias towards violating structures
            tidx = -1
            r = random.random()
            
            if rk4 > 0 and r < 0.4:
                # Pick from violating red K4 triples
                if violating_quad_triples:
                    tidx = random.choice(list(violating_quad_triples))
                else:
                    tidx = random.randint(0, num_triples - 1)
            
            elif bk5 > 0 and r < 0.7:
                # Pick from violating blue K5 triples
                if violating_quint_triples:
                    tidx = random.choice(list(violating_quint_triples))
                else:
                    tidx = random.randint(0, num_triples - 1)
            
            else:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]
            new_color = 1 - old_color

            # Compute delta in violations
            delta_rk4 = 0
            delta_bk5 = 0

            # Update red counts for quads containing tidx
            for qi in triple_in_quads[tidx]:
                if old_color == 0:
                    old_rc = red_count_in_quad[qi]
                    new_rc = old_rc - 1
                    if old_rc == 4:
                        delta_rk4 -= 1
                    if new_rc == 4:
                        delta_rk4 += 1
                    red_count_in_quad[qi] = new_rc
                else:
                    old_rc = red_count_in_quad[qi]
                    new_rc = old_rc + 1
                    if old_rc == 4:
                        delta_rk4 -= 1
                    if new_rc == 4:
                        delta_rk4 += 1
                    red_count_in_quad[qi] = new_rc

            # Update blue counts for quints containing tidx
            for qi in triple_in_quints[tidx]:
                if old_color == 1:
                    old_bc = blue_count_in_quint[qi]
                    new_bc = old_bc - 1
                    if old_bc == 10:
                        delta_bk5 -= 1
                    if new_bc == 10:
                        delta_bk5 += 1
                    blue_count_in_quint[qi] = new_bc
                else:
                    old_bc = blue_count_in_quint[qi]
                    new_bc = old_bc + 1
                    if old_bc == 10:
                        delta_bk5 -= 1
                    if new_bc == 10:
                        delta_bk5 += 1
                    blue_count_in_quint[qi] = new_bc

            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                color_arr[tidx] = new_color
                rk4 += delta_rk4
                bk5 += delta_bk5
                
                # Update violating structure sets
                if delta_rk4 != 0:
                    violating_quad_triples = set()
                    for qi in range(num_quads):
                        if red_count_in_quad[qi] == 4:
                            for t in quad_triples_idx[qi]:
                                violating_quad_triples.add(t)
                if delta_bk5 != 0:
                    violating_quint_triples = set()
                    for qi in range(num_quints):
                        if blue_count_in_quint[qi] == 10:
                            for t in quint_triples_idx[qi]:
                                violating_quint_triples.add(t)
            else:
                # Revert counts
                for qi in triple_in_quads[tidx]:
                    if old_color == 0:
                        red_count_in_quad[qi] += 1
                    else:
                        red_count_in_quad[qi] -= 1
                for qi in triple_in_quints[tidx]:
                    if old_color == 1:
                        blue_count_in_quint[qi] += 1
                    else:
                        blue_count_in_quint[qi] -= 1

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True
        return color_arr.tolist(), False

    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = build_structures(target_n)

        found = False
        max_restarts = 300 if target_n <= 20 else 150

        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break

            # Various initialization strategies
            if restart == 0:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 1:
                init = [0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 2:
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 3:
                init = [0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 4:
                init = list(np.random.randint(0, 2, size=num_triples))
            elif restart == 5:
                init = [0 if random.random() < 0.35 else 1 for _ in range(num_triples)]
            elif restart == 6:
                init = [0 if random.random() < 0.25 else 1 for _ in range(num_triples)]
            elif restart == 7:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
                for i in range(num_triples):
                    if random.random() < 0.1:
                        init[i] = 1 - init[i]
            elif restart == 8:
                init = [0 if a % 2 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 9:
                init = [0 if c % 3 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 10:
                # Use modular with different base
                init = [0 if (a*2+b*3+c*5) % 7 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 11:
                init = [0 if (a+b+c) % 6 == 0 else 1 for (a,b,c) in triple_list]
            else:
                base = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
                init = base[:]
                for i in range(num_triples):
                    if random.random() < 0.15:
                        init[i] = 1 - init[i]

            colors, success = local_search(target_n, init, deadline)
            if success:
                found = True
                if is_valid(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    break
                else:
                    found = False

        if found:
            continue
        if target_n > 12:
            break

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
