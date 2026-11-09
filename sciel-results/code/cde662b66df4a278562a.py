import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        if len(colors) != len(list(combinations(range(n), 3))):
            return False
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

        # Bitsets for quads and quints (using Python ints as arbitrary precision bitsets)
        quad_bitsets = []
        for qt in quad_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quad_bitsets.append(bs)

        quint_bitsets = []
        for qt in quint_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quint_bitsets.append(bs)

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
                quad_bitsets, quint_bitsets)

    def local_search(target_n, init_colors, deadline, structures):
        """Run local search to eliminate violations. Returns (colors, success)."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = structures

        color_arr = np.array(init_colors, dtype=np.int8)
        
        # red_mask: bit i is 1 if triple i is red (color 0)
        red_mask = 0
        for i in range(num_triples):
            if color_arr[i] == 0:
                red_mask |= (1 << i)

        # Count initial violations
        rk4 = 0
        for qbs in quad_bitsets:
            if (red_mask & qbs) == qbs:
                rk4 += 1
        bk5 = 0
        for qbs in quint_bitsets:
            if (red_mask & qbs) == 0:
                bk5 += 1

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        max_iters = 200000
        temp_start = 2.0
        temp_end = 0.001

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip
            tidx = -1
            r = random.random()
            
            # Heuristic: pick from violating structures if they exist
            if rk4 > 0 and r < 0.5:
                # Find a violating red K4
                for _ in range(20):
                    qi = random.randint(0, num_quads - 1)
                    if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                        qt = quad_triples_idx[qi]
                        tidx = qt[random.randint(0, 3)]
                        break
            elif bk5 > 0 and r < 0.85:
                # Find a violating blue K5
                for _ in range(20):
                    qi = random.randint(0, num_quints - 1)
                    if (red_mask & quint_bitsets[qi]) == 0:
                        qt = quint_triples_idx[qi]
                        tidx = qt[random.randint(0, 9)]
                        break

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]
            new_color = 1 - old_color

            # Compute delta in violations
            delta_rk4 = 0
            delta_bk5 = 0

            # Update red counts for quads containing tidx
            for qi in triple_in_quads[tidx]:
                qt = quad_triples_idx[qi]
                # Check if other 3 triples in this quad are red
                others_all_red = True
                for p in range(4):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 0:
                        others_all_red = False
                        break
                
                if others_all_red:
                    if old_color == 0:
                        # Was red, becomes blue -> breaks a red K4
                        delta_rk4 -= 1
                    else:
                        # Was blue, becomes red -> creates a red K4
                        delta_rk4 += 1

            # Update blue counts for quints containing tidx
            for qi in triple_in_quints[tidx]:
                qt = quint_triples_idx[qi]
                # Check if other 9 triples in this quint are blue
                others_all_blue = True
                for p in range(10):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 1:
                        others_all_blue = False
                        break
                
                if others_all_blue:
                    if old_color == 1:
                        # Was blue, becomes red -> breaks a blue K5
                        delta_bk5 -= 1
                    else:
                        # Was red, becomes blue -> creates a blue K5
                        delta_bk5 += 1

            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                color_arr[tidx] = new_color
                if new_color == 0:
                    red_mask |= (1 << tidx)
                else:
                    red_mask &= ~(1 << tidx)
                rk4 += delta_rk4
                bk5 += delta_bk5
            else:
                # Revert: do nothing, counts unchanged
                pass

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True
        return color_arr.tolist(), False

    # Try increasing n
    # Start from a reasonable n
    # We know n=9 is valid. Let's try up to 35.
    # Cache structures for efficiency if we restart n? No, n changes structures.
    
    for target_n in range(10, 36):
        if time.time() > deadline - 15:
            break

        # Build structures for this n
        structures = build_structures(target_n)
        triple_list, num_triples = structures[0], structures[1]

        found = False
        # Adaptive restarts: more for smaller n, fewer for larger n to save time
        max_restarts = 100 if target_n <= 20 else 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
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
                # Random with red bias (more reds might help avoid blue K5s initially?)
                init = [0 if random.random() < 0.4 else 1 for _ in range(num_triples)]
            elif restart == 6:
                # Random with blue bias
                init = [0 if random.random() < 0.3 else 1 for _ in range(num_triples)]
            elif restart == 7:
                # Perturbed modular
                base = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
                init = base[:]
                for i in range(num_triples):
                    if random.random() < 0.1:
                        init[i] = 1 - init[i]
            elif restart == 8:
                # Vertex-based: min vertex mod 2
                init = [0 if a % 2 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 9:
                # Vertex-based: max vertex mod 3
                init = [0 if c % 3 == 0 else 1 for (a,b,c) in triple_list]
            else:
                # Random perturbation of mod 4
                base = [0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list]
                init = base[:]
                for i in range(num_triples):
                    if random.random() < 0.15:
                        init[i] = 1 - init[i]

            colors, success = local_search(target_n, init, deadline, structures)
            if success:
                if is_valid(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
                else:
                    found = False

        if found:
            continue
        # If we couldn't find valid coloring for this n, stop trying larger n
        # unless we are very close to deadline and haven't tried much
        if target_n > 12 and not found:
             # Try a few more restarts for this n if time allows
             if time.time() < deadline - 20:
                 for restart in range(max_restarts, max_restarts + 50):
                     if time.time() > deadline - 5:
                         break
                     init = list(np.random.randint(0, 2, size=num_triples))
                     colors, success = local_search(target_n, init, deadline, structures)
                     if success and is_valid(target_n, colors):
                         best_n = target_n
                         best_colors = colors
                         found = True
                         break
             if not found:
                 break

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
