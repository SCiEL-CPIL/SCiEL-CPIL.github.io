import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_triple_indices(n):
        return list(combinations(range(n), 3))

    def build_structure(n):
        triples = get_triple_indices(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        k4s_arr = np.array(k4s, dtype=np.int32) if k4s else np.empty((0, 4), dtype=np.int32)
        k5s_arr = np.array(k5s, dtype=np.int32) if k5s else np.empty((0, 10), dtype=np.int32)

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s, k4s_arr, k5s_arr

    def compute_violations_vectorized(colors, k4s_arr, k5s_arr):
        """Returns (num_red_k4, num_blue_k5, violated_k4_set, violated_k5_set)"""
        red_k4_count = int(np.all(colors[k4s_arr] == 0, axis=1).sum()) if len(k4s_arr) > 0 else 0
        blue_k5_count = int(np.all(colors[k5s_arr] == 1, axis=1).sum()) if len(k5s_arr) > 0 else 0
        violated_k4s = set(np.where(np.all(colors[k4s_arr] == 0, axis=1))[0].tolist()) if len(k4s_arr) > 0 else set()
        violated_k5s = set(np.where(np.all(colors[k5s_arr] == 1, axis=1))[0].tolist()) if len(k5s_arr) > 0 else set()
        return red_k4_count, blue_k5_count, violated_k4s, violated_k5s

    def greedy_repair(n, time_limit, seed=0, init_variant=0):
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s, k4s_arr, k5s_arr = build_structure(n)
        np.random.seed(seed)

        # Initialize
        if init_variant == 0:
            colors = np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 1:
            colors = np.array([0 if (a + b + 2*c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 2:
            colors = np.array([(a*b + b*c + c*a) % 2 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 3:
            colors = np.array([(a + b + c) % 2 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 4:
            colors = np.array([(a*b*c) % 2 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 5:
            colors = np.array([0 if (a + b) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 6:
            colors = np.array([0 if (b + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 7:
            colors = np.array([0 if (a + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 8:
            colors = np.random.randint(0, 2, num_triples).astype(np.int8)
        elif init_variant == 9:
            pair_vals = np.random.randint(0, 3, (n, n))
            colors = np.array([0 if (pair_vals[a,b] + pair_vals[b,c] + pair_vals[a,c]) % 3 == 0 else 1 
                              for a, b, c in triples], dtype=np.int8)
        elif init_variant == 10:
            pair_vals = np.random.randint(0, 3, (n, n))
            colors = np.array([0 if (pair_vals[a,b] + 2*pair_vals[b,c] + pair_vals[a,c]) % 3 == 0 else 1 
                              for a, b, c in triples], dtype=np.int8)
        elif init_variant == 11:
            colors = np.array([(a*b + b*c) % 2 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 12:
            colors = np.array([0 if c % 2 == 0 else 1 for a, b, c in triples], dtype=np.int8)
        elif init_variant == 13:
            colors = np.array([0 if (a+b+c) % 2 == 0 and (a*b + b*c + c*a) % 2 == 0 else 1 
                              for a, b, c in triples], dtype=np.int8)
        else:
            colors = np.array([0 if (a + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)

        red_k4_count, blue_k5_count, violated_k4s, violated_k5s = compute_violations_vectorized(colors, k4s_arr, k5s_arr)
        best_cost = red_k4_count + blue_k5_count
        best_colors = colors.copy()

        if best_cost == 0:
            return colors, True

        # Precompute for each triple: which K4s and K5s it belongs to
        # For greedy: for each violated K4/K5, pick a triple in it to flip
        # Priority: red K4s first (harder constraint)

        no_improve = 0
        max_no_improve = 5000
        stagnation_restarts = 0

        while time.time() < time_limit:
            # Check if we've solved it
            if not violated_k4s and not violated_k5s:
                return colors, True

            # Choose which violation to fix
            # Prioritize red K4s over blue K5s
            if violated_k4s and (not violated_k5s or np.random.rand() < 0.7):
                # Pick a random violated K4
                k4_idx = np.random.choice(list(violated_k4s))
                idxs = k4s[k4_idx]
                # Pick a triple in this K4 to flip
                t = idxs[np.random.randint(4)]
            elif violated_k5s:
                k5_idx = np.random.choice(list(violated_k5s))
                idxs = k5s[k5_idx]
                t = idxs[np.random.randint(10)]
            else:
                # No violations? Shouldn't happen, but break
                break

            # Compute delta cost of flipping triple t
            old_color = colors[t]
            new_color = 1 - old_color

            # Count how this flip affects K4s and K5s containing t
            delta = 0
            for k4_i in triple_to_k4s[t]:
                idxs4 = k4s[k4_i]
                # Current red count in this K4
                red_count = sum(1 for i in idxs4 if colors[i] == 0)
                if old_color == 0:
                    # Removing a red: if all 4 were red, we fix it
                    if red_count == 4:
                        delta -= 1
                else:
                    # Adding a red: if 3 were red, we create a violation
                    if red_count == 3:
                        delta += 1

            for k5_i in triple_to_k5s[t]:
                idxs5 = k5s[k5_i]
                blue_count = sum(1 for i in idxs5 if colors[i] == 1)
                if old_color == 1:
                    # Removing a blue: if all 10 were blue, we fix it
                    if blue_count == 10:
                        delta -= 1
                else:
                    # Adding a blue: if 9 were blue, we create a violation
                    if blue_count == 9:
                        delta += 1

            # Accept if it doesn't increase cost, or with some probability if it does
            if delta <= 0:
                accept = True
            else:
                # Allow some worsening moves to escape local optima
                accept = np.random.rand() < 0.15

            if accept:
                colors[t] = new_color

                # Update violation sets incrementally
                for k4_i in triple_to_k4s[t]:
                    idxs4 = k4s[k4_i]
                    if all(colors[j] == 0 for j in idxs4):
                        violated_k4s.add(k4_i)
                    else:
                        violated_k4s.discard(k4_i)

                for k5_i in triple_to_k5s[t]:
                    idxs5 = k5s[k5_i]
                    if all(colors[j] == 1 for j in idxs5):
                        violated_k5s.add(k5_i)
                    else:
                        violated_k5s.discard(k5_i)

                current_cost = len(violated_k4s) + len(violated_k5s)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve = 0
                    if best_cost == 0:
                        return best_colors, True
                else:
                    no_improve += 1
            else:
                no_improve += 1

            # Stagnation handling: restart from best with perturbation
            if no_improve > max_no_improve:
                stagnation_restarts += 1
                if stagnation_restarts > 10:
                    # Give up on this seed
                    break
                # Restart from best with random perturbation
                colors = best_colors.copy()
                # Perturb 5% of triples
                n_perturb = max(1, num_triples // 20)
                for _ in range(n_perturb):
                    t_perturb = np.random.randint(num_triples)
                    colors[t_perturb] = 1 - colors[t_perturb]
                red_k4_count, blue_k5_count, violated_k4s, violated_k5s = compute_violations_vectorized(colors, k4s_arr, k5s_arr)
                no_improve = 0

        return best_colors, (best_cost == 0)

    def verify(n, colors):
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    final_n = 0
    final_colors = []

    target_ns = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        remaining = deadline - time.time()
        if remaining < 3:
            break

        if n == 35:
            time_budget = min(60.0, remaining * 0.8)
        elif n == 34:
            time_budget = min(50.0, remaining * 0.7)
        elif n >= 30:
            time_budget = min(25.0, remaining * 0.5)
        elif n >= 25:
            time_budget = min(15.0, remaining * 0.4)
        elif n >= 20:
            time_budget = min(10.0, remaining * 0.3)
        else:
            time_budget = min(5.0, remaining * 0.2)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget

        found = False
        num_seeds = 40 if n >= 35 else 35 if n >= 34 else 25

        for seed in range(num_seeds):
            if time.time() > time_limit:
                break

            variant = seed % 14

            try:
                colors, is_valid = greedy_repair(n, time_limit, seed=seed, init_variant=variant)

                if is_valid:
                    if verify(n, colors.tolist()):
                        final_n = n
                        final_colors = colors.tolist()
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    if final_n == 0 or not final_colors:
        n = 9
        triples = get_triple_indices(n)
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            np.random.seed(42)
            for _ in range(1000):
                colors = [np.random.randint(0, 2) for _ in range(len(triples))]
                if verify(n, colors):
                    final_n = n
                    final_colors = colors
                    break
            else:
                final_n = 1
                final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
