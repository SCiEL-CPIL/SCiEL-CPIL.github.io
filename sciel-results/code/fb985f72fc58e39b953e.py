import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_cliques(n):
        triples = get_triples(n)
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

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def count_violations(colors, k4s, k5s):
        red = 0
        for idxs in k4s:
            if all(colors[i] == 0 for i in idxs):
                red += 1
        blue = 0
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                blue += 1
        return red + blue

    def delta_cost(colors, t, old_color, k4s, k5s, triple_to_k4s, triple_to_k5s):
        delta = 0
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            red_count = sum(1 for i in idxs if colors[i] == 0)
            if old_color == 0:
                if red_count == 4:
                    delta -= 1
            else:
                if red_count == 3:
                    delta += 1
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            blue_count = sum(1 for i in idxs if colors[i] == 1)
            if old_color == 1:
                if blue_count == 10:
                    delta -= 1
            else:
                if blue_count == 9:
                    delta += 1
        return delta

    def pair_to_triple_colors(n, pair_vals, coeff_a, coeff_b, coeff_c, threshold_map):
        """
        Given pair values in F3 and coefficients, compute triple colors.
        triple(a,b,c) value = coeff_a*v_ab + coeff_b*v_bc + coeff_c*v_ac (mod 3)
        Map to binary: threshold_map[value] -> 0 or 1
        """
        triples = get_triples(n)
        colors = np.zeros(len(triples), dtype=np.int8)
        for idx, (a, b, c) in enumerate(triples):
            v_ab = pair_vals[a * n + b]  # flat index for pair (a,b)
            v_bc = pair_vals[b * n + c]
            v_ac = pair_vals[a * n + c]
            s = (coeff_a * v_ab + coeff_b * v_bc + coeff_c * v_ac) % 3
            colors[idx] = threshold_map[s]
        return colors

    def local_search_refine(n, colors, time_limit, k4s, k5s, triple_to_k4s, triple_to_k5s, seed=0):
        """Local search to refine a starting coloring."""
        np.random.seed(seed)
        colors = colors.copy()
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()
        num_triples = len(colors)

        if current_cost == 0:
            return colors, True

        temp = 1.0
        cool_rate = 0.9997
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 2000

        while time.time() < time_limit:
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color

            dc = delta_cost(colors, t, old_color, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + dc

            if dc <= 0:
                accept = True
            else:
                if temp > min_temp:
                    accept = np.random.rand() < np.exp(-dc / temp)
                else:
                    accept = False

            if accept:
                colors[t] = new_color
                current_cost = new_cost
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

            temp *= cool_rate

            if no_improve > max_no_improve and temp > min_temp:
                temp = max(0.5, temp * 1.5)
                no_improve = 0

            if no_improve > max_no_improve * 5:
                colors = best_colors.copy()
                current_cost = best_cost
                temp = max(0.8, temp)
                no_improve = 0

        return best_colors, (best_cost == 0)

    def genetic_algorithm(n, time_limit, seed=0):
        """
        Genetic algorithm over F3 pair-value assignments.
        Genome: (pair_vals, coeff_a, coeff_b, coeff_c, threshold_map)
        pair_vals: array of length n*n with values in {0,1,2} for pairs (i,j) where i<j
        coeff_a, coeff_b, coeff_c: in {1, 2} (non-zero in F3)
        threshold_map: maps {0,1,2} -> {0,1}
        """
        np.random.seed(seed)

        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_cliques(n)

        # Number of pairs
        num_pairs = n * (n - 1) // 2

        def encode_genome(pair_vals, ca, cb, cc, tm):
            """Encode genome as a flat array for GA operations."""
            return np.concatenate([pair_vals, [ca, cb, cc], list(tm.values())])

        def decode_genome(g):
            """Decode genome back to components."""
            pair_vals = g[:num_pairs].astype(np.int8)
            ca = int(g[num_pairs])
            cb = int(g[num_pairs + 1])
            cc = int(g[num_pairs + 2])
            tm = {0: int(g[num_pairs + 3]), 1: int(g[num_pairs + 4]), 2: int(g[num_pairs + 5])}
            return pair_vals, ca, cb, cc, tm

        def fitness(pair_vals, ca, cb, cc, tm):
            """Lower is better. Return number of violations."""
            colors = pair_to_triple_colors(n, pair_vals, ca, cb, cc, tm)
            return count_violations(colors, k4s, k5s)

        # Initialize population
        pop_size = 30
        population = []
        for i in range(pop_size):
            # Random pair values
            pair_vals = np.random.randint(0, 3, num_pairs).astype(np.int8)
            # Random coefficients (non-zero in F3)
            ca = np.random.randint(1, 3)
            cb = np.random.randint(1, 3)
            cc = np.random.randint(1, 3)
            # Random threshold map
            tm = {0: np.random.randint(0, 2), 1: np.random.randint(0, 2), 2: np.random.randint(0, 2)}
            population.append((pair_vals.copy(), ca, cb, cc, dict(tm)))

        # Evaluate initial population
        fitnesses = [fitness(pv, ca, cb, cc, tm) for pv, ca, cb, cc, tm in population]

        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_genome = population[best_idx]

        if best_fitness == 0:
            colors = pair_to_triple_colors(n, best_genome[0], best_genome[1], best_genome[2], best_genome[3], best_genome[4])
            return colors, True

        # GA loop
        gen = 0
        max_gens = 200
        while time.time() < time_limit:
            gen += 1

            # Sort by fitness
            order = np.argsort(fitnesses)
            population = [population[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            # Elitism: keep top 5
            elite = population[:5]

            # Crossover and mutation
            new_population = list(elite)

            while len(new_population) < pop_size:
                # Tournament selection
                def select():
                    idxs = np.random.randint(0, pop_size, 3)
                    best = min(idxs, key=lambda x: fitnesses[x])
                    return population[best]

                parent1 = select()
                parent2 = select()

                # Crossover: mix pair values
                mask = np.random.rand(num_pairs) < 0.5
                child_pv = np.where(mask, parent1[0], parent2[0]).copy()

                # Inherit coefficients and threshold map from one parent
                if np.random.rand() < 0.5:
                    child_ca, child_cb, child_cc = parent1[1], parent1[2], parent1[3]
                    child_tm = dict(parent1[4])
                else:
                    child_ca, child_cb, child_cc = parent2[1], parent2[2], parent2[3]
                    child_tm = dict(parent2[4])

                # Mutation
                # Mutate some pair values
                num_mut = np.random.randint(1, max(2, num_pairs // 10) + 1)
                mut_idx = np.random.choice(num_pairs, num_mut, replace=False)
                child_pv[mut_idx] = np.random.randint(0, 3, num_mut)

                # Mutate coefficients with small probability
                if np.random.rand() < 0.1:
                    child_ca = np.random.randint(1, 3)
                if np.random.rand() < 0.1:
                    child_cb = np.random.randint(1, 3)
                if np.random.rand() < 0.1:
                    child_cc = np.random.randint(1, 3)

                # Mutate threshold map with small probability
                if np.random.rand() < 0.1:
                    key = np.random.randint(0, 3)
                    child_tm[key] = np.random.randint(0, 2)

                new_population.append((child_pv, child_ca, child_cb, child_cc, child_tm))

            # Evaluate new population
            fitnesses = [fitness(pv, ca, cb, cc, tm) for pv, ca, cb, cc, tm in new_population]
            population = new_population

            # Update best
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < best_fitness:
                best_fitness = fitnesses[best_idx]
                best_genome = population[best_idx]
                if best_fitness == 0:
                    colors = pair_to_triple_colors(n, best_genome[0], best_genome[1], best_genome[2], best_genome[3], best_genome[4])
                    return colors, True

        # Return best from GA
        colors = pair_to_triple_colors(n, best_genome[0], best_genome[1], best_genome[2], best_genome[3], best_genome[4])
        return colors, (count_violations(colors, k4s, k5s) == 0)

    def verify(n, colors):
        triples = get_triples(n)
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

        if n >= 33:
            time_budget = min(50.0, remaining * 0.7)
        elif n >= 30:
            time_budget = min(35.0, remaining * 0.6)
        elif n >= 25:
            time_budget = min(20.0, remaining * 0.5)
        elif n >= 20:
            time_budget = min(12.0, remaining * 0.4)
        else:
            time_budget = min(8.0, remaining * 0.3)

        if time_budget < 2.0:
            break

        time_limit = time.time() + time_budget

        found = False
        num_seeds = 8 if n >= 33 else 6

        for seed in range(num_seeds):
            if time.time() > time_limit - 2:
                break

            try:
                # Phase 1: Genetic algorithm over F3 pair values
                ga_time = time_limit - (time.time() - (time_limit - time_budget)) * 0.6
                ga_time = min(ga_time, time_limit - 5)
                if ga_time < 3:
                    ga_time = time_limit - 2

                colors, is_valid = genetic_algorithm(n, time.time() + ga_time, seed=seed)

                if is_valid and verify(n, colors.tolist()):
                    final_n = n
                    final_colors = colors.tolist()
                    found = True
                    break

                # Phase 2: Local search refinement on GA result
                if time.time() < time_limit - 2:
                    triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_cliques(n)
                    colors, is_valid = local_search_refine(n, colors, time_limit - 2, k4s, k5s, triple_to_k4s, triple_to_k5s, seed=seed + 100)

                    if is_valid and verify(n, colors.tolist()):
                        final_n = n
                        final_colors = colors.tolist()
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    # Fallback for small n
    if final_n == 0 or not final_colors:
        for n in [9, 8, 7, 6, 5, 4, 3, 2, 1]:
            triples = get_triples(n)
            # Try structured coloring
            colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
            if verify(n, colors):
                final_n = n
                final_colors = colors
                break
            # Try random
            np.random.seed(42)
            for _ in range(500):
                colors = [np.random.randint(0, 2) for _ in range(len(triples))]
                if verify(n, colors):
                    final_n = n
                    final_colors = colors
                    break
            if final_n > 0:
                break

    if final_n == 0:
        final_n = 1
        final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
