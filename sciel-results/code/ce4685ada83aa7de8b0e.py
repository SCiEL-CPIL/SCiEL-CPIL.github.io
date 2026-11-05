import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_triple_indices(n):
        return list(combinations(range(n), 3))

    def build_structures(n):
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

        k4_masks = []
        for idxs in k4s:
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            k4_masks.append(mask)

        k5_masks = []
        for idxs in k5s:
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            k5_masks.append(mask)

        return {
            'triples': triples,
            'num_triples': num_triples,
            'k4s': k4s,
            'k5s': k5s,
            'k4_masks': k4_masks,
            'k5_masks': k5_masks
        }

    def count_violations(colors, struct):
        k4_masks = struct['k4_masks']
        k5_masks = struct['k5_masks']
        red_count = 0
        for mask in k4_masks:
            if (colors & mask) == 0:
                red_count += 1
        blue_count = 0
        for mask in k5_masks:
            if (colors & mask) == mask:
                blue_count += 1
        return red_count + blue_count

    def delta_cost(colors, t, struct, triples):
        """Compute cost delta when flipping triple t."""
        delta = 0
        bit = 1 << t
        is_blue = (colors & bit) != 0
        a, b, c = triples[t]

        # K4s containing t: find the 4th vertex
        for d in range(struct['num_triples'] // struct['num_triples'] * 0 + 0, 0):
            pass

        # Check all K4s containing this triple
        # A triple (a,b,c) is in K4 with any d > max(a,b,c) or d between them
        # More efficiently: iterate over possible 4th vertices
        verts = set([a, b, c])
        for d in range(struct['num_triples'] // struct['num_triples'] * 0 + 0, 0):
            pass

        # Simpler: precompute which K4s contain each triple
        # For now, use direct check
        for k4_idx, idxs in enumerate(struct['k4s']):
            if t in idxs:
                other_reds = 0
                for ot in idxs:
                    if ot != t and ((colors >> ot) & 1) == 0:
                        other_reds += 1
                if is_blue:
                    if other_reds == 3:
                        delta += 1
                else:
                    if other_reds == 3:
                        delta -= 1

        for k5_idx, idxs in enumerate(struct['k5s']):
            if t in idxs:
                other_blues = 0
                for ot in idxs:
                    if ot != t and ((colors >> ot) & 1) == 1:
                        other_blues += 1
                if is_blue:
                    if other_blues == 9:
                        delta -= 1
                else:
                    if other_blues == 9:
                        delta += 1

        return delta

    def generate_coloring(n, params, struct):
        """Generate a coloring from a parameter vector.
        params layout: [c0,c1,c2, v0..v(n-1), noise_seed]
        c0,c1,c2: coefficients for linear form in vertex indices
        v0..v(n-1): vertex offsets
        noise_seed: seed for random perturbation
        """
        triples = struct['triples']
        num_triples = struct['num_triples']
        mask = 0

        c0 = params[0]
        c1 = params[1]
        c2 = params[2]
        offsets = params[3:3+n]
        noise_seed = int(params[3+n]) if len(params) > 3+n else 0

        rng = np.random.RandomState(noise_seed % (2**31 - 1))
        noise = rng.randint(0, 2, size=num_triples)

        for i, (a, b, c) in enumerate(triples):
            # Linear form: c0*a + c1*b + c2*c + offsets
            val = (c0 * a + c1 * b + c2 * c + offsets[a] + offsets[b] + offsets[c]) % 3
            # Map to binary: 0->0, 1->1, 2->0 (or use threshold)
            base_color = 1 if val == 1 else 0
            # Add noise with small probability
            if noise[i] == 1:
                base_color = 1 - base_color
            if base_color:
                mask |= (1 << i)

        return mask

    def local_search(n, colors, struct, time_limit, rng):
        """Fast local search with delta moves."""
        triples = struct['triples']
        num_triples = struct['num_triples']
        current_cost = count_violations(colors, struct)
        best_cost = current_cost
        best_colors = colors

        # Precompute for each triple which K4s and K5s it belongs to
        triple_to_k4 = [[] for _ in range(num_triples)]
        for k4_idx, idxs in enumerate(struct['k4s']):
            for t in idxs:
                triple_to_k4[t].append(k4_idx)

        triple_to_k5 = [[] for _ in range(num_triples)]
        for k5_idx, idxs in enumerate(struct['k5s']):
            for t in idxs:
                triple_to_k5[t].append(k5_idx)

        T = 2.0
        T_min = 0.05
        cooling = 0.999

        while time.time() < time_limit and T > T_min:
            t = rng.randint(num_triples)
            bit = 1 << t
            is_blue = (colors & bit) != 1

            dc = 0
            # K4s containing t
            for k4_idx in triple_to_k4[t]:
                idxs = struct['k4s'][k4_idx]
                other_reds = 0
                for ot in idxs:
                    if ot != t and ((colors >> ot) & 1) == 0:
                        other_reds += 1
                if is_blue:
                    if other_reds == 3:
                        dc += 1
                else:
                    if other_reds == 3:
                        dc -= 1

            # K5s containing t
            for k5_idx in triple_to_k5[t]:
                idxs = struct['k5s'][k5_idx]
                other_blues = 0
                for ot in idxs:
                    if ot != t and ((colors >> ot) & 1) == 1:
                        other_blues += 1
                if is_blue:
                    if other_blues == 9:
                        dc -= 1
                else:
                    if other_blues == 9:
                        dc += 1

            new_cost = current_cost + dc
            if dc < 0 or rng.rand() < np.exp(-dc / T):
                colors ^= bit
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors
                    if best_cost == 0:
                        return best_colors

            T *= cooling

        return best_colors

    def genetic_algorithm(n, time_limit, seed, struct):
        """GA over parameter vectors."""
        rng = np.random.RandomState(seed)
        num_triples = struct['num_triples']
        param_dim = 3 + n + 1  # c0,c1,c2, offsets, noise_seed

        # Initialize population
        pop_size = 40
        population = []
        for i in range(pop_size):
            params = rng.randint(0, 3, size=3 + n)
            params[3+n] = rng.randint(0, 10000)
            population.append(params)

        # Evaluate initial population
        def evaluate(params):
            colors = generate_coloring(n, params, struct)
            cost = count_violations(colors, struct)
            return cost, colors

        scores = []
        colorings = []
        for params in population:
            cost, colors = evaluate(params)
            scores.append(cost)
            colorings.append(colors)

        best_score = min(scores)
        best_colors = colorings[scores.index(best_score)]

        if best_score == 0:
            return best_colors

        # Local search on best
        local_limit = time.time() + min(2.0, (time_limit - time.time()) * 0.1)
        best_colors = local_search(n, best_colors, struct, local_limit, rng)
        best_score = count_violations(best_colors, struct)
        if best_score == 0:
            return best_colors

        # GA loop
        elite_size = 8
        crossover_rate = 0.8
        mutation_rate = 0.3

        while time.time() < time_limit:
            # Sort by fitness
            order = np.argsort(scores)
            new_population = []
            new_colorings = []

            # Elitism
            for i in range(min(elite_size, pop_size)):
                idx = order[i]
                new_population.append(population[idx].copy())
                new_colorings.append(colorings[idx])

            # Crossover and mutation
            while len(new_population) < pop_size:
                if time.time() >= time_limit:
                    break

                # Tournament selection
                def tournament():
                    candidates = rng.choice(len(scores), size=3, replace=False)
                    best_c = min(candidates, key=lambda x: scores[x])
                    return best_c

                p1_idx = tournament()
                p2_idx = tournament()

                if rng.rand() < crossover_rate:
                    # Uniform crossover
                    mask = rng.randint(0, 2, size=param_dim)
                    child = np.where(mask == 1, population[p1_idx], population[p2_idx])
                else:
                    child = population[p1_idx].copy()

                # Mutation
                if rng.rand() < mutation_rate:
                    num_mut = rng.randint(1, 4)
                    for _ in range(num_mut):
                        pos = rng.randint(0, param_dim)
                        if pos < 3:
                            child[pos] = rng.randint(0, 3)
                        elif pos < 3 + n:
                            child[pos] = rng.randint(0, 3)
                        else:
                            child[pos] = rng.randint(0, 10000)

                new_population.append(child)
                # Evaluate
                cost, colors = evaluate(child)
                new_colorings.append(colors)

            population = new_population
            colorings = new_colorings
            scores = [count_violations(c, struct) for c in colorings]

            best_score = min(scores)
            best_idx = scores.index(best_score)
            if best_score < count_violations(best_colors, struct):
                best_colors = colorings[best_idx]
                if best_score == 0:
                    return best_colors

            # Periodic local search on best
            if time.time() < time_limit - 1.0:
                local_limit = time.time() + min(1.5, (time_limit - time.time()) * 0.05)
                improved = local_search(n, best_colors, struct, local_limit, rng)
                if count_violations(improved, struct) < count_violations(best_colors, struct):
                    best_colors = improved
                    if count_violations(best_colors, struct) == 0:
                        return best_colors

        return best_colors

    def verify(n, colors_bitmask):
        """Verify coloring has no red K4 and no blue K5."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            i1 = triple_to_idx[(a, b, c)]
            i2 = triple_to_idx[(a, b, d)]
            i3 = triple_to_idx[(a, c, d)]
            i4 = triple_to_idx[(b, c, d)]

            if ((colors_bitmask >> i1) & 1) == 0 and \
               ((colors_bitmask >> i2) & 1) == 0 and \
               ((colors_bitmask >> i3) & 1) == 0 and \
               ((colors_bitmask >> i4) & 1) == 0:
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                idx = triple_to_idx[triple]
                if ((colors_bitmask >> idx) & 1) != 1:
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
            time_budget = min(45.0, remaining * 0.5)
        elif n == 34:
            time_budget = min(35.0, remaining * 0.45)
        elif n >= 30:
            time_budget = min(20.0, remaining * 0.35)
        elif n >= 25:
            time_budget = min(12.0, remaining * 0.3)
        elif n >= 20:
            time_budget = min(8.0, remaining * 0.25)
        else:
            time_budget = min(5.0, remaining * 0.2)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget

        found = False
        num_seeds = 15 if n >= 35 else (10 if n >= 30 else 8)

        struct = build_structures(n)

        for seed in range(num_seeds):
            if time.time() > time_limit:
                break

            try:
                colors_bitmask = genetic_algorithm(n, time_limit, seed=seed, struct=struct)

                if count_violations(colors_bitmask, struct) == 0:
                    if verify(n, colors_bitmask):
                        num_triples = struct['num_triples']
                        colors_list = [int((colors_bitmask >> i) & 1) for i in range(num_triples)]
                        final_n = n
                        final_colors = colors_list
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    if final_n == 0 or not final_colors:
        # Fallback: try simple structured colorings for smaller n
        for n in range(12, 5, -1):
            if time.time() > deadline - 2:
                break
            triples = get_triple_indices(n)
            struct = build_structures(n)
            # Try several simple patterns
            for variant in range(6):
                mask = 0
                for i, (a, b, c) in enumerate(triples):
                    if variant == 0:
                        color = 1 if (a + b + c) % 3 != 0 else 0
                    elif variant == 1:
                        color = 1 if (a + 2*b + 3*c) % 3 != 0 else 0
                    elif variant == 2:
                        color = 1 if (a*b + b*c + c*a) % 2 == 1 else 0
                    elif variant == 3:
                        color = 1 if (a + b + c) % 2 == 1 else 0
                    elif variant == 4:
                        color = 1 if (a*c + b) % 3 != 0 else 0
                    else:
                        color = 1 if (a + b*c) % 2 == 1 else 0
                    if color:
                        mask |= (1 << i)
                if count_violations(mask, struct) == 0 and verify(n, mask):
                    final_n = n
                    final_colors = [int((mask >> i) & 1) for i in range(len(triples))]
                    break
            if final_n > 0:
                break

    if final_n == 0 or not final_colors:
        n = 1
        final_n = 1
        final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
