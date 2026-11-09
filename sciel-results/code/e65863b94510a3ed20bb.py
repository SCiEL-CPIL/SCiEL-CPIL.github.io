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
        """Precompute all structural mappings for a given n."""
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

        # Inverse mappings: for each triple, which quads/quints contain it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append((qi, pos))

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append((qi, pos))

        quad_triples_np = np.array(quad_triples_idx, dtype=np.int32)
        quint_triples_np = np.array(quint_triples_idx, dtype=np.int32)

        return {
            'triple_list': triple_list,
            'num_triples': num_triples,
            'quad_triples_np': quad_triples_np,
            'quint_triples_np': quint_triples_np,
            'num_quads': num_quads,
            'num_quints': num_quints,
            'triple_in_quads': triple_in_quads,
            'triple_in_quints': triple_in_quints,
        }

    def compute_violations(color_arr, structs):
        """Compute violation counts using vectorized numpy."""
        quad_colors = structs['quad_triples_np']
        quad_red_counts = np.sum(color_arr[quad_colors] == 0, axis=1)
        rk4 = int(np.sum(quad_red_counts == 4))

        quint_colors = structs['quint_triples_np']
        quint_blue_counts = np.sum(color_arr[quint_colors] == 1, axis=1)
        bk5 = int(np.sum(quint_blue_counts == 10))

        return rk4, bk5, quad_red_counts, quint_blue_counts

    def local_search(color_arr, structs, max_iters, deadline, temp_start=2.0, temp_end=0.01):
        """Run simulated annealing local search, return (colors, rk4, bk5)."""
        num_triples = structs['num_triples']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']

        rk4, bk5, quad_red_counts, quint_blue_counts = compute_violations(color_arr, structs)

        if rk4 == 0 and bk5 == 0:
            return color_arr, rk4, bk5

        it = 0
        while it < max_iters:
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                break

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip - bias toward violating structures
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.5:
                violating_quads = np.where(quad_red_counts == 4)[0]
                if len(violating_quads) > 0:
                    qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                    qt = quad_triples_np[qi]
                    tidx = int(qt[random.randint(0, 3)])
            elif bk5 > 0 and r < 0.85:
                violating_quints = np.where(quint_blue_counts == 10)[0]
                if len(violating_quints) > 0:
                    qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                    qt = quint_triples_np[qi]
                    tidx = int(qt[random.randint(0, 9)])

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = int(color_arr[tidx])

            # Compute delta_rk4 incrementally
            delta_rk4 = 0
            for (qi, pos) in triple_in_quads[tidx]:
                qt = quad_triples_np[qi]
                others = [qt[p] for p in range(4) if p != pos]
                others_red = all(int(color_arr[t]) == 0 for t in others)
                if old_color == 0 and others_red:
                    delta_rk4 -= 1
                elif old_color == 1 and others_red:
                    delta_rk4 += 1

            # Compute delta_bk5 incrementally
            delta_bk5 = 0
            for (qi, pos) in triple_in_quints[tidx]:
                qt = quint_triples_np[qi]
                others = [qt[p] for p in range(10) if p != pos]
                others_blue = all(int(color_arr[t]) == 1 for t in others)
                if old_color == 1 and others_blue:
                    delta_bk5 -= 1
                elif old_color == 0 and others_blue:
                    delta_bk5 += 1

            delta_total = delta_rk4 + delta_bk5

            # Acceptance criterion
            if delta_total <= 0:
                color_arr[tidx] = 1 - old_color
                rk4 += delta_rk4
                bk5 += delta_bk5
            else:
                if random.random() < np.exp(-delta_total / max(temp, 1e-10)):
                    color_arr[tidx] = 1 - old_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5

            it += 1

        return color_arr, rk4, bk5

    def generate_initial_coloring(target_n, restart_idx, num_triples, triple_list):
        """Generate diverse initial colorings."""
        if restart_idx == 0:
            return np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 1:
            return np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 2:
            return np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 3:
            return np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        elif restart_idx == 4:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 0
            return arr
        elif restart_idx == 5:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 1
            return arr
        else:
            base_mod = 3 + (restart_idx % 3)
            base = np.array([0 if (a+b+c) % base_mod == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            flip_mask = np.random.random(num_triples) < 0.15
            base[flip_mask] = 1 - base[flip_mask]
            return base

    def genetic_algorithm(target_n, structs, deadline, pop_size=30, generations=20):
        """
        Genetic algorithm with local search refinement.
        This is the structural change: instead of pure local search from
        multiple restarts, we maintain a population and use crossover + mutation.
        """
        num_triples = structs['num_triples']
        triple_list = structs['triple_list']

        # Initialize population
        population = []
        for i in range(pop_size):
            color_arr = generate_initial_coloring(target_n, i, num_triples, triple_list)
            population.append(color_arr)

        best_colors = None
        best_violations = float('inf')

        for gen in range(generations):
            if time.time() > deadline - 5:
                break

            # Evaluate fitness (violations) for each individual
            fitness_scores = []
            for ind in population:
                rk4, bk5, _, _ = compute_violations(ind, structs)
                fitness_scores.append(rk4 + bk5)

            # Find best in current generation
            min_idx = np.argmin(fitness_scores)
            if fitness_scores[min_idx] == 0:
                return population[min_idx]
            if fitness_scores[min_idx] < best_violations:
                best_violations = fitness_scores[min_idx]
                best_colors = population[min_idx].copy()

            # Tournament selection for parents
            def tournament_select(pool, scores, k=3):
                indices = [random.randint(0, len(pool)-1) for _ in range(k)]
                best = min(indices, key=lambda i: scores[i])
                return pool[best].copy(), scores[best]

            # Create next generation
            new_population = []

            # Elitism: keep best few
            sorted_indices = np.argsort(fitness_scores)
            elite_count = max(2, pop_size // 5)
            for i in range(elite_count):
                new_population.append(population[sorted_indices[i]].copy())

            # Fill rest with crossover + mutation + local search
            while len(new_population) < pop_size:
                if time.time() > deadline - 3:
                    break

                # Select parents via tournament
                parent1, f1 = tournament_select(population, fitness_scores)
                parent2, f2 = tournament_select(population, fitness_scores)

                # Uniform crossover
                mask = np.random.random(num_triples) < 0.5
                child = np.where(mask, parent1, parent2).astype(np.int8)

                # Mutation: flip some random triples
                mut_rate = 0.05
                mut_mask = np.random.random(num_triples) < mut_rate
                child[mut_mask] = 1 - child[mut_mask]

                # Apply local search to child
                child, rk4, bk5 = local_search(child, structs, max_iters=2000, deadline=deadline)

                new_population.append(child)

            population = new_population

        # Return best found
        if best_colors is not None:
            rk4, bk5, _, _ = compute_violations(best_colors, structs)
            if rk4 == 0 and bk5 == 0:
                return best_colors
        return None

    # Main loop: try increasing n with genetic algorithm + local search
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        num_triples = structs['num_triples']

        found = False

        # Strategy 1: Genetic Algorithm
        ga_result = genetic_algorithm(target_n, structs, deadline, pop_size=25, generations=15)
        if ga_result is not None:
            found = True

        # Strategy 2: If GA failed, try more local search restarts
        if not found:
            max_restarts = 50
            for restart in range(max_restarts):
                if time.time() > deadline - 3:
                    break

                color_arr = generate_initial_coloring(target_n, restart, num_triples, structs['triple_list'])
                color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=30000, deadline=deadline)

                if rk4 == 0 and bk5 == 0:
                    found = True
                    break

        if found:
            # Determine which solution to use
            if ga_result is not None:
                colors_list = ga_result.tolist()
            else:
                colors_list = color_arr.tolist()

            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                break
        else:
            if target_n >= 28:
                break

    return {'n': best_n, 'colors': best_colors}
