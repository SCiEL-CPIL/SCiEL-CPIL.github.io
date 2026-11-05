import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # For the specific problem: q=3, n=6, radius=1
    # Use direct sum of ternary tetracode (9 words of length 4) with all tails
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    else:
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4

    if n < base_n:
        return [list(range(n))]

    # Build initial code: direct sum
    tails = list(itertools.product(range(q), repeat=n - base_n))
    code = [tuple(h) + tuple(tail) for h in base_code for tail in tails]
    num_codewords = len(code)
    total_words = q ** n
    sphere_size = 1 + n * (q - 1)

    # Compute word indices for each codeword
    def compute_word_indices(codewords):
        indices = np.zeros(len(codewords), dtype=np.int64)
        for i in range(n):
            indices += codewords[:, i] * (q ** (n - 1 - i))
        return indices

    code_arr = np.array(code, dtype=np.int32)
    cw_indices = compute_word_indices(code_arr)

    # Precompute coverage: for each codeword, list of word indices it covers
    # And reverse: for each word, list of codewords that cover it
    codeword_covers = []
    covered_by = [[] for _ in range(total_words)]

    for ci in range(num_codewords):
        c = code_arr[ci]
        covers = set()
        covers.add(cw_indices[ci])
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_idx = cw_indices[ci] + (val - orig) * (q ** (n - 1 - pos))
                covers.add(new_idx)
        codeword_covers.append(list(covers))
        for wi in covers:
            covered_by[wi].append(ci)

    # Convert codeword_covers to numpy arrays for vectorized operations
    max_cover_size = max(len(c) for c in codeword_covers)
    codeword_covers_arr = np.full((num_codewords, max_cover_size), -1, dtype=np.int32)
    codeword_covers_len = np.zeros(num_codewords, dtype=np.int32)
    for ci in range(num_codewords):
        covers = codeword_covers[ci]
        codeword_covers_arr[ci, :len(covers)] = covers
        codeword_covers_len[ci] = len(covers)

    # Build adjacency: which codewords' balls overlap with codeword ci's ball
    # For each codeword, store the set of codewords that cover at least one of its words
    overlap_sets = []
    for ci in range(num_codewords):
        covers = codeword_covers[ci]
        neighbors = set()
        for wi in covers:
            for cj in covered_by[wi]:
                if cj != ci:
                    neighbors.add(cj)
        overlap_sets.append(neighbors)

    # Best code tracking
    best_code = None
    best_count = num_codewords

    # Helper: check if removing codeword ci is safe (all its words covered by others)
    def is_redundant(ci, coverage_count):
        covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
        return np.all(coverage_count[covers] > 1)

    # Helper: perform removal of codeword ci
    def remove_codeword(ci, active, coverage_count):
        active[ci] = False
        covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
        coverage_count[covers] -= 1

    # Helper: perform addition of codeword ci
    def add_codeword(ci, active, coverage_count):
        active[ci] = True
        covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
        coverage_count[covers] += 1

    # Compute fitness: sum of squared coverage counts (lower is better, indicates redundancy)
    def compute_fitness(coverage_count):
        return float(np.sum(coverage_count ** 2))

    # Genetic algorithm parameters
    population_size = 12
    elite_fraction = 0.2
    crossover_rate = 0.7
    mutation_rate = 0.3
    generations = 30

    # Initialize population with diverse starting points
    def greedy_remove_pass(active, coverage_count, order):
        """Perform greedy removal passes"""
        removed_any = True
        while removed_any:
            removed_any = False
            for ci in order:
                if not active[ci]:
                    continue
                if is_redundant(ci, coverage_count):
                    remove_codeword(ci, active, coverage_count)
                    removed_any = True

    # Create initial population
    population = []
    for trial in range(population_size):
        if time.time() > deadline - 10.0:
            break

        active = np.ones(num_codewords, dtype=bool)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
            coverage_count[covers] += 1

        # Random order for removal
        order = list(range(num_codewords))
        random.shuffle(order)

        # Greedy removal
        greedy_remove_pass(active, coverage_count, order)

        # Some swap-based local search
        active_list = np.where(active)[0]
        inactive_list = np.where(~active)[0]
        if len(inactive_list) > 0 and len(active_list) > 0:
            for _ in range(200):
                if time.time() > deadline - 10.0:
                    break
                ci_remove = int(random.choice(active_list))
                ci_add = int(random.choice(inactive_list))

                covers_remove = codeword_covers_arr[ci_remove, :codeword_covers_len[ci_remove]]
                covers_add = codeword_covers_arr[ci_add, :codeword_covers_len[ci_add]]

                at_risk_mask = (coverage_count[covers_remove] == 1)
                at_risk = covers_remove[at_risk_mask]

                if len(at_risk) == 0:
                    continue

                at_risk_set = set(at_risk.tolist())
                add_set = set(covers_add.tolist())
                if at_risk_set.issubset(add_set):
                    remove_codeword(ci_remove, active, coverage_count)
                    add_codeword(ci_add, active, coverage_count)
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    if len(inactive_list) == 0:
                        break

        pop_count = int(np.sum(active))
        fitness = compute_fitness(coverage_count)
        population.append((pop_count, fitness, active.copy(), coverage_count.copy()))

    # Sort population by count (primary), then fitness (secondary)
    population.sort(key=lambda x: (x[0], x[1]))

    # Track best
    if len(population) > 0:
        best_count = min(best_count, population[0][0])
        best_indices = np.where(population[0][2])[0]
        best_code = [list(code[i]) for i in best_indices]

    # Genetic algorithm loop
    for gen in range(generations):
        if time.time() > deadline - 15.0:
            break

        # Selection: keep elites
        elite_count = max(1, int(population_size * elite_fraction))
        elites = population[:elite_count]
        parents = population[:population_size]  # Use all for selection

        new_population = list(elites)

        # Generate offspring
        while len(new_population) < population_size:
            if time.time() > deadline - 15.0:
                break

            # Tournament selection
            def tournament_select():
                candidates = random.sample(parents, min(3, len(parents)))
                return min(candidates, key=lambda x: (x[0], x[1]))

            parent1 = tournament_select()
            parent2 = tournament_select()

            if random.random() < crossover_rate and len(parent1) == len(parent2):
                # Crossover: combine active sets
                # For each codeword, decide which parent's state to take
                # Bias towards the better parent
                child_active = np.zeros(num_codewords, dtype=bool)
                for ci in range(num_codewords):
                    if random.random() < 0.5:
                        child_active[ci] = parent1[2][ci]
                    else:
                        child_active[ci] = parent2[2][ci]

                # Recompute coverage for child
                child_coverage = np.zeros(total_words, dtype=np.int32)
                for ci in range(num_codewords):
                    if child_active[ci]:
                        covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
                        child_coverage[covers] += 1

                # Check validity: all words must be covered
                if np.all(child_coverage > 0):
                    # Mutation: try to remove some redundant codewords
                    if random.random() < mutation_rate:
                        order = list(range(num_codewords))
                        random.shuffle(order)
                        greedy_remove_pass(child_active, child_coverage, order)

                    child_count = int(np.sum(child_active))
                    child_fitness = compute_fitness(child_coverage)
                    new_population.append((child_count, child_fitness, child_active.copy(), child_coverage.copy()))
                else:
                    # Invalid: fall back to parent1
                    new_population.append((parent1[0], parent1[1], parent1[2].copy(), parent1[3].copy()))
            else:
                # No crossover: copy parent with possible mutation
                child_active = parent1[2].copy()
                child_coverage = parent1[3].copy()

                if random.random() < mutation_rate:
                    # Try to remove a redundant codeword
                    active_list = np.where(child_active)[0]
                    if len(active_list) > 0:
                        ci = int(random.choice(active_list))
                        if is_redundant(ci, child_coverage):
                            remove_codeword(ci, child_active, child_coverage)

                    # Try to swap
                    active_list = np.where(child_active)[0]
                    inactive_list = np.where(~child_active)[0]
                    if len(active_list) > 0 and len(inactive_list) > 0:
                        ci_remove = int(random.choice(active_list))
                        ci_add = int(random.choice(inactive_list))
                        covers_remove = codeword_covers_arr[ci_remove, :codeword_covers_len[ci_remove]]
                        covers_add = codeword_covers_arr[ci_add, :codeword_covers_len[ci_add]]
                        at_risk_mask = (child_coverage[covers_remove] == 1)
                        at_risk = covers_remove[at_risk_mask]
                        if len(at_risk) == 0:
                            remove_codeword(ci_remove, child_active, child_coverage)
                        else:
                            at_risk_set = set(at_risk.tolist())
                            add_set = set(covers_add.tolist())
                            if at_risk_set.issubset(add_set):
                                remove_codeword(ci_remove, child_active, child_coverage)
                                add_codeword(ci_add, child_active, child_coverage)

                child_count = int(np.sum(child_active))
                child_fitness = compute_fitness(child_coverage)
                new_population.append((child_count, child_fitness, child_active.copy(), child_coverage.copy()))

        population = new_population
        population.sort(key=lambda x: (x[0], x[1]))

        # Update best
        if population[0][0] < best_count:
            best_count = population[0][0]
            best_indices = np.where(population[0][2])[0]
            best_code = [list(code[i]) for i in best_indices]

        # Early exit if we reached the known optimum
        if best_count <= 73:
            break

    # Final local search on the best individual
    if best_code is not None and time.time() < deadline - 5.0:
        active = np.zeros(num_codewords, dtype=bool)
        for i in best_code:
            active[i] = True
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            if active[ci]:
                covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
                coverage_count[covers] += 1

        # Final greedy removal
        order = list(range(num_codewords))
        random.shuffle(order)
        greedy_remove_pass(active, coverage_count, order)

        # Final swap search
        active_list = np.where(active)[0]
        inactive_list = np.where(~active)[0]
        if len(inactive_list) > 0 and len(active_list) > 0:
            no_improve = 0
            for _ in range(1000):
                if time.time() > deadline - 3.0:
                    break
                if no_improve > 100:
                    break
                ci_remove = int(random.choice(active_list))
                ci_add = int(random.choice(inactive_list))

                covers_remove = codeword_covers_arr[ci_remove, :codeword_covers_len[ci_remove]]
                covers_add = codeword_covers_arr[ci_add, :codeword_covers_len[ci_add]]

                at_risk_mask = (coverage_count[covers_remove] == 1)
                at_risk = covers_remove[at_risk_mask]

                if len(at_risk) == 0:
                    remove_codeword(ci_remove, active, coverage_count)
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    no_improve = 0
                    if len(inactive_list) == 0:
                        break
                    continue

                at_risk_set = set(at_risk.tolist())
                add_set = set(covers_add.tolist())
                if at_risk_set.issubset(add_set):
                    remove_codeword(ci_remove, active, coverage_count)
                    add_codeword(ci_add, active, coverage_count)
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    no_improve = 0
                    if len(inactive_list) == 0:
                        break
                else:
                    no_improve += 1

        final_count = int(np.sum(active))
        if final_count < best_count:
            best_count = final_count
            best_indices = np.where(active)[0]
            best_code = [list(code[i]) for i in best_indices]

    if best_code is None:
        best_code = [list(c) for c in code]

    # Final verification
    if time.time() < deadline - 2.0 and len(best_code) <= 100:
        covered_set = set()
        for c in best_code:
            c_arr = np.array(c, dtype=np.int32)
            idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            covered_set.add(idx)
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_idx = idx + (val - orig) * (q ** (n - 1 - pos))
                    covered_set.add(new_idx)
        if len(covered_set) < total_words:
            best_code = [list(c) for c in code]

    return best_code
