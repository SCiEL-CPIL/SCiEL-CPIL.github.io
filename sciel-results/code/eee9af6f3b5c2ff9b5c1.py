<<<<<<< SEARCH
    def genetic_improve(selected_indices, pop_size=4, generations=10):
        """Genetic algorithm to improve the covering."""
        if not selected_indices or len(selected_indices) < 2:
            return selected_indices

        def fitness(sol):
            return len(sol)

        def crossover(p1, p2):
            """Uniform crossover with repair."""
            if len(p1) <= len(p2):
                shorter, longer = p1, p2
            else:
                shorter, longer = p2, p1

            # Take random subset from shorter, fill with random from longer
            new_sol = set()
            for b in shorter:
                if random.random() < 0.5:
                    new_sol.add(b)
            for b in longer:
                if random.random() < 0.3:
                    new_sol.add(b)
            return list(new_sol)

        def mutate(sol):
            """Swap a block with a random other block."""
            if not sol:
                return sol
            i = random.randint(0, len(sol) - 1)
            g = sol[i]
            # Find a random replacement
            for _ in range(20):
                g_new = random.randint(0, N_blocks - 1)
                if g_new != g:
                    sol[i] = g_new
                    return sol
            return sol

        def is_valid_cover(sol):
            """Check if sol covers all t-subsets."""
            covered = np.zeros(N_t, dtype=np.int8)
            for g in sol:
                for si in block_tsubs[g]:
                    covered[si] = 1
            return np.all(covered == 1)

        population = [list(selected_indices)]
        # Add some variants
        for _ in range(pop_size - 1):
            variant = list(selected_indices)
            if random.random() < 0.5 and variant:
                variant = mutate(variant)
            population.append(variant)

        best = min(population, key=fitness)

        for gen in range(generations):
            if time.time() > DEADLINE:
                break

            next_pop = [list(best)]  # Elitism

            while len(next_pop) < pop_size:
                # Tournament selection
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                p1 = population[p1_idx]
                p2 = population[p2_idx]

                child = crossover(p1, p2)
                if random.random() < 0.3:
                    child = mutate(child)

                # Reduce
                child = reduce_solution_fast(child)

                # Check validity
                if is_valid_cover(child):
                    next_pop.append(child)
                else:
                    # Repair: add blocks from best to cover gaps
                    covered = np.zeros(N_t, dtype=np.int8)
                    for g in child:
                        for si in block_tsubs[g]:
                            covered[si] = 1
                    for g in best:
                        if g not in child:
                            for si in block_tsubs[g]:
                                if covered[si] == 0:
                                    child.append(g)
                                    for si2 in block_tsubs[g]:
                                        covered[si2] = 1
                                    break
                    child = reduce_solution_fast(child)
                    if is_valid_cover(child):
                        next_pop.append(child)

            population = next_pop
            new_best = min(population, key=fitness)
            if fitness(new_best) < fitness(best):
                best = new_best

        return best
=======
    def local_search_swap(selected_indices):
        """Try to swap pairs of blocks for a single block that covers their union."""
        if not selected_indices or len(selected_indices) < 2:
            return selected_indices

        n_sel = len(selected_indices)
        in_solution = set(selected_indices)
        total_count = np.zeros(N_t, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            n_sel = len(selected_indices)
            in_solution = set(selected_indices)

            # Rebuild total_count
            total_count = np.zeros(N_t, dtype=np.int32)
            for g in selected_indices:
                for si in block_tsubs[g]:
                    total_count[si] += 1

            # Try all pairs
            for i in range(n_sel):
                if time.time() > DEADLINE:
                    return selected_indices
                for j in range(i + 1, n_sel):
                    if time.time() > DEADLINE:
                        return selected_indices

                    g1 = selected_indices[i]
                    g2 = selected_indices[j]

                    # Find t-subsets that would be uncovered if we remove both
                    uncovered = []
                    for si in block_tsubs[g1]:
                        if total_count[si] <= 1:
                            uncovered.append(si)
                    for si in block_tsubs[g2]:
                        if total_count[si] <= 1 and si not in uncovered:
                            # Check if it's covered by g1
                            if total_count[si] == 0 or (total_count[si] == 1 and si not in block_tsubs[g1]):
                                uncovered.append(si)

                    if not uncovered:
                        # Can remove both without replacement
                        # Remove j first (higher index)
                        selected_indices.pop(j)
                        selected_indices.pop(i)
                        improved = True
                        break

                    # Try to find a single block that covers all uncovered
                    found = False
                    for g_new in all_blocks:
                        if g_new in in_solution:
                            continue
                        covers_all = True
                        for si in uncovered:
                            if si not in block_tsubs[g_new]:
                                covers_all = False
                                break
                        if covers_all:
                            found = True
                            # Replace both with g_new
                            selected_indices[j] = g_new
                            selected_indices.pop(i)
                            in_solution = set(selected_indices)
                            improved = True
                            break

                    if found:
                        break

        return selected_indices
>>>>>>> REPLACE

