import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit

    if q != 3 or n != 6 or radius != 1:
        # Fallback: return all words (valid but large)
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]

    # Precompute neighbors as bitsets
    # cover_mask[i] is an integer with bits set for all j such that dist(i, j) <= 1
    cover_mask = [0] * total
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        mask = 1 << i
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    mask |= (1 << new_i)
        cover_mask[i] = mask

    full_mask = (1 << total) - 1

    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    def greedy_set_cover():
        """Greedy set cover using bitsets."""
        uncovered = full_mask
        codewords = []

        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total):
                count = (cover_mask[c] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c

            codewords.append(best_c)
            uncovered &= ~cover_mask[best_c]

        return codewords

    def compute_coverage_mask(codewords):
        """Compute the union of all cover masks."""
        mask = 0
        for c in codewords:
            mask |= cover_mask[c]
        return mask

    def is_valid(codewords):
        """Check if codewords cover all words."""
        mask = 0
        for c in codewords:
            mask |= cover_mask[c]
        return mask == full_mask

    def fitness(codewords):
        """
        Fitness function: sum of squared coverage counts.
        Lower is better - indicates more balanced coverage (less redundancy).
        """
        coverage = np.zeros(total, dtype=np.int32)
        for c in codewords:
            # Extract bits from cover_mask[c] and increment coverage
            mask = cover_mask[c]
            while mask:
                bit = (mask & -mask).bit_length() - 1
                coverage[bit] += 1
                mask &= mask - 1
        return np.sum(coverage * coverage)

    def local_search_polish(codewords, time_budget):
        """Perform local search to reduce code size using bitsets."""
        S = set(codewords)
        S_mask = 0
        for c in S:
            S_mask |= (1 << c)

        best_S = set(S)
        best_size = len(S)

        def try_remove():
            """Try to find a removable word."""
            nonlocal S, S_mask, best_S, best_size
            items = list(S)
            random.shuffle(items)

            for w in items:
                union_others = 0
                for v in S:
                    if v != w:
                        union_others |= cover_mask[v]

                L = cover_mask[w] & ~union_others

                if L == 0:
                    S.remove(w)
                    S_mask &= ~(1 << w)
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = set(S)
                    return True
            return False

        def try_swap():
            """Try to swap a word with another that covers its unique words."""
            nonlocal S, S_mask, best_S, best_size
            items = list(S)
            random.shuffle(items)

            for w in items:
                union_others = 0
                for v in S:
                    if v != w:
                        union_others |= cover_mask[v]

                L = cover_mask[w] & ~union_others

                if L == 0:
                    continue

                candidates = full_mask
                temp_L = L
                while temp_L:
                    x = (temp_L & -temp_L).bit_length() - 1
                    temp_L &= temp_L - 1
                    candidates &= cover_mask[x]
                    if candidates == 0:
                        break

                if candidates == 0:
                    continue

                candidates &= ~S_mask
                candidates &= ~(1 << w)

                if candidates:
                    bit_count = candidates.bit_count()
                    if bit_count > 0:
                        r = random.getrandbits(64)
                        masked = candidates & r
                        if masked:
                            w_new = (masked & -masked).bit_length() - 1
                        else:
                            w_new = (candidates & -candidates).bit_length() - 1

                        S.remove(w)
                        S.add(w_new)
                        S_mask = 0
                        for c in S:
                            S_mask |= (1 << c)
                        return True

            return False

        def try_two_swap():
            """Try to remove two words and add one that covers both their unique words."""
            nonlocal S, S_mask, best_S, best_size
            if len(S) < 3:
                return False

            items = list(S)
            random.shuffle(items)

            for i in range(min(len(items), 20)):
                for j in range(i + 1, min(len(items), 20)):
                    w1, w2 = items[i], items[j]

                    union_others = 0
                    for v in S:
                        if v != w1 and v != w2:
                            union_others |= cover_mask[v]

                    L1 = cover_mask[w1] & ~union_others
                    L2 = cover_mask[w2] & ~union_others
                    L = L1 | L2

                    if L == 0:
                        S.remove(w1)
                        S.remove(w2)
                        S_mask &= ~(1 << w1)
                        S_mask &= ~(1 << w2)
                        if len(S) < best_size:
                            best_size = len(S)
                            best_S = set(S)
                        return True

                    candidates = full_mask
                    temp_L = L
                    while temp_L:
                        x = (temp_L & -temp_L).bit_length() - 1
                        temp_L &= temp_L - 1
                        candidates &= cover_mask[x]
                        if candidates == 0:
                            break

                    if candidates == 0:
                        continue

                    candidates &= ~S_mask
                    candidates &= ~(1 << w1)
                    candidates &= ~(1 << w2)

                    if candidates:
                        r = random.getrandbits(64)
                        masked = candidates & r
                        if masked:
                            w_new = (masked & -masked).bit_length() - 1
                        else:
                            w_new = (candidates & -candidates).bit_length() - 1

                        S.remove(w1)
                        S.remove(w2)
                        S.add(w_new)
                        S_mask = 0
                        for c in S:
                            S_mask |= (1 << c)
                        if len(S) < best_size:
                            best_size = len(S)
                            best_S = set(S)
                        return True

            return False

        stuck_count = 0
        max_stuck = 50
        last_improvement = time.time()

        while time.time() < time_budget:
            r = random.random()
            if r < 0.4:
                success = try_remove()
            elif r < 0.8:
                success = try_swap()
            else:
                success = try_two_swap()

            if success and len(S) <= best_size:
                stuck_count = 0
                last_improvement = time.time()
                max_stuck = random.randint(30, 100)
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    stuck_count = 0
                    max_stuck = random.randint(50, 200)
                    if time.time() - last_improvement > 5.0:
                        if len(S) > 2:
                            items = list(S)
                            random.shuffle(items)
                            for _ in range(random.randint(1, 3)):
                                if len(S) > 2:
                                    w_out = random.choice(list(S))
                                    S.remove(w_out)
                                    S_mask &= ~(1 << w_out)
                                    candidates = full_mask & ~S_mask
                                    if candidates:
                                        r2 = random.getrandbits(64)
                                        masked = candidates & r2
                                        if masked:
                                            w_in = (masked & -masked).bit_length() - 1
                                        else:
                                            w_in = (candidates & -candidates).bit_length() - 1
                                        S.add(w_in)
                                        S_mask |= (1 << w_in)
                            last_improvement = time.time()
                    if max_stuck > 500:
                        break

        return list(best_S)

    def genetic_algorithm(codewords, time_budget, pop_size=10):
        """
        Genetic algorithm to optimize the code.
        Fitness: sum of squared coverage counts (lower is better).
        Crossover: preserves Hamming ball structure.
        """
        population = [list(codewords)]
        for _ in range(pop_size - 1):
            # Create variants by random swaps
            variant = list(codewords)
            for _ in range(random.randint(1, 3)):
                if len(variant) > 0:
                    idx = random.randint(0, len(variant) - 1)
                    old = variant[idx]
                    # Find replacement
                    S_mask = 0
                    for c in variant:
                        S_mask |= (1 << c)
                    candidates = full_mask & ~S_mask
                    if candidates:
                        r = random.getrandbits(64)
                        masked = candidates & r
                        if masked:
                            new = (masked & -masked).bit_length() - 1
                        else:
                            new = (candidates & -candidates).bit_length() - 1
                        variant[idx] = new
            if is_valid(variant):
                population.append(variant)
            else:
                population.append(list(codewords))

        best_individual = list(min(population, key=lambda x: (len(x), fitness(x))))
        best_fitness = (len(best_individual), fitness(best_individual))

        generation = 0
        while time.time() < time_budget:
            generation += 1

            # Evaluate fitness
            fitnesses = [(len(ind), fitness(ind)) for ind in population]

            # Select parents (tournament selection)
            def tournament_select(k=3):
                candidates = random.sample(range(len(population)), min(k, len(population)))
                best_idx = min(candidates, key=lambda i: fitnesses[i])
                return list(population[best_idx])

            new_population = []
            for _ in range(pop_size // 2):
                parent1 = tournament_select()
                parent2 = tournament_select()

                # Crossover: take half from each, then repair
                child = list(parent1[:len(parent1)//2]) + list(parent2[len(parent2)//2:])

                # Remove duplicates
                child = list(dict.fromkeys(child))

                # Repair: ensure validity
                if not is_valid(child):
                    uncovered = full_mask
                    for c in child:
                        uncovered &= ~cover_mask[c]
                    while uncovered:
                        best_c = -1
                        best_count = -1
                        for c in range(total):
                            if c not in child:
                                count = (cover_mask[c] & uncovered).bit_count()
                                if count > best_count:
                                    best_count = count
                                    best_c = c
                        if best_c == -1:
                            break
                        child.append(best_c)
                        uncovered &= ~cover_mask[best_c]

                # Mutation: random swap
                if random.random() < 0.3 and len(child) > 0:
                    idx = random.randint(0, len(child) - 1)
                    S_mask = 0
                    for c in child:
                        S_mask |= (1 << c)
                    candidates = full_mask & ~S_mask
                    if candidates:
                        r = random.getrandbits(64)
                        masked = candidates & r
                        if masked:
                            new = (masked & -masked).bit_length() - 1
                        else:
                            new = (candidates & -candidates).bit_length() - 1
                        child[idx] = new

                if is_valid(child):
                    new_population.append(child)

            # Elitism: keep best individual
            if population:
                best_in_pop = min(population, key=lambda x: (len(x), fitness(x)))
                new_population.append(best_in_pop)

            # Fill remaining slots with best from new_population or copies
            while len(new_population) < pop_size:
                if new_population:
                    new_population.append(list(random.choice(new_population)))
                else:
                    new_population.append(list(codewords))

            population = new_population

            # Track best
            current_best = min(population, key=lambda x: (len(x), fitness(x)))
            if (len(current_best), fitness(current_best)) < best_fitness:
                best_fitness = (len(current_best), fitness(current_best))
                best_individual = list(current_best)

        return best_individual

    # Multi-restart strategy
    best_overall = None
    best_overall_size = float('inf')

    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1

        elapsed = time.time() - start_time
        remaining = deadline - 5 - time.time()

        if best_overall_size <= 74:
            time_budget = min(time.time() + 5, deadline - 5)
        else:
            time_budget = min(time.time() + 12, deadline - 5)

        # Greedy initialization
        codewords = greedy_set_cover()

        # Genetic algorithm
        codewords = genetic_algorithm(codewords, min(time.time() + time_budget * 0.5, deadline - 5))

        # Local search polish
        codewords = local_search_polish(codewords, min(time.time() + time_budget * 0.5, deadline - 5))

        if len(codewords) < best_overall_size:
            best_overall_size = len(codewords)
            best_overall = list(codewords)

        if best_overall_size == 73 and time.time() < deadline - 10:
            codewords = local_search_polish(list(best_overall), deadline - 5)
            if len(codewords) < best_overall_size:
                best_overall_size = len(codewords)
                best_overall = list(codewords)

    if best_overall is None:
        best_overall = [0]

    return [idx_to_list(idx) for idx in best_overall]
