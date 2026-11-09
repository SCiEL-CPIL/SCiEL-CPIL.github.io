"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 80  # safety margin within 90s wall-clock
    n = 128

    def check_validity(colors):
        """Check if coloring satisfies the constraint."""
        for a in range(1, n - 2):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                c4 = colors[a + 3 * d - 1]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True

    def count_violations(colors):
        """Count number of violating APs."""
        count = 0
        for a in range(1, n - 2):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                c4 = colors[a + 3 * d - 1]
                if len({c1, c2, c3, c4}) < 3:
                    count += 1
        return count

    def greedy_simple(num_colors):
        """Simple greedy: assign smallest valid color to each position."""
        colors = [0] * n
        colors[0] = 0
        for x in range(2, n + 1):
            valid_colors = set(range(num_colors))
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                existing = {c1, c2, c3}
                if len(existing) >= 3:
                    pass
                elif len(existing) == 2:
                    for c in list(valid_colors):
                        if c in existing:
                            valid_colors.discard(c)
                else:
                    for c in list(valid_colors):
                        if c == list(existing)[0]:
                            valid_colors.discard(c)
            if not valid_colors:
                return None
            colors[x - 1] = min(valid_colors)
        if check_validity(colors):
            return colors
        return None

    def greedy_lookahead(num_colors, lookahead_k=4):
        """Greedy with look-ahead to avoid dead-ends."""
        colors = [0] * n
        colors[0] = 0

        def simulate_next_k(colors, current_x, k):
            for step in range(1, k + 1):
                x = current_x + step
                if x > n:
                    return True
                valid = set(range(num_colors))
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a < 1:
                        break
                    c1 = colors[a - 1]
                    c2 = colors[a + d - 1]
                    c3 = colors[a + 2 * d - 1]
                    existing = {c1, c2, c3}
                    if len(existing) >= 3:
                        pass
                    elif len(existing) == 2:
                        for c in list(valid):
                            if c in existing:
                                valid.discard(c)
                    else:
                        for c in list(valid):
                            if c == list(existing)[0]:
                                valid.discard(c)
                if not valid:
                    return False
            return True

        for x in range(2, n + 1):
            valid_colors = set(range(num_colors))
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                existing = {c1, c2, c3}
                if len(existing) >= 3:
                    pass
                elif len(existing) == 2:
                    for c in list(valid_colors):
                        if c in existing:
                            valid_colors.discard(c)
                else:
                    for c in list(valid_colors):
                        if c == list(existing)[0]:
                            valid_colors.discard(c)

            if not valid_colors:
                return None

            best_color = None
            for c in sorted(valid_colors):
                colors[x - 1] = c
                if simulate_next_k(colors, x, lookahead_k):
                    best_color = c
                    break

            if best_color is None:
                colors[x - 1] = min(valid_colors)

        if check_validity(colors):
            return colors
        return None

    def genetic_algorithm(num_colors, pop_size=100, generations=50, mutation_rate=0.1):
        """Genetic algorithm with AP-aware crossover."""
        # Precompute all APs for fast violation checking
        all_aps = []
        for a in range(1, n - 2):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                all_aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

        def fitness(colors):
            """Lower is better: count violations."""
            violations = 0
            for ap in all_aps:
                c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if len({c1, c2, c3, c4}) < 3:
                    violations += 1
            return violations

        def crossover(parent1, parent2):
            """AP-aware crossover: swap segments at positions not forming APs together."""
            child = list(parent1)
            # Find a random split point
            split = random.randint(1, n - 1)
            child[split:] = parent2[split:]
            return child

        def mutate(colors, rate):
            """Random mutation: change some positions to random valid colors."""
            result = list(colors)
            for i in range(n):
                if random.random() < rate:
                    result[i] = random.randint(0, num_colors - 1)
            return result

        # Initialize population with greedy solutions and random
        population = []
        for _ in range(pop_size):
            if random.random() < 0.5:
                greedy_result = greedy_simple(num_colors)
                if greedy_result is not None:
                    population.append(list(greedy_result))
                else:
                    population.append([random.randint(0, num_colors - 1) for _ in range(n)])
            else:
                population.append([random.randint(0, num_colors - 1) for _ in range(n)])

        best_fitness = min(fitness(ind) for ind in population)
        best_individual = max(population, key=lambda x: -fitness(x))

        for gen in range(generations):
            if time.time() > deadline - 5:
                break

            # Evaluate fitness
            fitnesses = [fitness(ind) for ind in population]

            # Update best
            gen_best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = list(population[gen_best_idx])

            if best_fitness == 0:
                break

            # Selection: tournament
            new_population = []
            for _ in range(pop_size):
                # Tournament of 3
                contestants = random.sample(range(pop_size), 3)
                winner = min(contestants, key=lambda i: fitnesses[i])
                new_population.append(list(population[winner]))

            # Crossover and mutation
            for i in range(0, pop_size, 2):
                if i + 1 < pop_size:
                    p1 = new_population[i]
                    p2 = new_population[i + 1]
                    c1 = crossover(p1, p2)
                    c2 = crossover(p2, p1)
                    new_population[i] = mutate(c1, mutation_rate)
                    new_population[i + 1] = mutate(c2, mutation_rate)

            population = new_population

        if check_validity(best_individual):
            return best_individual
        return None

    # Strategy 1: Try greedy with 3 colors
    if time.time() < deadline - 20:
        result = greedy_lookahead(3, lookahead_k=4)
        if result is not None:
            return result

    if time.time() < deadline - 15:
        result = greedy_simple(3)
        if result is not None:
            return result

    # Strategy 2: Genetic algorithm with 3 colors
    if time.time() < deadline - 10:
        random.seed(42)  # Deterministic for reproducibility
        result = genetic_algorithm(3, pop_size=80, generations=30, mutation_rate=0.05)
        if result is not None:
            return result

    # Strategy 3: Try greedy with 4 colors
    if time.time() < deadline - 5:
        result = greedy_lookahead(4, lookahead_k=4)
        if result is not None:
            return result

    if time.time() < deadline - 5:
        result = greedy_simple(4)
        if result is not None:
            return result

    # Strategy 4: Genetic algorithm with 4 colors
    if time.time() < deadline - 3:
        random.seed(42)
        result = genetic_algorithm(4, pop_size=60, generations=20, mutation_rate=0.05)
        if result is not None:
            return result

    # Fallback: original seed approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
