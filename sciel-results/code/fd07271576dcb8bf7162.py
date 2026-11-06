import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit

    def time_check():
        return time.time() - start_time

    def count_triangles(adj_masks, n):
        """Count triangles in the graph."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    common = adj_masks[u] & adj_masks[v]
                    count += common.bit_count()
        return count // 3

    def find_large_independent_set(adj_masks, n, target=10):
        """
        Try to find an independent set of size >= target.
        Returns the size of the largest found, or -1 if none of target size.
        Uses branch and bound with bitsets.
        """
        best = [0]
        found_target = [False]

        def dfs(candidates, size):
            if found_target[0]:
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                if size >= target:
                    found_target[0] = True
                return

            # Upper bound: can we even reach target?
            if size + candidates.bit_count() < target:
                return

            # Pick vertex with highest degree in candidate subgraph (to prune)
            # For speed, just pick lowest set bit
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1

            # Include v
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            dfs(new_candidates, size + 1)

            if found_target[0]:
                return

            # Exclude v
            dfs(candidates & ~lsb, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def fitness(adj_masks, n, target_alpha=10):
        """
        Fitness: lower is better.
        Penalty for triangles + penalty for independent sets >= target.
        """
        tri = count_triangles(adj_masks, n)
        if tri > 0:
            return 1000000 + tri  # Huge penalty for triangles

        # Check for independent set of size >= target
        alpha = find_large_independent_set(adj_masks, n, target_alpha)
        if alpha >= target_alpha:
            return 1000 + (alpha - target_alpha + 1) * 100  # Penalty for too-large alpha
        return alpha  # Lower alpha is better

    def is_valid(adj_masks, n):
        """Check if graph is triangle-free and has no independent set of size 10."""
        tri = count_triangles(adj_masks, n)
        if tri > 0:
            return False
        alpha = find_large_independent_set(adj_masks, n, 10)
        return alpha < 10

    def get_edges(adj_masks, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def make_random_triangle_free(n, seed):
        """Create a random triangle-free graph by greedy edge addition."""
        random.seed(seed)
        adj_masks = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if adj_masks[u] & adj_masks[v]:
                continue
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def make_circulant(n, offsets, seed):
        """Create a circulant graph with given offsets."""
        random.seed(seed)
        adj_masks = [0] * n
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if j != i:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    def mutate(adj_masks, n, mutation_rate=0.05, seed=None):
        """Mutate a graph by flipping some edges, ensuring triangle-free."""
        if seed is not None:
            random.seed(seed)
        new_masks = adj_masks[:]
        num_flips = max(1, int(n * n * mutation_rate / 2))
        for _ in range(num_flips):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            # Try to flip edge (u,v)
            has_edge = (adj_masks[u] >> v) & 1
            if has_edge:
                # Remove edge
                new_masks[u] &= ~(1 << v)
                new_masks[v] &= ~(1 << u)
            else:
                # Add edge only if it doesn't create a triangle
                if new_masks[u] & new_masks[v]:
                    continue
                new_masks[u] |= (1 << v)
                new_masks[v] |= (1 << u)
        return new_masks

    def crossover(adj_masks1, adj_masks2, n, seed=None):
        """Crossover: take edges from both parents, ensure triangle-free."""
        if seed is not None:
            random.seed(seed)
        adj_masks = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            in1 = (adj_masks1[u] >> v) & 1
            in2 = (adj_masks2[u] >> v) & 1
            if in1 or in2:
                if adj_masks[u] & adj_masks[v]:
                    continue
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
        return adj_masks

    # Initialize population with diverse starting points
    POP_SIZE = 20
    n_target = 39

    population = []
    # Add some random triangle-free graphs
    for i in range(POP_SIZE // 2):
        adj = make_random_triangle_free(n_target, seed=i * 12345)
        population.append(adj)

    # Add some circulant graphs
    for i in range(POP_SIZE // 4):
        # Try different offset sets
        offsets = [1, 2, 3, 4, 5]
        random.seed(i * 7777)
        random.shuffle(offsets)
        adj = make_circulant(n_target, offsets[:5], seed=i * 7777)
        population.append(adj)

    # Add more random
    for i in range(POP_SIZE - len(population)):
        adj = make_random_triangle_free(n_target, seed=(i + 100) * 9999)
        population.append(adj)

    # Evaluate initial population
    fitnesses = [fitness(adj, n_target) for adj in population]

    # Find best in initial population
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    if fitnesses[best_idx] < 1000:  # No triangles and alpha < 10
        return {'n': n_target, 'edges': get_edges(population[best_idx], n_target)}

    best_fitness = fitnesses[best_idx]
    best_adj = population[best_idx]

    # Genetic algorithm loop
    generation = 0
    no_improve_count = 0

    while time_check() < DEADLINE - 15:
        generation += 1

        # Sort by fitness
        indexed = sorted(range(len(population)), key=lambda i: fitnesses[i])
        population = [population[i] for i in indexed]
        fitnesses = [fitnesses[i] for i in indexed]

        # Check if any is valid
        if fitnesses[0] < 1000:
            return {'n': n_target, 'edges': get_edges(population[0], n_target)}

        # Elitism: keep top few
        elite_count = max(2, POP_SIZE // 5)
        new_population = population[:elite_count]
        new_fitnesses = fitnesses[:elite_count]

        # Generate offspring
        while len(new_population) < POP_SIZE:
            if time_check() > DEADLINE - 15:
                break

            # Selection: tournament
            r1 = random.randint(0, len(population) - 1)
            r2 = random.randint(0, len(population) - 1)
            if fitnesses[r1] <= fitnesses[r2]:
                parent1 = population[r1]
            else:
                parent1 = population[r2]

            r1 = random.randint(0, len(population) - 1)
            r2 = random.randint(0, len(population) - 1)
            if fitnesses[r1] <= fitnesses[r2]:
                parent2 = population[r1]
            else:
                parent2 = population[r2]

            # Crossover or mutation
            if random.random() < 0.5:
                child = crossover(parent1, parent2, n_target, seed=generation * 1000 + len(new_population))
            else:
                child = mutate(parent1, n_target, mutation_rate=0.03, seed=generation * 2000 + len(new_population))

            child_fit = fitness(child, n_target)
            new_population.append(child)
            new_fitnesses.append(child_fit)

            # Check if child is valid
            if child_fit < 1000:
                return {'n': n_target, 'edges': get_edges(child, n_target)}

        population = new_population
        fitnesses = new_fitnesses

        # Track improvement
        current_best = min(fitnesses)
        if current_best < best_fitness - 1:
            best_fitness = current_best
            best_adj = population[fitnesses.index(best_fitness)]
            no_improve_count = 0
        else:
            no_improve_count += 1

        # If stuck, try a different n
        if no_improve_count > 30:
            # Try n=38
            if n_target == 39:
                n_target = 38
                population = []
                for i in range(POP_SIZE):
                    adj = make_random_triangle_free(n_target, seed=i * 5555)
                    population.append(adj)
                fitnesses = [fitness(adj, n_target) for adj in population]
                if min(fitnesses) < 1000:
                    best_idx = fitnesses.index(min(fitnesses))
                    return {'n': n_target, 'edges': get_edges(population[best_idx], n_target)}
                best_fitness = min(fitnesses)
                best_adj = population[fitnesses.index(best_fitness)]
                no_improve_count = 0
            elif n_target == 38:
                # Give up on GA, fall back to seed search
                break

    # Final fallback: try seed search for n=39 and 38
    for n_try in [39, 38]:
        if time_check() > DEADLINE - 5:
            break
        for seed in range(5000):
            if time_check() > DEADLINE - 5:
                break
            adj = make_random_triangle_free(n_try, seed=seed)
            if is_valid(adj, n_try):
                return {'n': n_try, 'edges': get_edges(adj, n_try)}

    # Return best found
    if best_fitness < 1000:
        return {'n': n_target, 'edges': get_edges(best_adj, n_target)}

    # Absolute fallback: small known valid graph
    # Andrásfai graph And(3) on 9 vertices? No, too small.
    # Let's just return a small triangle-free graph with small alpha.
    # C5 (cycle of length 5) is triangle-free, alpha=2
    n = 5
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    return {'n': n, 'edges': edges}
