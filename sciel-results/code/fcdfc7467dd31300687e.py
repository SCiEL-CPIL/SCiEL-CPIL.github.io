"""Covering design C(24,7,3): GA with vectorized greedy and local search."""
import itertools
import time
import random
import numpy as np


def _nCr(n, r):
    if r > n or r < 0:
        return 0
    if r == 0:
        return 1
    res = 1
    for i in range(r):
        res *= (n - i)
        res //= (i + 1)
    return res


def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill

    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}

    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    t_count = _nCr(k, t)

    # Precompute block_t_indices: which t-sets each k-block covers
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = tset_index[sub]

    rng = random.Random(42)

    def _vectorized_greedy(seed, time_budget):
        """Greedy covering using vectorized gain computation."""
        deadline = time.time() + time_budget
        np_rng = np.random.RandomState(seed)
        covered = np.zeros(n_tsets, dtype=np.int8)
        block_indices = []
        remaining = n_tsets

        while remaining > 0 and time.time() < deadline:
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            # Pick from top-50 by gain for diversity
            top_n = min(50, n_ks)
            top_idx = np.argpartition(gains, -top_n)[-top_n:]
            best_idx = top_idx[np_rng.randint(0, len(top_idx))]
            block_indices.append(best_idx)
            subs = block_t_indices[best_idx]
            newly = np.sum(covered[subs] == 0)
            remaining -= newly
            covered[subs] = 1

        return block_indices

    def _remove_redundant(block_indices):
        """Remove redundant blocks using coverage counts."""
        if not block_indices:
            return block_indices
        coverage_count = np.zeros(n_tsets, dtype=np.int8)
        for bi in block_indices:
            coverage_count[block_t_indices[bi]] += 1
        result = []
        for bi in block_indices:
            subs = block_t_indices[bi]
            if np.min(coverage_count[subs]) > 1:
                coverage_count[subs] -= 1
            else:
                result.append(bi)
        return result

    def _fitness(block_indices):
        """Fitness: (uncovered_count, num_blocks). Lower is better."""
        if not block_indices:
            return (n_tsets, 0)
        covered = np.zeros(n_tsets, dtype=np.int8)
        for bi in block_indices:
            covered[block_t_indices[bi]] = 1
        n_uncov = np.sum(covered == 0)
        return (n_uncov, len(block_indices))

    def _crossover(p1_idx, p2_idx, rng, time_budget):
        """Crossover: mix blocks from parents, fill gaps greedily."""
        deadline = time.time() + time_budget
        child = []
        used = set()
        p1_shuf = list(p1_idx)
        p2_shuf = list(p2_idx)
        rng.shuffle(p1_shuf)
        rng.shuffle(p2_shuf)
        n_from_p1 = rng.randint(len(p1_shuf) // 3, max(1, 2 * len(p1_shuf) // 3))
        n_from_p2 = max(0, len(p1_idx) - n_from_p1)

        for bi in p1_shuf[:n_from_p1]:
            if bi not in used:
                child.append(bi)
                used.add(bi)
        for bi in p2_shuf[:n_from_p2]:
            if bi not in used:
                child.append(bi)
                used.add(bi)

        # Fill gaps with vectorized greedy
        covered = np.zeros(n_tsets, dtype=np.int8)
        for bi in child:
            covered[block_t_indices[bi]] = 1
        remaining = np.sum(covered == 0)

        while remaining > 0 and time.time() < deadline:
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            top_n = min(30, n_ks)
            top_idx = np.argpartition(gains, -top_n)[-top_n:]
            best_idx = top_idx[rng.randint(0, len(top_idx) - 1)]
            if best_idx not in used:
                child.append(best_idx)
                used.add(best_idx)
                subs = block_t_indices[best_idx]
                newly = np.sum(covered[subs] == 0)
                remaining -= newly
                covered[subs] = 1
            else:
                # Already have this block, pick next best
                gains[best_idx] = -1
                top_idx = np.argpartition(gains, -top_n)[-top_n:]
                best_idx = top_idx[rng.randint(0, len(top_idx) - 1)]
                if best_idx not in used:
                    child.append(best_idx)
                    used.add(best_idx)
                    subs = block_t_indices[best_idx]
                    newly = np.sum(covered[subs] == 0)
                    remaining -= newly
                    covered[subs] = 1

        return child

    def _local_search(block_indices, time_budget):
        """Local search: remove redundant blocks iteratively."""
        deadline = time.time() + time_budget
        current = list(block_indices)

        while time.time() < deadline:
            improved = False
            current = _remove_redundant(current)
            # Try single-block replacement
            n = len(current)
            for i in range(n):
                if time.time() >= deadline:
                    break
                trial = current[:i] + current[i + 1:]
                covered = np.zeros(n_tsets, dtype=np.int8)
                for bi in trial:
                    covered[block_t_indices[bi]] = 1
                if np.all(covered >= 1):
                    current = trial
                    improved = True
                    break
            if not improved:
                break

        return current

    # Phase 1: Generate initial population
    population = []
    pop_size = 8
    greedy_time = 3.0

    for seed in range(pop_size):
        if time.time() - start_time > time_limit * 0.25:
            break
        blocks = _vectorized_greedy(seed * 1000 + 42, greedy_time)
        blocks = _remove_redundant(blocks)
        if blocks:
            population.append(blocks)

    if not population:
        blocks = _vectorized_greedy(0, time_limit * 0.3)
        blocks = _remove_redundant(blocks)
        population = [blocks]

    # Phase 2: Genetic algorithm
    ga_deadline = start_time + time_limit * 0.85
    best_fitness = min(_fitness(p) for p in population)
    best_blocks = min(population, key=lambda p: _fitness(p))

    gen = 0
    while time.time() < ga_deadline:
        gen += 1
        new_population = []

        sorted_pop = sorted(population, key=lambda p: _fitness(p))
        new_population.append(sorted_pop[0])

        while len(new_population) < pop_size and time.time() < ga_deadline:
            p1 = sorted_pop[rng.randint(0, min(3, len(sorted_pop) - 1))]
            p2 = sorted_pop[rng.randint(0, min(3, len(sorted_pop) - 1))]
            child = _crossover(p1, p2, rng, time_budget=2.0)
            child = _remove_redundant(child)
            new_population.append(child)

        for i in range(len(new_population)):
            if time.time() >= ga_deadline:
                break
            fitness = _fitness(new_population[i])
            if fitness < best_fitness:
                best_fitness = fitness
                best_blocks = new_population[i]

        population = new_population
        sorted_pop = sorted(population, key=lambda p: _fitness(p))

    # Phase 3: Final local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 3:
        best_blocks = _local_search(best_blocks, remaining_time * 0.9)
        best_blocks = _remove_redundant(best_blocks)

    # Convert to list of lists
    return [list(all_ks[i]) for i in best_blocks]
