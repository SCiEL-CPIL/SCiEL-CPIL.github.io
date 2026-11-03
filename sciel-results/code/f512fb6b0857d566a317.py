"""Genetic algorithm for C(19,8,5) covering design with greedy repair."""
import time
import itertools
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 118.0  # safety margin before 120s kill
    random.seed(42)

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_to_idx = {tuple(c): i for i, c in enumerate(all_t)}

    # Generate all possible k-subsets
    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)

    # For each block, precompute which t-subsets it covers
    block_t_coverage = []
    blocks_covering_t = defaultdict(list)
    for b_idx, blk in enumerate(all_k):
        covered_t = []
        for S in itertools.combinations(blk, t):
            idx = t_to_idx.get(tuple(S))
            if idx is not None:
                covered_t.append(idx)
                blocks_covering_t[idx].append(b_idx)
        block_t_coverage.append(covered_t)

    def greedy_cover(start_covered=None, start_blocks=None):
        """Run greedy set cover, optionally starting from a partial solution."""
        if start_covered is not None:
            covered = start_covered.copy()
            remaining = n_t - int(covered.sum())
            used = np.zeros(n_k, dtype=bool)
            for b_idx in start_blocks:
                used[b_idx] = True
            blocks = list(start_blocks)
        else:
            covered = np.zeros(n_t, dtype=bool)
            remaining = n_t
            used = np.zeros(n_k, dtype=bool)
            blocks = []

        gain = np.array([len(ct) for ct in block_t_coverage], dtype=np.int32)
        # Initialize gains considering already-covered t-subsets
        if start_covered is not None:
            for b_idx in range(n_k):
                gain[b_idx] = sum(1 for t_idx in block_t_coverage[b_idx] if not covered[t_idx])

        while remaining > 0:
            if time.time() > deadline:
                break

            masked_gain = gain.copy()
            masked_gain[used] = -1
            b_star = int(np.argmax(masked_gain))

            if masked_gain[b_star] <= 0:
                break

            used[b_star] = True
            blocks.append(b_star)

            for t_idx in block_t_coverage[b_star]:
                if not covered[t_idx]:
                    covered[t_idx] = True
                    remaining -= 1
                    for b in blocks_covering_t[t_idx]:
                        if not used[b]:
                            gain[b] -= 1

        return blocks, covered, remaining

    # Initial population: multiple greedy solutions with different random seeds
    population = []
    pop_size = 5

    for i in range(pop_size):
        if time.time() > deadline:
            break
        blocks, covered, remaining = greedy_cover()
        population.append((len(blocks), blocks, covered))

    # Genetic algorithm loop
    best_size = min(pop[0] for pop in population)
    best_blocks = min(pop, key=lambda x: x[0])[1]

    generation = 0
    while time.time() < deadline - 5.0:  # keep 5s buffer
        generation += 1

        # Selection: keep best half, generate new offspring
        population.sort(key=lambda x: x[0])
        elite = population[:max(1, pop_size // 2)]
        new_population = list(elite)

        # Crossover: combine two parents
        while len(new_population) < pop_size and time.time() < deadline - 5.0:
            if len(elite) < 2:
                break

            # Tournament selection
            def select():
                candidates = random.sample(population, min(3, len(population)))
                return min(candidates, key=lambda x: x[0])

            parent1 = select()
            parent2 = select()

            # Crossover: randomly pick blocks from each parent
            child_blocks = []
            used_in_child = set()

            # Start with some blocks from parent1
            for b_idx in random.sample(parent1[1], min(len(parent1[1]), len(parent1[1]) // 2)):
                if b_idx not in used_in_child:
                    child_blocks.append(b_idx)
                    used_in_child.add(b_idx)

            # Add blocks from parent2 that aren't already in child
            for b_idx in parent2[1]:
                if b_idx not in used_in_child and len(child_blocks) < best_size:
                    child_blocks.append(b_idx)
                    used_in_child.add(b_idx)

            # Repair: check coverage and add missing blocks via greedy
            covered = np.zeros(n_t, dtype=bool)
            for b_idx in child_blocks:
                for t_idx in block_t_coverage[b_idx]:
                    covered[t_idx] = True

            remaining = n_t - int(covered.sum())
            if remaining > 0:
                # Greedily add blocks to cover remaining t-subsets
                gain = np.array([0] * n_k, dtype=np.int32)
                for b_idx in range(n_k):
                    if b_idx not in used_in_child:
                        gain[b_idx] = sum(1 for t_idx in block_t_coverage[b_idx] if not covered[t_idx])

                while remaining > 0 and time.time() < deadline - 5.0:
                    masked_gain = gain.copy()
                    masked_gain[list(used_in_child)] = -1
                    b_star = int(np.argmax(masked_gain))

                    if masked_gain[b_star] <= 0:
                        break

                    used_in_child.add(b_star)
                    child_blocks.append(b_star)

                    for t_idx in block_t_coverage[b_star]:
                        if not covered[t_idx]:
                            covered[t_idx] = True
                            remaining -= 1
                            for b in blocks_covering_t[t_idx]:
                                if b not in used_in_child:
                                    gain[b] -= 1

            child_size = len(child_blocks)
            if child_size <= best_size:
                new_population.append((child_size, child_blocks, covered))

            # Mutation: remove random block and repair
            if time.time() < deadline - 5.0:
                parent = select()
                mutated_blocks = list(parent[1])

                if len(mutated_blocks) > 1:
                    # Remove a random block
                    remove_idx = random.randint(0, len(mutated_blocks) - 1)
                    removed_block = mutated_blocks.pop(remove_idx)
                    used_in_mutated = set(mutated_blocks)

                    # Recompute coverage
                    covered = np.zeros(n_t, dtype=bool)
                    for b_idx in mutated_blocks:
                        for t_idx in block_t_coverage[b_idx]:
                            covered[t_idx] = True

                    remaining = n_t - int(covered.sum())
                    if remaining > 0:
                        # Greedily repair
                        gain = np.array([0] * n_k, dtype=np.int32)
                        for b_idx in range(n_k):
                            if b_idx not in used_in_mutated:
                                gain[b_idx] = sum(1 for t_idx in block_t_coverage[b_idx] if not covered[t_idx])

                        while remaining > 0 and time.time() < deadline - 5.0:
                            masked_gain = gain.copy()
                            masked_gain[list(used_in_mutated)] = -1
                            b_star = int(np.argmax(masked_gain))

                            if masked_gain[b_star] <= 0:
                                break

                            used_in_mutated.add(b_star)
                            mutated_blocks.append(b_star)

                            for t_idx in block_t_coverage[b_star]:
                                if not covered[t_idx]:
                                    covered[t_idx] = True
                                    remaining -= 1
                                    for b in blocks_covering_t[t_idx]:
                                        if b not in used_in_mutated:
                                            gain[b] -= 1

                    if len(mutated_blocks) <= best_size:
                        new_population.append((len(mutated_blocks), mutated_blocks, covered))

        # Update population
        population = new_population[:pop_size]

        # Track best
        current_best = min(population, key=lambda x: x[0])
        if current_best[0] < best_size:
            best_size = current_best[0]
            best_blocks = current_best[1]

    # Convert block indices to actual blocks
    return [list(all_k[b_idx]) for b_idx in best_blocks]
