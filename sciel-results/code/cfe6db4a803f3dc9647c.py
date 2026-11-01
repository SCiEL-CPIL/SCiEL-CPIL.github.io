"""Genetic algorithm for covering design C(24,7,3)."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_triples(v, t):
    triples = list(itertools.combinations(range(v), t))
    triple_idx = {ts: i for i, ts in enumerate(triples)}
    return triples, triple_idx, len(triples)

def _block_to_triple_indices(block, triple_idx, t):
    """Return numpy array of triple indices covered by a block."""
    return np.array([triple_idx[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)

def _compute_coverage_counts(blocks, triple_idx, t, n_triples):
    """Vectorized coverage count computation."""
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            counts[triple_idx[ts]] += 1
    return counts

def _count_uncovered(counts):
    return int(np.sum(counts == 0))

def _greedy_init(v, k, t, triples, triple_idx, rng, time_limit=3.0):
    """Fast greedy initialization using bitset-based gain."""
    deadline = time.time() + time_limit
    n_triples = len(triples)
    uncovered = np.ones(n_triples, dtype=np.bool_)
    blocks = []
    
    # Precompute for each point, which triples it participates in
    # For speed, we'll use a simpler approach: random seed + greedy fill
    
    while np.any(uncovered) and time.time() < deadline:
        # Pick a random uncovered triple
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        ts_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
        ts = triples[ts_idx]
        block = list(ts)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    tset = tuple(sorted(S + (x,)))
                    idx = triple_idx[tset]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for ts2 in itertools.combinations(block, t):
            uncovered[triple_idx[ts2]] = False
    
    return [list(b) for b in blocks]

def _remove_redundant(blocks, triples, triple_idx, v, k, t, time_limit=5.0):
    """Remove blocks whose all triples are covered by other blocks."""
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    
    while time.time() < deadline:
        n = len(blocks)
        if n == 0:
            break
        
        counts = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[triple_idx[ts]] += 1
        
        redundant_idx = -1
        for i in range(n):
            b = blocks[i]
            all_covered = True
            for ts in itertools.combinations(b, t):
                idx = triple_idx[ts]
                if counts[idx] == 1:
                    all_covered = False
                    break
            if all_covered:
                redundant_idx = i
                break
        
        if redundant_idx == -1:
            break
        blocks.pop(redundant_idx)
    
    return blocks

def _is_valid_covering(blocks, v, k, t, triples, triple_idx):
    n_triples = len(triples)
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            counts[triple_idx[ts]] += 1
    return int(np.sum(counts == 0)) == 0

def _repair_covering(blocks, v, k, t, triples, triple_idx, rng, time_limit=5.0):
    """Add blocks to cover any uncovered triples."""
    deadline = time.time() + time_limit
    n_triples = len(triples)
    blocks = [list(b) for b in blocks]
    
    while time.time() < deadline:
        counts = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[triple_idx[ts]] += 1
        
        unc = np.where(counts == 0)[0]
        if len(unc) == 0:
            break
        
        # Greedily add blocks to cover uncovered triples
        ts_idx = unc[rng.randint(0, len(unc) - 1)]
        ts = triples[ts_idx]
        block = list(ts)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    tset = tuple(sorted(S + (x,)))
                    idx = triple_idx[tset]
                    if counts[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        block.sort()
        blocks.append(block)
        
        for ts2 in itertools.combinations(block, t):
            counts[triple_idx[ts2]] += 1
    
    return blocks

def _crossover(parent1, parent2, v, k, t, triple_idx, rng):
    """Uniform crossover: for each position, pick a block from either parent."""
    n1 = len(parent1)
    n2 = len(parent2)
    child = []
    for i in range(max(n1, n2)):
        if i < n1 and i < n2:
            if rng.random() < 0.5:
                child.append(list(parent1[i]))
            else:
                child.append(list(parent2[i]))
        elif i < n1:
            child.append(list(parent1[i]))
        elif i < n2:
            child.append(list(parent2[i]))
    return child

def _mutate_block(block, v, k, t, triple_idx, counts, rng):
    """Replace a block with a new one that targets uncovered triples."""
    n_triples = len(counts)
    unc = np.where(counts == 0)[0]
    
    if len(unc) > 0:
        # Target an uncovered triple
        ts_idx = unc[rng.randint(0, len(unc) - 1)]
        ts = triples[ts_idx]
        new_block = list(ts)
    else:
        # Random block
        new_block = sorted(rng.sample(range(v), k))
    
    # Fill to size k
    while len(new_block) < k:
        candidates = [x for x in range(v) if x not in new_block]
        if not candidates:
            break
        # Pick point that maximizes coverage of currently-uncovered triples
        best_x = None
        best_gain = -1
        for x in candidates:
            gain = 0
            for S in itertools.combinations(new_block, t - 1):
                tset = tuple(sorted(S + (x,)))
                idx = triple_idx[tset]
                if counts[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None:
            break
        new_block.append(best_x)
    
    if len(new_block) < k:
        # Fill randomly
        remaining = [x for x in range(v) if x not in new_block]
        extra = rng.sample(remaining, k - len(new_block))
        new_block.extend(extra)
    
    new_block.sort()
    return new_block

def _genetic_algorithm(v, k, t, triples, triple_idx, rng, time_limit):
    """Genetic algorithm for covering design."""
    deadline = time.time() + time_limit
    n_triples = len(triples)
    
    # Initialize population
    pop_size = 20
    population = []
    for i in range(pop_size):
        blocks = _greedy_init(v, k, t, triples, triple_idx, rng, time_limit=1.5)
        population.append(blocks)
    
    # Evaluate initial population
    def evaluate(blocks):
        counts = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[triple_idx[ts]] += 1
        return _count_uncovered(counts), counts
    
    best_blocks = None
    best_energy = float('inf')
    
    # Evaluate and find best
    for blocks in population:
        energy, _ = evaluate(blocks)
        if energy < best_energy:
            best_energy = energy
            best_blocks = [list(b) for b in blocks]
    
    # If we already have a valid covering, try to reduce
    if best_energy == 0:
        best_blocks = _remove_redundant(best_blocks, triples, triple_idx, v, k, t, time_limit=3.0)
        if not _is_valid_covering(best_blocks, v, k, t, triples, triple_idx):
            best_blocks = _repair_covering(best_blocks, v, k, t, triples, triple_idx, rng, 3.0)
        best_energy = _count_uncovered(evaluate(best_blocks)[1])
    
    # Genetic algorithm loop
    gen = 0
    stagnation_count = 0
    max_stagnation = 30
    
    while time.time() < deadline - 10.0:
        gen += 1
        if stagnation_count > max_stagnation:
            break
        
        # Select parents (tournament selection)
        def tournament_select(pop, k_tourn=3):
            indices = rng.sample(range(len(pop)), k_tourn)
            best_idx = indices[0]
            best_e = float('inf')
            for idx in indices:
                e, _ = evaluate(pop[idx])
                if e < best_e:
                    best_e = e
                    best_idx = idx
            return pop[best_idx]
        
        new_population = []
        
        # Elitism: keep best
        new_population.append([list(b) for b in best_blocks])
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline - 5.0:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            
            child = _crossover(p1, p2, v, k, t, triple_idx, rng)
            
            # Mutation: replace some blocks
            n_mut = rng.randint(1, 3)
            counts = np.zeros(n_triples, dtype=np.int32)
            for b in child:
                for ts in itertools.combinations(b, t):
                    counts[triple_idx[ts]] += 1
            
            for _ in range(n_mut):
                if child:
                    idx = rng.randint(0, len(child) - 1)
                    child[idx] = _mutate_block(child[idx], v, k, t, triple_idx, counts, rng)
                    # Update counts
                    old_contrib = np.zeros(n_triples, dtype=np.int32)
                    for ts in itertools.combinations(child[idx], t):
                        old_contrib[triple_idx[ts]] += 1
                    # Recompute counts for simplicity
                    counts = np.zeros(n_triples, dtype=np.int32)
                    for b in child:
                        for ts in itertools.combinations(b, t):
                            counts[triple_idx[ts]] += 1
            
            # Repair if needed
            energy, _ = evaluate(child)
            if energy > 0:
                child = _repair_covering(child, v, k, t, triples, triple_idx, rng, time_limit=2.0)
                energy, _ = evaluate(child)
            
            new_population.append(child)
        
        # Evaluate new population
        for blocks in new_population:
            energy, _ = evaluate(blocks)
            if energy < best_energy:
                best_energy = energy
                best_blocks = [list(b) for b in blocks]
                stagnation_count = 0
            else:
                stagnation_count += 1
        
        population = new_population
        
        # Periodic redundancy removal on best
        if gen % 10 == 0 and time.time() < deadline - 15.0:
            reduced = _remove_redundant(best_blocks, triples, triple_idx, v, k, t, time_limit=3.0)
            if _is_valid_covering(reduced, v, k, t, triples, triple_idx):
                best_blocks = reduced
                best_energy = 0
            else:
                reduced = _repair_covering(reduced, v, k, t, triples, triple_idx, rng, 3.0)
                if _is_valid_covering(reduced, v, k, t, triples, triple_idx):
                    best_blocks = reduced
                    best_energy = 0
    
    return best_blocks, best_energy

def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0  # Safety margin for 150s kill
    
    # Precompute
    triples, triple_idx, n_triples = _precompute_triples(v, t)
    
    rng = random.Random(42)
    
    # Phase 1: Genetic algorithm
    best_blocks, best_energy = _genetic_algorithm(v, k, t, triples, triple_idx, rng, total_budget * 0.8)
    
    if best_blocks is None:
        best_blocks = _greedy_init(v, k, t, triples, triple_idx, rng, 10.0)
    
    # Phase 2: Final redundancy removal
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 2.0:
        reduced = _remove_redundant(best_blocks, triples, triple_idx, v, k, t, time_limit=remaining)
        if _is_valid_covering(reduced, v, k, t, triples, triple_idx):
            best_blocks = reduced
        else:
            repaired = _repair_covering(reduced, v, k, t, triples, triple_idx, rng, min(5.0, remaining))
            if _is_valid_covering(repaired, v, k, t, triples, triple_idx):
                best_blocks = repaired
    
    # Ensure validity
    if not _is_valid_covering(best_blocks, v, k, t, triples, triple_idx):
        best_blocks = _repair_covering(best_blocks, v, k, t, triples, triple_idx, rng, 10.0)
    
    return best_blocks

# EVOLVE-BLOCK-END
