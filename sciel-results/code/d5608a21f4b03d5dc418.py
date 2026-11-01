"""
Genetic Algorithm for covering design C(24,7,3).
Strategy: Evolve a population of fixed-size block sets.
Fitness: Minimize (num_uncovered * large_penalty + num_blocks).
Moves: Crossover (combine blocks from two parents), Mutation (replace random block with a new one).
"""
import itertools
import random
import time
import sys
from collections import defaultdict

import numpy as np

def _generate_greedy_block(uncovered_sets, current_blocks, v, k, t, rng):
    """
    Greedily construct a block that covers the most uncovered t-sets.
    Returns a tuple of k distinct integers.
    """
    if not uncovered_sets:
        # If no uncovered sets, just generate a random block to maintain diversity
        return tuple(sorted(rng.sample(range(v), k)))
    
    # Start with a random uncovered set
    first_set = rng.choice(list(uncovered_sets))
    block = set(first_set)
    
    # Greedily add elements to maximize coverage of uncovered sets
    while len(block) < k:
        best_x = None
        best_gain = -1
        # Candidate elements
        candidates = [x for x in range(v) if x not in block]
        
        for x in candidates:
            gain = 0
            # Check which uncovered sets would be covered by adding x
            # An uncovered set S is covered by x if x in S and S \ {x} subset of block
            # Since |S| = t, and we are adding x, S must contain x and t-1 elements from block.
            # We can iterate over combinations of (t-1) elements from block to form potential sets.
            # But it's faster to just check if any uncovered set containing x is a subset of block U {x}.
            # For v=24, k=7, t=3, this is manageable.
            
            # Optimization: Precompute neighbors? 
            # For simplicity and speed in this specific context:
            # Iterate over all uncovered sets containing x? That's expensive if many uncovered.
            # Better: Iterate over combinations of (t-1) from block.
            if t == 3:
                # S = {a, b, x} where a,b in block
                for a, b in itertools.combinations(block, 2):
                    s = frozenset((a, b, x))
                    if s in uncovered_sets:
                        gain += 1
            else:
                # General case (slower but correct)
                for S in itertools.combinations(block, t-1):
                    if x not in S:
                        s = frozenset(S + (x,))
                        if s in uncovered_sets:
                            gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            # No gain, pick random
            best_x = rng.choice(candidates)
        block.add(best_x)
        
    return tuple(sorted(block))

def _evaluate_population(population, v, k, t, tset_to_idx, num_tsets):
    """
    Evaluate fitness for each individual in population.
    Returns list of (fitness, num_blocks, num_uncovered).
    """
    results = []
    for blocks in population:
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = tset_to_idx[ts]
                counts[idx] += 1
        
        num_uncovered = np.sum(counts == 0)
        num_blocks = len(blocks)
        
        # Fitness: We want to minimize uncovered, then minimize blocks.
        # Since valid covering has 0 uncovered, we prioritize that.
        # If uncovered > 0, fitness is very high (bad).
        # If uncovered == 0, fitness is num_blocks.
        if num_uncovered > 0:
            fitness = num_uncovered * 10000 + num_blocks
        else:
            fitness = num_blocks
            
        results.append((fitness, num_blocks, num_uncovered))
        
    return results

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0
    deadline = start_time + total_budget - 5.0
    
    # Precompute t-set indices
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    num_tsets = len(all_tsets)
    
    rng = random.Random(42)
    
    # 1. Initialize Population
    pop_size = 20
    population = []
    
    # First individual: Greedy covering
    # We need a fast greedy to get a valid or near-valid start
    uncovered = set(all_tsets)
    blocks = []
    while uncovered and time.time() < deadline:
        b = _generate_greedy_block(uncovered, blocks, v, k, t, rng)
        blocks.append(b)
        # Update uncovered
        for ts in itertools.combinations(b, t):
            uncovered.discard(ts)
    population.append(blocks)
    
    # Other individuals: Perturbed greedy or random
    for _ in range(pop_size - 1):
        # Start with a copy of the best so far and mutate
        base = population[0]
        new_blocks = [list(b) for b in base]
        # Remove some random blocks
        num_remove = rng.randint(0, 3)
        for _ in range(num_remove):
            if new_blocks:
                idx = rng.randint(0, len(new_blocks)-1)
                new_blocks.pop(idx)
        # Add random blocks to fix
        # Calculate uncovered
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b in new_blocks:
            for ts in itertools.combinations(b, t):
                counts[tset_to_idx[ts]] += 1
        uncovered_idx = np.where(counts == 0)[0]
        uncovered_sets = {all_tsets[i] for i in uncovered_idx}
        
        # Add blocks to cover
        while uncovered_sets and len(new_blocks) < 100: # Safety cap
             b = _generate_greedy_block(uncovered_sets, new_blocks, v, k, t, rng)
             if b not in new_blocks:
                 new_blocks.append(b)
                 for ts in itertools.combinations(b, t):
                     uncovered_sets.discard(ts)
        
        population.append(new_blocks)
    
    # 2. GA Loop
    best_fitness = float('inf')
    best_blocks = None
    stagnation = 0
    max_stagnation = 50
    
    while time.time() < deadline:
        # Evaluate
        results = _evaluate_population(population, v, k, t, tset_to_idx, num_tsets)
        
        # Find best in population
        pop_best_idx = min(range(len(population)), key=lambda i: results[i][0])
        pop_best_fit = results[pop_best_idx][0]
        
        if pop_best_fit < best_fitness:
            best_fitness = pop_best_fit
            best_blocks = [list(b) for b in population[pop_best_idx]]
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                # Restart or heavy mutation
                # Break if we are stuck for too long
                break
        
        # Selection: Tournament
        def select():
            i1, i2 = rng.sample(range(pop_size), 2)
            if results[i1][0] < results[i2][0]:
                return population[i1]
            else:
                return population[i2]
        
        # Crossover and Mutation
        new_population = []
        
        # Keep best individual (Elitism)
        new_population.append([list(b) for b in population[pop_best_idx]])
        
        while len(new_population) < pop_size:
            p1 = select()
            p2 = select()
            
            # Crossover: Take some blocks from p1, some from p2
            # Ensure distinctness and coverage
            # Simple strategy: Randomly sample blocks from p1 and p2 to form a new set
            # We want to maintain a similar size
            target_size = min(len(p1), len(p2)) + rng.randint(-2, 2)
            target_size = max(10, target_size) # Min size
            
            # Combine unique blocks
            combined = list(set(p1 + p2))
            
            # If we have too many, sample
            if len(combined) > target_size:
                # Prefer blocks from the better parent
                # Simple heuristic: Sort by how many "rare" t-sets they cover? 
                # For now, random sample
                child = rng.sample(combined, target_size)
            else:
                child = combined
                # Fill if needed
                while len(child) < target_size:
                    b = _generate_greedy_block(set(all_tsets), child, v, k, t, rng)
                    if b not in child:
                        child.append(b)
            
            # Mutation: Replace a few random blocks
            num_mutate = rng.randint(0, 3)
            for _ in range(num_mutate):
                if child:
                    idx = rng.randint(0, len(child)-1)
                    # Generate a new block
                    # Use current coverage to guide it
                    counts = np.zeros(num_tsets, dtype=np.int32)
                    for b in child:
                        for ts in itertools.combinations(b, t):
                            counts[tset_to_idx[ts]] += 1
                    uncovered_idx = np.where(counts == 0)[0]
                    uncovered_sets = {all_tsets[i] for i in uncovered_idx}
                    
                    new_b = _generate_greedy_block(uncovered_sets, child, v, k, t, rng)
                    if new_b not in child:
                        child[idx] = new_b
            
            # Ensure distinct blocks
            # Convert to tuple for dedup
            seen = set()
            unique_child = []
            for b in child:
                tb = tuple(b)
                if tb not in seen:
                    seen.add(tb)
                    unique_child.append(tb)
            
            new_population.append(unique_child)
        
        population = new_population
        
        # Check time
        if time.time() > deadline:
            break

    # Final Verification
    if best_blocks is None:
        # Fallback to first population member
        best_blocks = population[0]
        
    # Verify validity
    counts = np.zeros(num_tsets, dtype=np.int32)
    for b in best_blocks:
        for ts in itertools.combinations(b, t):
            counts[tset_to_idx[ts]] += 1
            
    if np.any(counts == 0):
        # Invalid, try to repair
        uncovered_idx = np.where(counts == 0)[0]
        uncovered_sets = {all_tsets[i] for i in uncovered_idx}
        while uncovered_sets and time.time() < deadline:
            b = _generate_greedy_block(uncovered_sets, best_blocks, v, k, t, rng)
            if b not in best_blocks:
                best_blocks.append(b)
                for ts in itertools.combinations(b, t):
                    uncovered_sets.discard(ts)
                    counts[tset_to_idx[ts]] += 1
            else:
                # If block already exists, we are stuck with duplicates or need a different one
                # Force a random block
                b = tuple(sorted(rng.sample(range(v), k)))
                best_blocks.append(b)
                for ts in itertools.combinations(b, t):
                    uncovered_sets.discard(ts)
                    counts[tset_to_idx[ts]] += 1

    # Clean up duplicates
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(list(b))
            
    return unique_blocks

# EVOLVE-BLOCK-END
