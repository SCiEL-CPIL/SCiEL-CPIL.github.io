import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # 1. Precomputation
    # Generate all t-subsets (constraints)
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Generate all k-subsets (potential blocks/variables)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map t-subsets to indices for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, find which t-subsets it covers
    # block_tsubs[i] is a list of indices of t-subsets covered by block i
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
    
    # For each t-subset, find which blocks cover it
    # tsub_blocks[j] is a list of indices of blocks covering t-subset j
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Helper: Calculate cover count and uncovered count from a set of block indices
    def compute_coverage(block_indices):
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs[b_idx]] += 1
        uncovered_count = int(np.sum(cover_count == 0))
        return cover_count, uncovered_count

    # Helper: Fast Greedy Repair
    # Given a current set of blocks and cover_count, add blocks to cover all uncovered t-subsets
    def greedy_repair(selected_set, cover_count, uncovered_count):
        if uncovered_count == 0:
            return selected_set, cover_count, uncovered_count
        
        rng = random.Random(42)
        selected_set = set(selected_set)
        
        # List of uncovered t-subsets
        uncovered_list = [i for i in range(num_tsubs) if cover_count[i] == 0]
        
        while uncovered_list:
            if check_time(5.0):
                break
            
            # Pick a random uncovered t-subset to prioritize
            # Or pick the one with fewest covering blocks (most constrained)
            # For speed, just pick random from uncovered_list
            target_idx = random.choice(uncovered_list)
            
            # Find a block to cover it
            candidates = tsub_blocks[target_idx]
            if not candidates:
                break # Impossible
                
            # Choose block that covers most other uncovered t-subsets
            best_block = -1
            best_gain = -1
            
            # Limit candidates to speed up if list is huge (it's not, max 364)
            for b_idx in candidates:
                if b_idx in selected_set:
                    continue
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if cover_count[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            # Add block
            selected_set.add(best_block)
            for t_idx in block_tsubs[best_block]:
                if cover_count[t_idx] == 0:
                    uncovered_list.remove(t_idx) # This is O(N) but N is small enough? 
                    # Actually, removing from list is slow. Let's just mark it.
                cover_count[t_idx] += 1
            
            # Rebuild uncovered_list efficiently? 
            # For simplicity in repair, we can just re-scan if the list gets small.
            # Or better: don't use a dynamic list, just check cover_count in the loop.
            # Let's switch to a simple loop that picks any uncovered.
            # This repair is called often, so it must be fast.
            
            # Optimization: Instead of maintaining uncovered_list, just pick a random 
            # index and check if it's uncovered. If covered, try again.
            # If many are covered, this is slow.
            # Better: Keep the list but don't remove immediately. Just check if cover_count[target] == 0 before processing.
            
        # Final check
        uncovered_count = int(np.sum(cover_count == 0))
        return selected_set, cover_count, uncovered_count

    # Helper: Local Search Pruning (Safe Removal)
    def prune_blocks(selected_set, cover_count):
        if not selected_set:
            return selected_set, cover_count
            
        selected_list = list(selected_set)
        changed = True
        max_passes = 10
        for _ in range(max_passes):
            if check_time(2.0):
                break
            changed = False
            new_list = []
            for b_idx in selected_list:
                # Check if any t-subset in this block has count == 1
                is_essential = False
                for t_idx in block_tsubs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                        
                if not is_essential:
                    # Remove it
                    changed = True
                    for t_idx in block_tsubs[b_idx]:
                        cover_count[t_idx] -= 1
                else:
                    new_list.append(b_idx)
                    
            selected_list = new_list
            if not changed:
                break
                
        return set(selected_list), cover_count

    # Helper: Fitness Function
    def fitness(selected_set, uncovered_count):
        # Lower is better
        # Penalty for uncovered is huge
        return len(selected_set) + 10000 * uncovered_count

    # 2. Memetic Genetic Algorithm
    
    population_size = 20
    rng = random.Random(123)
    
    # Initialize population with diverse greedy solutions
    population = []
    for i in range(population_size):
        if check_time(10.0):
            break
        # Random seed for greedy init
        # We'll do a simple greedy init here
        selected_set = set()
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        uncovered_count = num_tsubs
        
        # Simple greedy: pick block covering most uncovered
        while uncovered_count > 0:
            if check_time(5.0):
                break
            # Find uncovered
            found = False
            for _ in range(50):
                idx = rng.randint(0, num_tsubs - 1)
                if cover_count[idx] == 0:
                    found = True
                    break
            if not found:
                # Linear scan
                for idx in range(num_tsubs):
                    if cover_count[idx] == 0:
                        found = True
                        break
                if not found:
                    break
            
            candidates = tsub_blocks[idx]
            if not candidates:
                break
                
            best_block = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in selected_set:
                    continue
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if cover_count[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected_set.add(best_block)
            for t_idx in block_tsubs[best_block]:
                if cover_count[t_idx] == 0:
                    uncovered_count -= 1
                cover_count[t_idx] += 1
                
        # Prune
        selected_set, cover_count = prune_blocks(selected_set, cover_count)
        uncovered_count = int(np.sum(cover_count == 0))
        
        if uncovered_count == 0:
            population.append((selected_set, cover_count, len(selected_set)))
            
    # If no valid population, force one
    if not population:
        # Fallback to simple greedy
        selected_set = set()
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        uncovered_count = num_tsubs
        while uncovered_count > 0:
            idx = rng.randint(0, num_tsubs - 1)
            if cover_count[idx] == 0:
                candidates = tsub_blocks[idx]
                if candidates:
                    b_idx = candidates[0]
                    selected_set.add(b_idx)
                    for t_idx in block_tsubs[b_idx]:
                        if cover_count[t_idx] == 0:
                            uncovered_count -= 1
                        cover_count[t_idx] += 1
        selected_set, cover_count = prune_blocks(selected_set, cover_count)
        population.append((selected_set, cover_count, len(selected_set)))

    best_solution = min(population, key=lambda x: x[2])
    best_size = best_solution[2]
    
    generations = 0
    max_generations = 100
    
    while generations < max_generations:
        if check_time(15.0):
            break
        generations += 1
        
        new_population = []
        
        # Keep the best few from previous generation (Elitism)
        population.sort(key=lambda x: x[2])
        for i in range(min(5, len(population))):
            new_population.append(population[i])
            
        # Generate offspring
        while len(new_population) < population_size:
            if check_time(5.0):
                break
                
            # Crossover
            parent1 = rng.choice(population)
            parent2 = rng.choice(population)
            
            # Uniform Crossover on block indices
            # Represent as sets
            set1 = parent1[0]
            set2 = parent2[0]
            
            # Combine and prune? Or just pick random blocks from union?
            # Better: Start with a random subset of the union, then repair.
            # Let's try: Take 50% of blocks from parent1, 50% from parent2?
            # Or: Take all blocks from parent1, then randomly remove some, then repair.
            
            # Strategy: "Recombination with Perturbation"
            # Start with parent1's blocks.
            child_set = set(parent1[0])
            
            # Remove some random blocks (mutation)
            num_remove = rng.randint(0, len(child_set) // 4)
            for _ in range(num_remove):
                if child_set:
                    b_to_remove = rng.choice(list(child_set))
                    child_set.remove(b_to_remove)
                    
            # Add some random blocks from parent2 that are not in child
            extra_blocks = parent2[0] - child_set
            if extra_blocks:
                num_add = rng.randint(0, min(5, len(extra_blocks)))
                for _ in range(num_add):
                    if extra_blocks:
                        b_to_add = rng.choice(list(extra_blocks))
                        child_set.add(b_to_add)
                        extra_blocks.remove(b_to_add)
                        
            # Compute coverage for child
            cover_count = np.zeros(num_tsubs, dtype=np.int32)
            for b_idx in child_set:
                cover_count[block_tsubs[b_idx]] += 1
            uncovered_count = int(np.sum(cover_count == 0))
            
            # Repair if invalid
            if uncovered_count > 0:
                child_set, cover_count, uncovered_count = greedy_repair(child_set, cover_count, uncovered_count)
                
            # Prune
            child_set, cover_count = prune_blocks(child_set, cover_count)
            uncovered_count = int(np.sum(cover_count == 0))
            
            if uncovered_count == 0:
                child_size = len(child_set)
                new_population.append((child_set, cover_count, child_size))
                
                # Local Search on the child (Simulated Annealing light)
                # Try to remove blocks that are not essential
                # We already pruned. Let's try a few random swaps?
                # For now, just prune again to be sure.
                
        # Update best
        for sol in new_population:
            if sol[2] < best_size:
                best_size = sol[2]
                best_solution = sol
                
        population = new_population
        
    # Final result
    final_set = best_solution[0]
    return [list(all_blocks[b]) for b in final_set]

if __name__ == "__main__":
    # Quick test
    blocks = construct(19, 8, 5)
    print(f"Found covering with {len(blocks)} blocks")
