import time
import random
import itertools
import math
import numpy as np
from collections import defaultdict, deque

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    # All t-subsets (items to cover)
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx_t = {s: i for i, s in enumerate(all_t)}
    
    # All k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    M = len(all_blocks)
    
    # block_cover[b] = list of indices of t-subsets covered by block b
    block_cover = []
    for b in all_blocks:
        covered = [idx_t[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[i] = list of block indices that cover t-subset i
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper: Check if a list of block indices forms a valid cover
    def is_valid(blocks):
        if not blocks:
            return N == 0
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return np.all(cnt > 0)

    # Helper: Compute coverage counts
    def get_counts(blocks):
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return cnt

    # 2. Initial Construction: Greedy Set Cover
    # We need a valid starting population. Greedy is fast and provides a baseline.
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = set(range(N))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered item to break symmetry
            target = next(iter(uncovered))
            candidates = t_to_blocks[target]
            if not candidates:
                return None 
            
            best_b = -1
            max_gain = -1
            
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_b = bi
                    
            if best_b == -1:
                return None
                
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
                
        return blocks

    # Run a few quick greedy restarts to find a good starting point
    best_init = None
    best_init_len = float('inf')
    for s in range(5):
        if time.time() > deadline - 10.0:
            break
        cand = greedy_init(s)
        if cand and len(cand) < best_init_len:
            best_init_len = len(cand)
            best_init = cand
            
    if not best_init:
        # Fallback
        return [list(b) for b in all_blocks[:100]] 

    # 3. Genetic Algorithm Strategy
    # Representation: A set of block indices.
    # However, for GA, it's often easier to work with a population of "solutions" (lists of blocks).
    # Crossover: We need to combine two valid covers into a valid cover.
    # Simple union is valid but large. We can try to prune.
    # Mutation: 
    #   - Add a random block (increases size, might enable future drops)
    #   - Remove a redundant block (decreases size, if possible)
    #   - Swap a block with another (keeps size, explores neighborhood)
    
    # We will maintain a population of the best K solutions found so far.
    # To save memory and time, we store them as frozensets of block indices.
    
    population = [frozenset(best_init)]
    best_solution = set(best_init)
    best_score = len(best_solution)
    
    # Precompute block_cover as a list of lists for fast iteration.
    # We also need to quickly check if a block is redundant in a given solution.
    
    def prune_set(s):
        """Prune a set of blocks to be minimal."""
        s = set(s)
        if not s:
            return s
        cnt = np.zeros(N, dtype=np.int32)
        for bi in s:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        
        # Identify blocks that can be removed
        # A block b is redundant if for all t in block_cover[b], cnt[t] > 1.
        # We can iterate and remove.
        # To be efficient, we can just check each block.
        to_remove = []
        for bi in s:
            redundant = True
            for ti in block_cover[bi]:
                if cnt[ti] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(bi)
                
        if to_remove:
            for bi in to_remove:
                s.discard(bi)
                for ti in block_cover[bi]:
                    cnt[ti] -= 1
            # Recursively prune? Sometimes removing one makes another redundant.
            # For speed, we do one pass, but we can repeat if time permits.
            # Let's do a second pass if we removed anything.
            if to_remove:
                # Re-calculate counts? No, we updated them.
                # Check again
                to_remove2 = []
                for bi in s:
                    redundant = True
                    for ti in block_cover[bi]:
                        if cnt[ti] == 1:
                            redundant = False
                            break
                    if redundant:
                        to_remove2.append(bi)
                if to_remove2:
                    for bi in to_remove2:
                        s.discard(bi)
                        for ti in block_cover[bi]:
                            cnt[ti] -= 1
        return s

    # Apply initial pruning
    best_solution = prune_set(best_solution)
    best_score = len(best_solution)
    population = [frozenset(best_solution)]
    
    # GA Loop
    # Population size: small, e.g., 10-20
    pop_size = 15
    # Fill initial population with variations of the best solution
    for _ in range(pop_size - 1):
        if time.time() > deadline - 5.0:
            break
        # Random mutation of best solution
        sol = set(best_solution)
        # Try to add a random block
        if random.random() < 0.5 and M > len(sol):
            random_block = random.randrange(M)
            sol.add(random_block)
        # Prune
        sol = prune_set(sol)
        if len(sol) < best_score:
            best_solution = sol
            best_score = len(sol)
        population.append(frozenset(sol))
        
    # Main GA Loop
    while time.time() < deadline - 2.0:
        # Selection: Pick two parents
        # Weighted by fitness (smaller is better)
        # Simple: Random selection from top half
        half = max(1, len(population) // 2)
        # Sort population by size
        population.sort(key=lambda x: len(x))
        # Pick from top half
        p1 = random.choice(population[:half])
        p2 = random.choice(population[:half])
        
        # Crossover: Union of parents
        # This guarantees validity.
        child = set(p1) | set(p2)
        
        # Prune the child
        child = prune_set(child)
        
        # Mutation
        # 1. Try to drop a random block (if redundant)
        if child:
            # Pick a random block from child
            # Check if it can be dropped
            # To speed up, we can just try a few random drops
            for _ in range(3):
                if not child:
                    break
                rand_block = random.choice(list(child))
                # Check if rand_block is redundant in child
                # This is O(N) roughly.
                # Let's use the prune_set logic but just for this block?
                # Or just try to remove and re-prune.
                # Actually, prune_set already does this.
                # So we can just add a random block and prune again.
                # Or remove a random block and check validity? No, must stay valid.
                # So mutation should be: Add random block, then Prune.
                # Or: Swap a block with another.
                
                # Let's do: Add a random block, then Prune.
                # This allows exploration of larger sets which might contain smaller subcovers.
                if random.random() < 0.7:
                    rand_add = random.randrange(M)
                    child.add(rand_add)
                    child = prune_set(child)
                else:
                    # Try to swap
                    if child:
                        old_b = random.choice(list(child))
                        # Find a block that covers the "weak" spots of old_b
                        # Weak spots are t-subsets covered only by old_b in child.
                        # If none, old_b is redundant (should have been pruned).
                        # So there is at least one weak spot.
                        # Find candidates covering all weak spots.
                        # This is expensive. 
                        # Alternative: Just add a random block and prune.
                        # Or: Remove old_b, add a random block, check validity?
                        # If invalid, revert.
                        # This is a "random walk" mutation.
                        
                        # Let's stick to Add+Prune as it's safe and simple.
                        pass

        # Evaluate child
        child_size = len(child)
        
        # Replacement
        # If child is better than worst in population, replace worst.
        # Or if better than best, replace best.
        if child_size < best_score:
            best_solution = child
            best_score = child_size
            population.append(frozenset(child))
            # Keep population size bounded
            if len(population) > pop_size * 2:
                population.sort(key=lambda x: len(x))
                population = population[:pop_size]
        else:
            # Replace a random member if child is smaller than it
            if population:
                idx = random.randrange(len(population))
                if child_size < len(population[idx]):
                    population[idx] = frozenset(child)
                elif child_size == len(population[idx]) and random.random() < 0.1:
                    # Diversity
                    population[idx] = frozenset(child)
            else:
                population.append(frozenset(child))
                
            # Bound population
            if len(population) > pop_size:
                population.sort(key=lambda x: len(x))
                population = population[:pop_size]

    # Final validation
    if not is_valid(best_solution):
        # Fallback to initial valid cover
        best_solution = set(best_init)
        best_score = len(best_solution)

    return [list(all_blocks[bi]) for bi in best_solution]
