import itertools
import random
import time
import math

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_t_indices = []
    for block in all_blocks:
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(t_subset_indices[ts])
        block_t_indices.append(t_indices)
        
    # Precompute t_subset_to_blocks mapping for fast lookup
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, block in enumerate(all_blocks):
        for t_idx in block_t_indices[b_idx]:
            t_to_blocks[t_idx].append(b_idx)

    def greedy_init():
        """
        Generates a valid covering using a greedy algorithm.
        Returns a list of block indices.
        """
        covered = [0] * num_t
        uncovered_indices = set(range(num_t))
        blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered t-subset to target
            target_idx = random.choice(list(uncovered_indices))
            
            # Find the block among those covering target_idx that has max gain
            best_gain = -1
            best_block_idx = -1
            
            # Sample a subset of candidate blocks if the list is huge?
            # Here C(21,4)=5985 is manageable.
            for b_idx in t_to_blocks[target_idx]:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if not covered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    if gain == 35: # Max possible gain for k=7, t=3
                        break
            
            blocks.append(best_block_idx)
            
            # Update coverage
            for t_idx in block_t_indices[best_block_idx]:
                if not covered[t_idx]:
                    covered[t_idx] = 1
                    uncovered_indices.remove(t_idx)
            
        return blocks

    def reduce_solution(blocks):
        """
        Given a list of block indices that form a valid cover,
        remove redundant blocks.
        Returns a list of reduced block indices.
        """
        if not blocks:
            return []
        
        # Convert to set for O(1) lookup and removal
        current_blocks = set(blocks)
        
        # Calculate cover counts
        cover_count = [0] * num_t
        for b_idx in current_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        # Iteratively remove redundant blocks
        # We shuffle the blocks to remove bias
        block_list = list(current_blocks)
        random.shuffle(block_list)
        
        changed = True
        while changed:
            changed = False
            # Iterate over a copy because we might modify the set
            for b_idx in list(block_list):
                if b_idx not in current_blocks:
                    continue
                
                # Check if b_idx is redundant
                is_redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    current_blocks.remove(b_idx)
                    block_list.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return list(current_blocks)

    def crossover(parent1, parent2):
        """
        Combines two solutions and reduces them.
        """
        combined = list(set(parent1 + parent2))
        return reduce_solution(combined)

    def mutation(blocks):
        """
        Randomly swaps a block in the solution with a random other block,
        then reduces.
        """
        if not blocks:
            return blocks
        
        new_blocks = list(blocks)
        # Pick a random index to mutate
        idx = random.randint(0, len(new_blocks) - 1)
        old_block = new_blocks[idx]
        
        # Pick a new random block
        new_block = random.choice(range(num_blocks))
        
        if new_block != old_block:
            new_blocks[idx] = new_block
            # Ensure we don't have duplicates if the new block already exists elsewhere?
            # reduce_solution handles duplicates implicitly by using a set? 
            # No, reduce_solution takes a list. Let's convert to set first in reduce.
            # Actually, let's just pass to reduce.
            return reduce_solution(new_blocks)
        
        return blocks

    # Main Genetic Algorithm Loop
    pop_size = 20
    population = []
    
    # Initialize population
    start_time = time.time()
    time_limit = 115 # Leave 5s buffer
    
    while len(population) < pop_size and time.time() - start_time < time_limit:
        init_blocks = greedy_init()
        reduced = reduce_solution(init_blocks)
        population.append(reduced)
    
    # Sort population by size
    population.sort(key=len)
    best_solution = population[0] if population else []
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Selection: Tournament
        def select():
            # Pick 3 random individuals, return the best
            idxs = [random.randint(0, len(population) - 1) for _ in range(3)]
            best_idx = idxs[0]
            for idx in idxs[1:]:
                if len(population[idx]) < len(population[best_idx]):
                    best_idx = idx
            return population[best_idx]
        
        new_population = []
        
        # Keep the best individual (Elitism)
        new_population.append(best_solution)
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() - start_time < time_limit:
            parent1 = select()
            parent2 = select()
            
            if random.random() < 0.5:
                # Crossover
                child = crossover(parent1, parent2)
            else:
                # Mutation of a selected parent
                child = mutation(parent1)
            
            # If the child is not a valid cover (shouldn't happen with reduce, but just in case)
            # We verify validity in reduce implicitly? No, reduce assumes input is valid.
            # Crossover of two valid covers is a valid cover.
            # Mutation might break validity if we replace a block and don't add one?
            # My mutation replaces a block. If the old block was the *only* cover for some t-subset,
            # and the new block doesn't cover it, the cover is broken.
            # So we MUST check validity after mutation.
            
            # Check validity of child
            valid = True
            if child:
                cover_count = [0] * num_t
                for b_idx in child:
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] += 1
                if any(c == 0 for c in cover_count):
                    valid = False
            else:
                valid = (num_t == 0)
                
            if valid:
                # If mutation broke validity, repair it by adding random blocks until valid?
                # Or just discard? Discarding is safer and faster.
                # But let's try a quick repair if invalid.
                if not valid:
                    # Add random blocks until valid
                    cover_count = [0] * num_t
                    for b_idx in child:
                        for t_idx in block_t_indices[b_idx]:
                            cover_count[t_idx] += 1
                    uncovered = [i for i, c in enumerate(cover_count) if c == 0]
                    while uncovered:
                        target = random.choice(uncovered)
                        candidates = t_to_blocks[target]
                        if not candidates:
                            break
                        new_b = random.choice(candidates)
                        if new_b not in child:
                            child.append(new_b)
                            for t_idx in block_t_indices[new_b]:
                                if cover_count[t_idx] == 0:
                                    uncovered.remove(t_idx)
                                cover_count[t_idx] += 1
                    # Reduce again
                    child = reduce_solution(child)
                    
                new_population.append(child)
            else:
                # Discard invalid child
                pass
        
        # Update population
        population = new_population
        population.sort(key=len)
        
        if population[0]:
            best_solution = population[0]
            
        # Early stop if we hit lower bound
        if len(best_solution) <= 69:
            break
            
    # Final reduction on the best solution
    if best_solution:
        best_solution = reduce_solution(best_solution)
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[idx]) for idx in best_solution]
    
    return final_blocks
