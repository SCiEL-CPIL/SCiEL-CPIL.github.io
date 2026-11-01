import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Map t-subsets to integer IDs for faster lookup and counting
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    # Pre-compute all possible k-subsets? No, C(24,7) is 346104, which is manageable but might be slow to generate all at start.
    # We will generate blocks on the fly or sample from a pool.
    # For v=24, k=7, total blocks is ~346k. We can pre-generate a random sample of blocks to use for mutation/repair.
    # However, generating all is feasible. Let's see time.
    # C(24, 7) = 346104. Generating list of lists might take ~0.5s.
    # Let's stick to generating blocks on demand or a large random sample to save memory/time if not needed.
    # Actually, for repair, we need to find a block covering a specific triple.
    # If we have a list of ALL blocks, we can index them by the triples they cover.
    # But that's a lot of memory (346k blocks * 35 triples each).
    
    # Alternative: On-the-fly generation for repair.
    # When a triple is uncovered, we pick a random block containing it.
    # We can pre-compute for each triple, the list of all blocks containing it?
    # That's also large.
    
    # Let's use a simpler repair: 
    # If triple T is uncovered, pick a random set of k-3 points from the remaining v-3 points.
    # This creates a valid block containing T.
    
    def random_block_containing(triple):
        triple_set = set(triple)
        remaining = [x for x in all_points if x not in triple_set]
        # Need k - t more points
        extra = random.sample(remaining, k - t)
        block = list(triple) + extra
        random.shuffle(block)
        return block

    def greedy_cover():
        """Fast greedy cover to initialize or repair."""
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        
        while True:
            if not np.any(uncovered):
                break
            
            # Pick a random uncovered triple
            # To speed up, we can maintain a list of uncovered indices, but numpy any is fast enough for 2024 items.
            # Better: sample until we find an uncovered one.
            while True:
                idx = random.randint(0, num_triples - 1)
                if uncovered[idx]:
                    break
            
            seed_triple = all_t_subsets[idx]
            
            # Construct a block containing this triple
            # To be better than random, we can try to maximize coverage of other uncovered triples.
            # But for speed in repair, random is often sufficient if followed by local search.
            # Let's do a mini-greedy extension.
            block = list(seed_triple)
            block_set = set(block)
            
            # Extend block to size k
            while len(block) < k:
                best_gain = -1
                best_points = []
                # Calculate potential gain for each candidate point
                # We only care about triples formed with the current block elements
                # Specifically, triples that include the new point and t-1 points from the current block.
                # Since t=3, we need pairs from the current block.
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in all_points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for pair in block_pairs:
                        triple = tuple(sorted((x,) + pair))
                        idx_t = triple_to_idx[triple]
                        if uncovered[idx_t]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_points:
                    chosen = random.choice(best_points)
                else:
                    available = [x for x in all_points if x not in block_set]
                    if not available:
                        break
                    chosen = random.choice(available)
                
                block.append(chosen)
                block_set.add(chosen)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        
        return blocks

    def remove_redundant_blocks(blocks):
        """Remove blocks that are not necessary for coverage."""
        if not blocks:
            return blocks
        
        # Count coverage for each triple
        triple_count = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                
                # Update counts
                for i in to_remove:
                    block = blocks[i]
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                
                blocks = new_blocks
                changed = True
        
        return blocks

    def is_valid_cover(blocks):
        """Check if blocks cover all t-subsets."""
        # This is expensive, so we only use it sparingly or rely on repair.
        # For GA fitness, we assume individuals are repaired.
        pass

    def repair_and_optimize(blocks):
        """Ensure blocks cover all triples and remove redundancies."""
        # Check for uncovered triples
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                covered[idx] = 1
        
        uncovered_indices = np.where(covered == 0)[0]
        
        # Add blocks to cover uncovered triples
        if len(uncovered_indices) > 0:
            # To avoid duplicates, we track added blocks
            added = set()
            for idx in uncovered_indices:
                triple = all_t_subsets[idx]
                # Try a few random blocks containing this triple
                for _ in range(10):
                    new_block = random_block_containing(triple)
                    new_block.sort()
                    key = tuple(new_block)
                    if key not in added and key not in [tuple(b) for b in blocks]:
                        blocks.append(new_block)
                        added.add(key)
                        break
            # If we still have uncovered, force add
            covered = np.zeros(num_triples, dtype=np.uint8)
            for block in blocks:
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    covered[idx] = 1
            
            uncovered_indices = np.where(covered == 0)[0]
            for idx in uncovered_indices:
                triple = all_t_subsets[idx]
                new_block = random_block_containing(triple)
                new_block.sort()
                blocks.append(new_block)
        
        # Remove redundancies
        blocks = remove_redundant_blocks(blocks)
        
        return blocks

    # --- Genetic Algorithm Setup ---
    pop_size = 20
    elite_size = 5
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        blocks = greedy_cover()
        blocks = remove_redundant_blocks(blocks)
        population.append(blocks)
    
    best_blocks = min(population, key=len)
    
    while time.time() < deadline:
        # Evaluate fitness (length)
        population.sort(key=len)
        
        # Elitism: keep best
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Tournament selection
            p1 = random.choice(population[:10]) # Pick from top 10
            p2 = random.choice(population[:10])
            
            # Crossover
            if random.random() < 0.7:
                # Uniform crossover: pick each block from p1 or p2
                child_blocks = []
                seen = set()
                for block in p1 + p2:
                    key = tuple(block)
                    if key not in seen:
                        child_blocks.append(block)
                        seen.add(key)
            else:
                # No crossover, just copy one parent
                child_blocks = p1[:] if random.random() < 0.5 else p2[:]
            
            # Mutation
            # 1. Random removal (if valid after repair)
            if random.random() < 0.3 and len(child_blocks) > 10:
                # Remove a random block, repair will add it back if needed
                child_blocks.pop(random.randint(0, len(child_blocks) - 1))
            
            # 2. Random addition
            if random.random() < 0.2:
                # Add a random block
                new_block = random.sample(all_points, k)
                new_block.sort()
                child_blocks.append(new_block)
            
            # Repair and Optimize
            child_blocks = repair_and_optimize(child_blocks)
            
            new_population.append(child_blocks)
        
        # Update population
        population = new_population
        
        # Update best
        current_best = min(population, key=len)
        if len(current_best) < len(best_blocks):
            best_blocks = current_best
            # Optional: Add best to population to reinforce it
            # population[0] = best_blocks

    return best_blocks

if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Found covering with {len(blocks)} blocks")
