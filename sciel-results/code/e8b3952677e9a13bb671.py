import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Precompute all k-subsets (blocks) is too expensive to store all, 
    # but we can generate candidates on the fly or precompute mapping for specific strategies.
    
    # For v=24, k=7, t=3:
    # Total blocks C(24,7) = 346104.
    # Total triples C(24,3) = 2024.
    
    # We will use a bitset representation for triples in a block?
    # Or just a set of triple indices.
    
    # Greedy Strategy:
    # 1. Maintain a list `uncovered` of triple indices that are not yet covered.
    # 2. To pick the next block:
    #    - Pick a random (or first) uncovered triple T.
    #    - Generate all blocks containing T. There are C(v-t, k-t) = C(21, 4) = 5985 blocks.
    #    - For each candidate block, count how many triples in `uncovered` it covers.
    #    - Pick the candidate with the max count.
    #    - Add it to the solution.
    #    - Update `uncovered`.
    
    # Optimization: 
    # Instead of iterating all 5985 candidates every time, we can be smarter, 
    # but 5985 * 2024 checks is ~12M ops per step. 
    # Number of steps is roughly n_triples / (k choose t) = 2024 / 35 ~ 58 steps.
    # 58 * 12M = 700M ops. This might be too slow in pure Python.
    
    # Faster approach:
    # Precompute for each triple, the list of blocks containing it.
    # This uses memory but speeds up access.
    # C(24,3) = 2024 triples.
    # Each triple is in C(21, 4) = 5985 blocks.
    # Total storage: 2024 * 5985 ~ 12M integers. ~100MB. Acceptable.
    
    # Let's precompute `blocks_containing_triple`
    # Actually, generating all blocks and mapping them is C(24,7) = 346k blocks.
    # For each block, it contains C(7,3) = 35 triples.
    # 346k * 35 = 12M entries.
    # We can create a list of lists: `triple_to_blocks[i]` contains indices of blocks covering triple i.
    
    # Step 1: Generate all blocks and their triple masks/indices.
    all_blocks = []
    block_to_triples = []
    
    # To save time, we might not need all blocks if we use the "pick uncovered triple" strategy.
    # But precomputing the mapping is a one-time cost.
    # Let's try to precompute.
    
    # Initialize mapping
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for block in itertools.combinations(range(v), k):
        block_idx = len(all_blocks)
        all_blocks.append(block)
        triples_in_block = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            triples_in_block.append(idx)
            triple_to_blocks[idx].append(block_idx)
        block_to_triples.append(triples_in_block)
        
    # Now, Greedy Construction
    uncovered = set(range(n_triples))
    selected_blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick an uncovered triple
        # Heuristic: Pick the triple with the fewest available blocks? 
        # Or just the first one.
        # Since all blocks are available initially, "fewest available" is static until we pick blocks.
        # Let's pick the triple that is in the fewest *selected* blocks? No, we want to cover it.
        # Standard greedy: Pick the block that covers the most uncovered triples.
        # To speed up, we pick a random uncovered triple T, and only consider blocks containing T.
        # Among those, pick the one covering the most *other* uncovered triples.
        
        # Random choice to allow restarts or just deterministic?
        # Let's pick the first one for determinism, or random for diversity.
        # We'll do a few restarts if time permits.
        
        t_idx = next(iter(uncovered))
        
        best_block_idx = -1
        max_covered = -1
        
        for b_idx in triple_to_blocks[t_idx]:
            # Count uncovered triples covered by this block
            # block_to_triples[b_idx] is a list of triple indices
            covered_count = 0
            for tr_idx in block_to_triples[b_idx]:
                if tr_idx in uncovered:
                    covered_count += 1
                    # Early exit if we can't beat best? 
                    # Max possible is len(block_to_triples[b_idx]) = 35.
            
            if covered_count > max_covered:
                max_covered = covered_count
                best_block_idx = b_idx
                if max_covered == len(block_to_triples[b_idx]):
                    # Can't do better for this block
                    pass
                    
        if best_block_idx == -1:
            break # Should not happen
            
        selected_blocks.append(best_block_idx)
        
        # Update uncovered
        for tr_idx in block_to_triples[best_block_idx]:
            uncovered.discard(tr_idx)
            
    # Convert to list of lists
    solution = [list(all_blocks[i]) for i in selected_blocks]
    
    # Phase 2: Reduction
    # Check if any block is redundant.
    # A block is redundant if all its triples are covered by other blocks.
    # We can do this by counting coverage.
    
    def is_redundant(block_idx, selected_set):
        # Check if every triple in this block is covered by another block in selected_set
        for tr_idx in block_to_triples[block_idx]:
            # Check other blocks in selected_set
            covered_by_other = False
            for other_idx in selected_set:
                if other_idx == block_idx:
                    continue
                if tr_idx in block_to_triples[other_idx]:
                    covered_by_other = True
                    break
            if not covered_by_other:
                return False
        return True

    # To make this faster, we can maintain a count of how many blocks cover each triple.
    # Let's rebuild the solution as a list of indices and use counts.
    
    current_solution = selected_blocks[:]
    
    # Optimization for reduction:
    # Iterate through blocks. If removing it doesn't break coverage, remove it.
    # Repeat until no more removals possible.
    
    changed = True
    while changed:
        changed = False
        # Calculate coverage counts
        # n_triples is 2024. current_solution size is ~70.
        # 70 * 35 = 2450 operations to build counts.
        # Then for each block, check if min count of its triples is > 1.
        
        coverage_count = [0] * n_triples
        for b_idx in current_solution:
            for tr_idx in block_to_triples[b_idx]:
                coverage_count[tr_idx] += 1
        
        for i, b_idx in enumerate(current_solution):
            # Check if all triples in b_idx have coverage_count > 1
            redundant = True
            for tr_idx in block_to_triples[b_idx]:
                if coverage_count[tr_idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                current_solution.pop(i)
                changed = True
                # Recalculate counts? 
                # We can just break and restart the loop for simplicity, 
                # or update counts. Restarting is safer and fast enough for 70 blocks.
                break
                
    solution = [list(all_blocks[i]) for i in current_solution]
    
    # Phase 3: Local Search (Swap/Repair)
    # Try to swap a block with another to reduce size or improve future greedy?
    # Since we already reduced, size is minimal for this set.
    # Can we find a smaller set?
    # Try: Remove a block, then use greedy to repair. If new size < old size, keep.
    
    # This is expensive. Let's do a limited number of attempts.
    # Given the time spent on precomputation and greedy, we might have little time left.
    # Precomputation: 346k blocks * 35 triples = 12M operations. ~1-2 seconds.
    # Greedy: ~60 steps * 6000 candidates * 35 checks = 12M operations. ~1-2 seconds.
    # Reduction: ~70 blocks * 35 checks * 70 others = 170k operations. Fast.
    # Total ~5 seconds. We have 135 seconds left.
    
    # Let's implement a simple local search:
    # 1. Pick a random block from current_solution.
    # 2. Remove it.
    # 3. Find uncovered triples.
    # 4. Run a greedy repair to cover them.
    # 5. If len(new_solution) < len(old_solution), accept.
    # 6. If len(new_solution) == len(old_solution), maybe accept with some probability (simulated annealing)?
    #    Or just reject to keep it simple and fast.
    
    # To make repair fast, we reuse the `triple_to_blocks` mapping.
    
    best_solution = current_solution[:]
    
    # We will try a few restarts of the whole process if time allows, 
    # or just local search on the current one.
    
    # Let's do a limited local search loop.
    max_ls_iterations = 1000
    for _ in range(max_ls_iterations):
        if time.time() > deadline - 5.0:
            break
            
        if not best_solution:
            break
            
        # Pick a random block to remove
        remove_idx = random.randint(0, len(best_solution) - 1)
        block_to_remove = best_solution[remove_idx]
        
        # Tentatively remove it
        new_solution = best_solution[:]
        new_solution.pop(remove_idx)
        
        # Find uncovered triples
        # We need to know which triples are now uncovered.
        # We can recompute coverage or track it.
        # Recomputing is 70*35 = 2450 ops.
        
        coverage_count = [0] * n_triples
        for b_idx in new_solution:
            for tr_idx in block_to_triples[b_idx]:
                coverage_count[tr_idx] += 1
        
        uncovered_triples = []
        for tr_idx in range(n_triples):
            if coverage_count[tr_idx] == 0:
                uncovered_triples.append(tr_idx)
                
        if not uncovered_triples:
            # We removed a redundant block? 
            # This shouldn't happen if reduction was perfect, but maybe due to order.
            # If so, we improved!
            best_solution = new_solution
            continue
            
        # Greedy repair
        # Add blocks to cover uncovered_triples
        # We can use the same logic as Phase 1 but only for these triples.
        
        # To speed up, we pick an uncovered triple, find a block covering it, etc.
        # Since the number of uncovered triples is small (at most 35), this should be fast.
        
        temp_uncovered = set(uncovered_triples)
        
        while temp_uncovered:
            if time.time() > deadline - 2.0:
                break
                
            t_idx = next(iter(temp_uncovered))
            best_b_idx = -1
            max_cov = -1
            
            for b_idx in triple_to_blocks[t_idx]:
                # Only consider blocks not already in new_solution? 
                # Actually, we can consider any block. If it's already there, it covers 0 new ones.
                # So we can skip if b_idx in new_solution.
                if b_idx in new_solution:
                    continue
                    
                cov = 0
                for tr_idx in block_to_triples[b_idx]:
                    if tr_idx in temp_uncovered:
                        cov += 1
                        
                if cov > max_cov:
                    max_cov = cov
                    best_b_idx = b_idx
                    
            if best_b_idx == -1:
                break # Should not happen
                
            new_solution.append(best_b_idx)
            for tr_idx in block_to_triples[best_b_idx]:
                temp_uncovered.discard(tr_idx)
                
        if len(new_solution) < len(best_solution):
            best_solution = new_solution
        elif len(new_solution) == len(best_solution):
            # Optional: accept with small probability to escape local minima?
            # For now, reject to keep solution stable.
            pass

    return [list(all_blocks[i]) for i in best_solution]
