import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map triples to indices for fast lookup
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # For each block, compute the bitset of triples it covers
    # Using Python integers as bitsets is efficient for num_triples ~ 2024
    block_bitsets = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        bitset = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bitset |= (1 << idx)
        block_bitsets[i] = bitset
        
    # Initial solution: Greedy construction
    # We use a vectorized approach for speed.
    # uncovered is a bitset of all triples
    uncovered = (1 << num_triples) - 1
    selected_indices = []
    selected_set = set()
    
    # To make greedy faster, we can precompute the gain of each block?
    # No, gain depends on current uncovered.
    # But we can compute `block_bitsets[i] & uncovered` quickly.
    
    # Standard greedy is O(N^2 * BitOps).
    # N ~ 76. 76^2 = 5776. BitOps on 2024 bits are fast.
    # 5776 * (time for big int AND + bit_count) is very fast.
    
    while uncovered and time.time() < deadline - 10.0:
        best_gain = -1
        best_idx = -1
        for i in range(num_blocks):
            if i in selected_set:
                continue
            gain = (block_bitsets[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~block_bitsets[best_idx]
        
    if time.time() > deadline - 5.0:
        return [all_blocks[i] for i in selected_indices]
    
    # 2. Iterative Redundancy Pruning (The Structural Change)
    # Instead of complex SA to swap blocks, we exploit the high redundancy of covering designs.
    # We repeatedly scan the solution and remove any block that is redundant (i.e., removing it does not uncover any triple).
    # If we remove a block, we scan again from the beginning because removing one block might make another one redundant.
    
    # To make this efficient, we maintain counts of how many times each triple is covered.
    # counts[j] = number of blocks in selected_indices that cover triple j.
    
    def compute_counts(indices):
        counts = np.zeros(num_triples, dtype=np.int32)
        for idx in indices:
            b = all_blocks[idx]
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        return counts

    current_solution = list(selected_indices)
    
    # Initial counts
    counts = compute_counts(current_solution)
    
    # Pruning Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Iterate through current solution
        # We need to be careful with indices if we remove elements.
        # It's easier to iterate by index and mark for removal, or rebuild list.
        # Rebuilding is O(N^2) but N is small.
        
        new_solution = []
        # We need to update counts as we decide to keep/remove.
        # Actually, it's safer to:
        # 1. Tentatively remove block i.
        # 2. Check if any triple drops to 0.
        # 3. If yes, keep it. If no, remove it and update counts permanently.
        
        # Let's do it in-place on a copy of counts?
        # No, let's just check redundancy for each block in the current list.
        
        # Optimization: If a block covers any triple that is covered *only* by it, it cannot be removed.
        # So we only need to check blocks that do NOT have any "unique" triples.
        
        # Let's iterate through the solution.
        # To avoid index shifting issues, we can iterate over a snapshot or just rebuild.
        # Rebuild is cleanest.
        
        kept_indices = []
        # We need to maintain the `counts` array consistent with `kept_indices`?
        # No, `counts` currently reflects `current_solution`.
        # If we decide to keep a block, we do nothing.
        # If we decide to remove a block, we decrement counts for its triples.
        
        # However, if we remove block A, block B might become removable.
        # So we must check B *after* A is removed.
        # This implies a single pass isn't enough if we just skip.
        # But the `while improved` loop handles the multiple passes.
        
        # Inside a single pass:
        # If we remove A, we update counts.
        # When we get to B, we check B against the *updated* counts.
        # This works perfectly for a single forward pass.
        
        # Wait, if we remove A, and B was after A, B sees the updated counts.
        # If B was before A, B already passed.
        # So one pass removes a set of blocks. The next pass (if improved) removes more.
        
        for idx in current_solution:
            # Check if idx is redundant
            # It is redundant if for all triples tri in idx, counts[tri] > 1
            # (Because if we remove idx, counts[tri] becomes counts[tri]-1.
            #  We need counts[tri]-1 > 0 for all tri, i.e., counts[tri] > 1)
            
            is_redundant = True
            b = all_blocks[idx]
            for tri in itertools.combinations(b, t):
                tri_idx = triple_to_idx[tri]
                if counts[tri_idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove idx
                improved = True
                # Update counts
                for tri in itertools.combinations(b, t):
                    tri_idx = triple_to_idx[tri]
                    counts[tri_idx] -= 1
            else:
                # Keep idx
                kept_indices.append(idx)
                
        current_solution = kept_indices
        
        # If we didn't remove anything in this pass, no further removals are possible in subsequent passes.
        # (Because counts only decrease, so if counts[tri] > 1 was false before, it remains false or becomes false?
        #  Actually, if counts[tri] == 1, it stays 1 if we don't remove the unique cover, or becomes 0 if we do.
        #  If a block was NOT redundant because it had a unique triple, it remains NOT redundant because that triple is still unique to it (or now uncovered, which is bad, but we wouldn't remove the unique cover).
        #  Wait, if we remove *other* blocks, could a previously non-redundant block become redundant?
        #  No. Redundancy means "all its triples are covered by others".
        #  If a block B had a unique triple U (count 1), U is covered by B.
        #  If we remove other blocks, U's count stays 1 (since only B covers it).
        #  So B remains non-redundant.
        #  Therefore, if a pass yields no removals, the solution is "locally minimal" w.r.t single-block removal.
        #  We can break.
        
        if not improved:
            break

    # 3. Optional: Randomized Restart / Local Search for better packing?
    # The greedy + prune is very fast. We have lots of time left.
    # Can we do better?
    # Maybe start with a different greedy order? Or randomize greedy?
    # "Randomized Greedy" often finds better local minima.
    
    best_solution = current_solution
    best_size = len(current_solution)
    
    # Try multiple randomized greedy starts
    # Time remaining
    remaining = deadline - time.time() - 5.0
    if remaining > 10:
        # How many restarts?
        # Each greedy+prune is ~0.1s?
        # Let's estimate.
        # Greedy: ~76 * 175k ops? No, 76 * 175k is 13M ops. Python is slow.
        # Let's time one run.
        
        # Actually, the precomputation of bitsets is the slow part (done once).
        # The greedy loop:
        # 76 iterations.
        # Inner loop: 175k blocks.
        # 76 * 175k = 13.3 million iterations.
        # In Python, 13M iterations might take 1-2 seconds.
        # Pruning is very fast.
        
        # So we can afford ~50-100 restarts in the remaining time.
        
        # Strategy:
        # For each restart:
        #   1. Randomize the order of blocks or add noise to gains.
        #   2. Run Greedy.
        #   3. Run Pruning.
        #   4. If better, update best.
        
        # To randomize greedy:
        # Instead of max gain, pick from top K with some probability?
        # Or just shuffle the blocks list? No, that breaks the "max gain" logic.
        # Better: Add small random noise to the gain.
        # gain' = gain + random.uniform(0, noise)
        
        noise_level = 5.0 # Max noise
        
        for _ in range(100): # Max restarts
            if time.time() > deadline - 5.0:
                break
            
            uncovered = (1 << num_triples) - 1
            selected_indices = []
            selected_set = set()
            
            while uncovered and time.time() < deadline - 5.0:
                best_gain = -1
                best_idx = -1
                # To speed up, we could use numpy if we stored bitsets in a numpy array of uint64?
                # But 2024 bits requires 32 uint64s.
                # Let's stick to Python ints for now, it's likely fast enough for 100 restarts if greedy is optimized.
                # Optimization: Precompute list of blocks not in selected_set?
                # Or just check `if i in selected_set`.
                
                for i in range(num_blocks):
                    if i in selected_set:
                        continue
                    gain = (block_bitsets[i] & uncovered).bit_count()
                    # Add noise
                    gain += random.uniform(0, noise_level)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1 or best_gain <= noise_level: # If max gain is just noise, stop?
                    # Actually if best_gain is just noise, real gain is 0.
                    # But we should check real gain.
                    # If best_idx found, check real gain.
                    real_gain = (block_bitsets[best_idx] & uncovered).bit_count()
                    if real_gain == 0:
                        break
                selected_indices.append(best_idx)
                selected_set.add(best_idx)
                uncovered &= ~block_bitsets[best_idx]
            
            # Prune
            if selected_indices:
                counts = compute_counts(selected_indices)
                improved = True
                current_sol = selected_indices
                while improved and time.time() < deadline - 5.0:
                    improved = False
                    kept = []
                    for idx in current_sol:
                        is_red = True
                        b = all_blocks[idx]
                        for tri in itertools.combinations(b, t):
                            if counts[triple_to_idx[tri]] <= 1:
                                is_red = False
                                break
                        if is_red:
                            improved = True
                            for tri in itertools.combinations(b, t):
                                counts[triple_to_idx[tri]] -= 1
                        else:
                            kept.append(idx)
                    current_sol = kept
                
                if len(current_sol) < best_size:
                    best_size = len(current_sol)
                    best_solution = current_sol

    return [all_blocks[i] for i in best_solution]
