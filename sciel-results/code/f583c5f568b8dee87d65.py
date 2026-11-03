import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0 # Safety margin
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute blocks and their triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(all_blocks):
        covered = [triple_to_idx[tup] for tup in itertools.combinations(block, t)]
        block_triples.append(covered)
        for tr_idx in covered:
            triple_to_blocks[tr_idx].append(i)
            
    best_blocks = None
    best_count = float('inf')
    
    def get_covering_blocks(block_indices):
        return [all_blocks[i] for i in block_indices]

    def is_valid_covering(block_indices):
        # Check if all triples are covered
        # This is expensive, so we rely on the construction/repair process
        # But for safety in final check, we might need it.
        # However, if we maintain counts, we know validity.
        pass

    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = set(range(num_triples))
        selected_blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to anchor the search
            target_triple_idx = random.choice(list(uncovered))
            
            # Candidates are blocks covering this triple
            candidates = triple_to_blocks[target_triple_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            # To speed up, we can sample candidates if the list is huge, 
            # but 6000 is small enough to scan fully.
            for bi in candidates:
                # Count how many triples in this block are in 'uncovered'
                gain = 0
                for tr_idx in block_triples[bi]:
                    if tr_idx in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we can't beat best? 
                            # Max possible gain is 35.
                            if gain == 35:
                                break
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    if best_gain == 35:
                        break
            
            if best_block_idx != -1:
                selected_blocks.append(best_block_idx)
                # Update uncovered
                for tr_idx in block_triples[best_block_idx]:
                    uncovered.discard(tr_idx)
            else:
                # Should not happen
                break
                
        return selected_blocks

    def peel(block_indices):
        # Remove blocks that don't uniquely cover any triple
        # Returns new list of block indices
        if not block_indices:
            return block_indices
            
        # Count coverage
        counts = [0] * num_triples
        for bi in block_indices:
            for tr_idx in block_triples[bi]:
                counts[tr_idx] += 1
                
        new_blocks = []
        for bi in block_indices:
            removable = True
            for tr_idx in block_triples[bi]:
                if counts[tr_idx] == 1:
                    removable = False
                    break
            if not removable:
                new_blocks.append(bi)
            else:
                for tr_idx in block_triples[bi]:
                    counts[tr_idx] -= 1
                    
        return new_blocks

    def local_search(block_indices, time_limit):
        # block_indices: list of block indices
        # Try to improve by swapping/removing
        current_blocks = list(block_indices)
        current_size = len(current_blocks)
        
        # Build a set for O(1) lookup
        block_set = set(current_blocks)
        
        # Precompute coverage counts for current set
        counts = [0] * num_triples
        for bi in current_blocks:
            for tr_idx in block_triples[bi]:
                counts[tr_idx] += 1
                
        # We will try random removals and repairs
        # This is a heuristic search
        
        attempts = 0
        while time.time() < time_limit:
            if not current_blocks:
                break
                
            # Pick a random block to remove
            idx = random.randint(0, len(current_blocks) - 1)
            bi_to_remove = current_blocks[idx]
            
            # Identify triples that would become uncovered
            # These are triples in bi_to_remove with count == 1
            uncovered_triples = []
            for tr_idx in block_triples[bi_to_remove]:
                if counts[tr_idx] == 1:
                    uncovered_triples.append(tr_idx)
                    
            if not uncovered_triples:
                # Safe to remove
                current_blocks.pop(idx)
                block_set.discard(bi_to_remove)
                for tr_idx in block_triples[bi_to_remove]:
                    counts[tr_idx] -= 1
                current_size -= 1
                # Try to peel again immediately
                current_blocks = peel(current_blocks)
                # Rebuild counts and set? 
                # Peeling might remove more.
                # If we just peeled, counts are stale. 
                # Let's rebuild counts if we removed anything.
                # For efficiency in local search, we might skip full peel 
                # and just do the swap logic.
                # But peel is effective.
                # Let's rebuild counts if size changed significantly.
                if len(current_blocks) != current_size:
                     # Rebuild
                     counts = [0] * num_triples
                     for bi in current_blocks:
                         for tr_idx in block_triples[bi]:
                             counts[tr_idx] += 1
                     current_size = len(current_blocks)
                     block_set = set(current_blocks)
                continue
            
            # Need to repair: find a block that covers the most uncovered_triples
            best_block = -1
            best_cover = 0
            
            # Candidates: blocks that cover at least one of the uncovered_triples
            # We can union the candidate lists for the uncovered triples
            candidates = set()
            for tr_idx in uncovered_triples:
                candidates.update(triple_to_blocks[tr_idx])
            
            # Remove current blocks from candidates
            candidates -= block_set
            
            for cb in candidates:
                cover = 0
                for tr_idx in uncovered_triples:
                    # Check if cb covers tr_idx
                    # We don't have a direct map from (block, triple) -> bool
                    # We have block_triples[cb] which is a list.
                    # To check if tr_idx is in block_triples[cb], we need a set or binary search.
                    # Let's precompute block_triples_sets for faster lookup?
                    # Or just iterate?
                    # Since uncovered_triples is small (usually 1-3), 
                    # we can check if any of them are in block_triples[cb].
                    # Better: Precompute a set for each block's triples?
                    # Memory: 346k * 35 ints. ~12M ints. ~100MB. OK.
                    pass
            
            # Optimization: Precompute block_triple_sets
            # Let's assume we have a global `block_triple_sets`
            
            # For now, let's just pick a random candidate to keep it simple and fast
            # or pick the first one.
            if candidates:
                # Heuristic: Pick the block that covers the most uncovered triples
                # To do this efficiently, we need the set lookup.
                # Let's use a simple heuristic: Pick a random candidate.
                # This is a random walk.
                new_block = random.choice(list(candidates))
                
                # Add new block
                current_blocks.append(new_block)
                block_set.add(new_block)
                for tr_idx in block_triples[new_block]:
                    counts[tr_idx] += 1
                current_size += 1
                
                # Now we have a valid covering of size current_size.
                # We removed 1 and added 1. Size is same as before removal.
                # But we might be able to remove another block now?
                # Let's try to peel.
                # Peeling is expensive.
                # Let's just accept this move if it leads to a state where 
                # we can potentially remove a block later.
                # Actually, this "remove and repair" strategy often keeps size constant.
                # To reduce size, we need to find a case where we remove 1 and add 0 (impossible if uncovered)
                # or remove 1 and add 1, but the new block allows removing another block.
                
                # Let's try a different local search:
                # Swap a block for another block that covers the same "unique" triples?
                # No, that doesn't reduce size.
                
                # The most effective way to reduce size is "Peeling".
                # So, after any modification, we should peel.
                
                # Let's restart the loop to try another removal.
                # This is a bit slow if we rebuild counts every time.
                # But `uncovered_triples` is small.
            
            # If we couldn't repair, we just didn't change anything?
            # The code above adds the block.
            # We should verify validity.
            
            # Let's implement a proper "Swap" local search.
            # Pick a block B1.
            # Try to replace it with B2 such that B2 covers all triples that were uniquely covered by B1.
            # If such B2 exists, and B2 allows removing another block B3, we gain.
            
            # Given time constraints, let's stick to:
            # 1. Greedy
            # 2. Peel
            # 3. Random Restart Greedy + Peel
            # 4. Simple Local Search: Randomly remove a block, if valid, keep. If invalid, find best repair block, add it. If size didn't increase, keep.
            
            # The "keep if size didn't increase" allows exploration.
            # To actually decrease size, we rely on the "valid removal" case (uncovered_triples empty).
            # This happens if a block is redundant.
            # So local search should aggressively look for redundant blocks.
            
            # Let's just do multiple restarts of Greedy+Peel.
            # It's simpler and often sufficient.
            
            break # Exit local search for now, rely on restarts
            
        return current_blocks

    # Main Loop
    attempts = 0
    # Precompute sets for faster lookup in local search if we implement it
    # For now, just greedy + peel
    
    while time.time() - start_time < timeout:
        seed = attempts
        blocks_idx = greedy_cover(seed)
        blocks_idx = peel(blocks_idx)
        
        if len(blocks_idx) < best_count:
            best_count = len(blocks_idx)
            best_blocks = blocks_idx
            
        attempts += 1
        
        if time.time() - start_time > timeout - 5.0:
            break

    if best_blocks is None:
        blocks_idx = greedy_cover(0)
        best_blocks = peel(blocks_idx)
        
    return [list(all_blocks[i]) for i in best_blocks]
