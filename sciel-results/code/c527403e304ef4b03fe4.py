"""Optimized Simulated Annealing for covering design C(24,7,3)."""
import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using optimized simulated annealing.
    """
    random.seed(42)
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Helper to count coverages
    def count_coverages(blocks):
        counts = [0] * n_t_subsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                counts[idx] += 1
        return counts
    
    # Helper to get uncovered count
    def uncovered_count(counts):
        return sum(1 for c in counts if c == 0)
        
    # Initial Covering: Greedy Set Cover
    # We will generate blocks greedily. To speed up, we pick an uncovered triple,
    # and try to complete it to a block that covers many other uncovered triples.
    
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Precompute combinations of sizes for fast access? 
    # For v=24, k=7, t=3, C(24,3)=2024.
    
    while uncovered:
        # Pick the first uncovered triple
        target = next(iter(uncovered))
        block = list(target)
        
        # Greedily add points to maximize coverage of *uncovered* triples
        while len(block) < k:
            best_x = -1
            best_gain = -1
            current_set = set(block)
            
            for x in range(v):
                if x in current_set:
                    continue
                # Calculate gain: how many new triples are covered
                # A triple (a,b,x) is covered if {a,b} subset of block and x not in block
                # We need to check triples that include x and 2 elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                break
            block.append(best_x)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
            
    # Initial state
    current_blocks = [b[:] for b in blocks]
    counts = count_coverages(current_blocks)
    current_uncovered = uncovered_count(counts)
    
    best_blocks = [b[:] for b in current_blocks]
    best_uncovered = current_uncovered
    best_len = len(current_blocks)
    
    # Simulated Annealing
    start_time = time.time()
    max_time = 115.0
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    n_blocks = len(current_blocks)
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        
        # Move: Pick a random block, replace one point with another
        if n_blocks > 0:
            bi = random.randint(0, n_blocks - 1)
            block = current_blocks[bi]
            pi = random.randint(0, k - 1)
            old_p = block[pi]
            
            # Pick new point
            available = [x for x in range(v) if x not in block]
            if not available:
                continue
            new_p = random.choice(available)
            
            # Calculate Delta
            # Triples lost: contain old_p, not new_p
            # Triples gained: contain new_p, not old_p
            
            delta = 0
            
            # Generate triples in current block containing old_p
            # and check if they are "critical" (count == 1)
            # and if they don't contain new_p
            
            # Optimization: Precompute the list of triples for the block?
            # It's small (C(7,3)=35).
            
            old_triples = [ts for ts in itertools.combinations(block, t) if old_p in ts]
            new_block = block[:]
            new_block[pi] = new_p
            new_block.sort()
            new_triples = [ts for ts in itertools.combinations(new_block, t) if new_p in ts]
            
            # Check losses
            for ts in old_triples:
                if new_p not in ts:
                    idx = t_subset_to_idx[ts]
                    if counts[idx] == 1:
                        delta += 1
                        
            # Check gains
            for ts in new_triples:
                if old_p not in ts:
                    idx = t_subset_to_idx[ts]
                    if counts[idx] == 0:
                        delta -= 1
            
            # Accept/Reject
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Apply move
                if delta > 0:
                    # Update counts for losses
                    for ts in old_triples:
                        if new_p not in ts:
                            idx = t_subset_to_idx[ts]
                            counts[idx] -= 1
                if delta < 0:
                    # Update counts for gains
                    for ts in new_triples:
                        if old_p not in ts:
                            idx = t_subset_to_idx[ts]
                            counts[idx] += 1
                
                # If delta was 0, we still need to update counts if the set of triples changed
                # Actually, if delta is 0, it means no critical changes, but counts might still change?
                # No, delta is defined by changes in uncovered status.
                # However, the counts themselves change for ALL affected triples, not just critical ones.
                # We must update counts for all affected triples to maintain consistency.
                
                # Let's just recalculate counts for the affected indices to be safe and simple?
                # Or update them properly.
                
                # Proper update:
                # Decrement counts for old_triples that are not in new_triples (or rather, not covered by new block in same way)
                # Actually, the block changed. So:
                # All triples in old_block that are NOT in new_block lose 1 count.
                # All triples in new_block that are NOT in old_block gain 1 count.
                
                old_set = set(old_triples)
                new_set = set(new_triples)
                
                # Note: This is not strictly correct because other triples in the block (not containing old_p or new_p) remain the same.
                # But triples containing old_p might also contain other elements.
                # The set of triples in a block is C(block, 3).
                # When we replace old_p with new_p:
                # Triples containing old_p are removed.
                # Triples containing new_p are added.
                # Triples not containing either are unchanged.
                
                # So:
                for ts in old_triples:
                    idx = t_subset_to_idx[ts]
                    counts[idx] -= 1
                for ts in new_triples:
                    idx = t_subset_to_idx[ts]
                    counts[idx] += 1
                    
                current_blocks[bi] = new_block
                current_uncovered += delta
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = [b[:] for b in current_blocks]
                    best_len = len(current_blocks)
        
        # Move: Try to remove a block
        if n_blocks > 1:
             # Pick a random block
             bi = random.randint(0, n_blocks - 1)
             block = current_blocks[bi]
             
             # Check if redundant
             redundant = True
             for ts in itertools.combinations(block, t):
                 idx = t_subset_to_idx[ts]
                 if counts[idx] <= 1:
                     redundant = False
                     break
             
             if redundant:
                 # Remove block
                 for ts in itertools.combinations(block, t):
                     idx = t_subset_to_idx[ts]
                     counts[idx] -= 1
                 
                 current_blocks.pop(bi)
                 n_blocks -= 1
                 current_uncovered = uncovered_count(counts)
                 
                 if current_uncovered < best_uncovered:
                     best_uncovered = current_uncovered
                     best_blocks = [b[:] for b in current_blocks]
                     best_len = len(current_blocks)
        
        T *= cooling_rate
        
        # Periodic restart or local search?
        # If we are stuck, maybe try a different strategy.
        
    # Final Local Search: Delete redundant blocks
    final_blocks = [b[:] for b in best_blocks]
    changed = True
    while changed:
        changed = False
        counts = count_coverages(final_blocks)
        for i in range(len(final_blocks) - 1, -1, -1):
            block = final_blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                final_blocks.pop(i)
                changed = True
                break
                
    # Verify
    counts = count_coverages(final_blocks)
    if uncovered_count(counts) > 0:
        # Fallback
        final_blocks = [b[:] for b in blocks]
        
    return final_blocks
