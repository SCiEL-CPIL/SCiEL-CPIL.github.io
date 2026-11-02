import itertools
import time
import random
import sys

def construct(v, k, t):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Pre-compute triples
    # Map triple (tuple) to index
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute for each block (tuple of k points) the list of triple indices it contains?
    # No, we generate blocks on the fly based on a seed triple.
    # But we need to quickly get the triple indices for a candidate block.
    # We can compute this on the fly: for a block B, generate its t-subsets and map to idx.
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to run one greedy construction
    def run_greedy():
        coverage_count = [0] * num_triples
        uncovered_set = set(range(num_triples))
        blocks = []
        
        while uncovered_set:
            # Pick a random uncovered triple to seed the block
            seed_triple_idx = random.choice(list(uncovered_set))
            seed_triple = all_triples[seed_triple_idx]
            
            # Generate candidate blocks containing seed_triple
            # Remaining points: v - t
            # Need to choose k - t more points
            remaining_points = [p for p in range(v) if p not in seed_triple]
            
            max_gain = -1
            best_candidate = None
            candidates = []
            
            # To speed up, we can pre-compute the triple indices for the seed part?
            # No, just iterate.
            
            # Optimization: We only care about triples that are currently uncovered.
            # For a candidate block B, gain = count of triples in B that are in uncovered_set.
            
            # Number of candidates: C(21, 4) = 5985
            # For each, we check 35 triples.
            
            for comb in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(seed_triple + comb))
                
                # Calculate gain
                gain = 0
                for t_sub in itertools.combinations(block, t):
                    idx = triple_to_idx[t_sub]
                    if idx in uncovered_set:
                        gain += 1
                        # Early exit if gain exceeds current max?
                        # If gain > max_gain, we can stop checking other triples?
                        # No, we need exact gain for tie-breaking or if we want to know all.
                        # But if gain is already > max_gain, we can break early in the triple loop?
                        # Yes.
                        if gain > max_gain:
                            # We still need to finish the loop to get the block? No, we have the block.
                            # But we might miss that another block has even higher gain.
                            # We just need to know if this one is better.
                            # So we can break the inner triple loop if gain > max_gain.
                            pass # Can't break easily without restructuring
                
                # Actually, let's just compute full gain. 35 is small.
                # But we can optimize:
                # If gain <= max_gain, skip.
                # We can check triples one by one and break if gain > max_gain.
                
                # Let's refine the gain calculation loop
                gain = 0
                for t_sub in itertools.combinations(block, t):
                    idx = triple_to_idx[t_sub]
                    if idx in uncovered_set:
                        gain += 1
                    # Optimization: if gain > max_gain, we can't break yet because we don't know if it's the max,
                    # but we know it's better than previous best.
                    # We can just let it run. 35 iterations is trivial.
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidate = block
                    # Randomization: if we want to keep all max gain candidates to pick randomly later,
                    # we'd need to store them. Storing 5000 tuples is fine.
                    # But picking the first max is also a valid greedy.
                    # To improve diversity, let's collect all max gain candidates.
                    # However, that requires storing them.
                    # Let's just pick the first one found with max gain for now, 
                    # but randomize the order of candidates?
                    # Randomizing 5985 candidates is cheap.
            
            if best_candidate is None:
                # Should not happen if uncovered_set is not empty
                break
                
            # Add block
            blocks.append(best_candidate)
            
            # Update coverage
            for t_sub in itertools.combinations(best_candidate, t):
                idx = triple_to_idx[t_sub]
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered_set.remove(idx)
                    
        return blocks

    # Helper to reduce blocks
    def reduce_blocks(blocks):
        # Make a copy
        blocks = list(blocks)
        
        # Pre-compute triple indices for each block?
        # Or just use the global map.
        
        coverage_count = [0] * num_triples
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                idx = triple_to_idx[t_sub]
                coverage_count[idx] += 1
        
        improved = True
        while improved:
            improved = False
            # Randomize order for removal
            random.shuffle(blocks)
            i = 0
            while i < len(blocks):
                b = blocks[i]
                # Check if removing b leaves any triple with count 0
                can_remove = True
                for t_sub in itertools.combinations(b, t):
                    idx = triple_to_idx[t_sub]
                    if coverage_count[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for t_sub in itertools.combinations(b, t):
                        idx = triple_to_idx[t_sub]
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    # Do not increment i, check the next block at same index
                else:
                    i += 1
                    
        return blocks

    # Main loop
    while time.time() < deadline:
        # Run greedy
        raw_blocks = run_greedy()
        
        # Reduce
        reduced_blocks = reduce_blocks(raw_blocks)
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        # If we reach the known best (76), we might stop or try to break it?
        # 76 is the current record. Breaking it is unlikely but possible.
        # We will just keep searching for smaller.
        
        # Check time
        if time.time() > deadline:
            break

    if best_blocks is None:
        # Fallback: return the first greedy result if loop didn't run?
        # But loop runs at least once.
        return []
        
    return [list(b) for b in best_blocks]
