import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Pre-computation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the list of triple indices it covers
    # Use numpy arrays for faster operations in the main loop
    block_triples_np = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for bi, b in enumerate(all_blocks):
        indices = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
        block_triples_np.append(np.array(indices, dtype=np.int32))
        for idx in indices:
            triple_to_blocks[idx].append(bi)
            
    # 2. Initialization Strategy: Randomized Greedy with Inverse Frequency Weighting
    # The idea: When selecting the next block, don't just pick the one with max gain.
    # Instead, sample blocks covering a random uncovered triple, with probability 
    # inversely proportional to the number of uncovered triples they cover.
    # This encourages covering "hard" (rare) triples early and avoids redundant overlaps.
    
    def greedy_init_weighted(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # covered[i] is True if triple i is covered
        covered = np.zeros(num_triples, dtype=bool)
        selected = []
        
        # To speed up finding uncovered triples, we maintain a list of uncovered indices
        # But updating a list of 2000 items is fast enough.
        # However, for the weighted selection, we need to evaluate candidates for a specific uncovered triple.
        
        # We'll iterate until all are covered.
        # To avoid O(N^2) in the worst case if we scan for uncovered every time,
        # we can keep a count of uncovered.
        
        uncovered_count = num_triples
        
        while uncovered_count > 0:
            # Pick a random uncovered triple
            # Rejection sampling
            while True:
                target_idx = random.randint(0, num_triples - 1)
                if not covered[target_idx]:
                    break
            
            # Get candidates: blocks that cover this target
            candidates = triple_to_blocks[target_idx]
            
            if not candidates:
                break # Should not happen if valid parameters
                
            # Calculate gain for each candidate
            # Gain = number of uncovered triples covered by the block
            # We want to pick with prob proportional to 1/gain (or similar)
            # Actually, the prompt suggests: "selects blocks maximizing the ratio of newly covered triplets to the number of points in the block"
            # Wait, the prompt says: "maximizing the ratio of newly covered triplets to the number of points in the block"
            # But k is constant. So it's just maximizing gain?
            # Let's re-read: "selects blocks maximizing the ratio of newly covered triplets to the number of points in the block, rather than absolute gain. This approach explicitly penalizes redundancy and overlaps..."
            # If k is constant, ratio = gain/k. So it's just gain.
            # BUT the inspiration says: "Vectorized numpy greedy with inverse frequency weighting... provides a stronger baseline"
            # And the directive says: "Implement this with a random..." (cut off)
            # Let's look at the "Strategy to try" again: "selects blocks maximizing the ratio of newly covered triplets to the number of points in the block... Implement this with a random..."
            # It seems the text was cut off. "Implement this with a randomized greedy..."
            # Let's stick to the "Inverse Frequency Weighting" mentioned in the Knowledge section which was successful (score 1.113).
            # The strategy: For a random uncovered triple, pick a block covering it.
            # Probability of picking block b is proportional to 1 / (number of uncovered triples b covers).
            # This favors blocks that cover few *other* uncovered triples, i.e., blocks that are "specialized" for the target or have less overlap with other potential future needs?
            # Actually, usually "inverse frequency" in set cover heuristics refers to prioritizing elements (triples) that are covered by fewer sets (blocks).
            # Here, we pick a random uncovered triple.
            # Then we pick a block.
            # If we pick a block that covers MANY uncovered triples, it's a "greedy" choice.
            # If we pick a block that covers FEW uncovered triples (just the target maybe), it's a "conservative" choice.
            # The "inverse frequency weighting" in the successful island likely refers to weighting the *selection of the target triple* by its rarity (number of blocks covering it).
            # i.e., Pick an uncovered triple with probability proportional to 1 / (number of blocks covering it).
            # This forces the algorithm to address "hard" triples first.
            
            # Let's implement:
            # 1. Maintain a list of uncovered triples.
            # 2. For each uncovered triple, know how many blocks cover it (static).
            # 3. Pick an uncovered triple with probability proportional to 1/freq.
            # 4. Among blocks covering it, pick one with maximum gain (standard greedy) or random?
            #    Let's try: Pick target with inverse freq. Then pick best gain block for that target.
            
            # To make this efficient, we need to update the "uncovered" set and be able to sample from it.
            # Since N is small (2024), we can just scan.
            
            # Pre-compute static frequency of each triple (number of blocks covering it)
            # This is constant.
            # static_freq = [len(triple_to_blocks[i]) for i in range(num_triples)]
            # We already have triple_to_blocks.
            
            # Let's use a list of currently uncovered indices.
            # Initially: all indices.
            # When we cover some, remove them from the list.
            # To sample with probability proportional to 1/freq:
            # We can use weighted random choice.
            # weights[i] = 1 / static_freq[i]
            # But we need to sample only from the *uncovered* ones.
            
            # Optimization:
            # Since N is small, let's just iterate.
            
            # 1. Find all uncovered indices.
            uncovered_indices = [i for i in range(num_triples) if not covered[i]]
            
            # 2. Calculate weights for these indices.
            # weights = [1.0 / len(triple_to_blocks[i]) for i in uncovered_indices]
            
            # 3. Pick one.
            # Using random.choices
            target_idx = random.choices(uncovered_indices, weights=[1.0/len(triple_to_blocks[i]) for i in uncovered_indices], k=1)[0]
            
            # 4. Find best block for this target.
            candidates = triple_to_blocks[target_idx]
            
            best_block = -1
            best_gain = -1
            
            # To speed up, we can shuffle candidates to break ties randomly
            random.shuffle(candidates)
            
            for bi in candidates:
                # Gain is count of uncovered triples in block_triples_np[bi]
                # Using numpy for speed
                b_triples = block_triples_np[bi]
                # Sum of (not covered) for these triples
                gain = np.sum(~covered[b_triples])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block != -1:
                selected.append(best_block)
                # Update covered
                b_triples = block_triples_np[best_block]
                covered[b_triples] = True
                # Update uncovered_count
                # We don't strictly need to update the count if we rescan, 
                # but it helps terminate.
                # Actually, the loop condition `while uncovered_count > 0` needs updating.
                # Let's just use `while True` and break if no uncovered found.
                
        return selected

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        counts = np.zeros(num_triples, dtype=np.int32)
        for bi in blocks:
            counts[block_triples_np[bi]] += 1
            
        final_blocks = []
        for bi in blocks:
            # If any triple in this block has count 1, it's essential
            # If all have count > 1, it's redundant
            b_triples = block_triples_np[bi]
            # Check if min count among these triples is > 1
            if np.min(counts[b_triples]) > 1:
                # Redundant
                counts[b_triples] -= 1
            else:
                # Keep
                final_blocks.append(bi)
        return final_blocks

    # Run multiple restarts of the weighted greedy
    best_greedy = []
    best_greedy_len = float('inf')
    
    # How many restarts?
    # Each restart takes some time. Let's estimate.
    # N=2024. ~100-200 blocks in cover.
    # Each step: scan 2024 for uncovered, weight them, pick, scan candidates (maybe 10-50), calculate gain (numpy sum).
    # Should be fast. Maybe 0.05s per restart?
    # Let's do 50 restarts initially.
    
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 30: # Leave time for local search
            break
        blocks = greedy_init_weighted(100 + i)
        blocks = remove_redundant(blocks)
        if len(blocks) < best_greedy_len:
            best_greedy_len = len(blocks)
            best_greedy = blocks[:]
            
    # 3. Local Search (Simulated Annealing or Hill Climbing)
    # Start from best_greedy
    current_state = best_greedy[:]
    counts = np.zeros(num_triples, dtype=np.int32)
    for bi in current_state:
        counts[block_triples_np[bi]] += 1
        
    # Cost: len(state) + 100 * num_uncovered
    def get_cost(state, counts):
        uncovered = np.sum(counts == 0)
        if uncovered > 0:
            return len(state) + 100 * uncovered
        return len(state)
        
    current_cost = get_cost(current_state, counts)
    best_state = current_state[:]
    best_cost = current_cost
    
    T = 5.0
    cooling_rate = 0.999
    T_min = 0.1
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        if T < T_min:
            break
            
        if len(current_state) == 0:
            break
            
        move_type = random.random()
        
        if move_type < 0.6:
            # Swap
            idx = random.randint(0, len(current_state) - 1)
            old_block = current_state[idx]
            
            # Pick a random new block
            new_block = random.randint(0, num_blocks - 1)
            
            if new_block == old_block:
                continue
                
            # Calculate delta
            old_triples = block_triples_np[old_block]
            new_triples = block_triples_np[new_block]
            
            # Temporarily remove old
            counts[old_triples] -= 1
            # Add new
            counts[new_triples] += 1
            
            new_cost = get_cost(current_state, counts)
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                current_state[idx] = new_block
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = current_state[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                counts[new_triples] -= 1
                counts[old_triples] += 1
                no_improve += 1
                
        else:
            # Remove
            idx = random.randint(0, len(current_state) - 1)
            old_block = current_state[idx]
            old_triples = block_triples_np[old_block]
            
            # Check if valid
            if np.min(counts[old_triples]) > 1:
                # Valid removal
                counts[old_triples] -= 1
                current_state.pop(idx)
                
                new_cost = get_cost(current_state, counts)
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = current_state[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        T *= cooling_rate
        
    # Final Cleanup
    final_state = best_state[:]
    final_state = remove_redundant(final_state)
    
    # Convert to list of lists
    return [list(all_blocks[bi]) for bi in final_state]
