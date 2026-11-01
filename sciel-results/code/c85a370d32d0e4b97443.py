import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # 1. Precomputation
    # Map triples to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    triple_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_triples = len(all_tsubs)
    
    # Generate all blocks and their covered triple indices
    all_blocks = []
    block_triples_idx = []
    block_index_map = {}
    
    for block in itertools.combinations(range(v), k):
        block_tuple = tuple(block)
        idx = len(all_blocks)
        all_blocks.append(block_tuple)
        block_index_map[block_tuple] = idx
        
        tris = []
        for ts in itertools.combinations(block, t):
            tris.append(triple_to_idx[ts])
        block_triples_idx.append(tris)
        
        # Check time
        if time.time() - start_time > 5:
            break # Should not happen, but safety
    
    num_blocks_total = len(all_blocks)
    
    # 2. Initial Solution: Greedy Set Cover
    coverage = np.zeros(num_triples, dtype=np.int32)
    current_blocks = []
    uncovered = np.ones(num_triples, dtype=bool)
    
    # To speed up greedy, we can just iterate. 
    # Since num_blocks_total is ~346k, and we need ~76 blocks.
    # Naive greedy: O(NumBlocks * NumTriples) per step? No.
    # We want the block that maximizes intersection with uncovered.
    # We can precompute nothing more, just loop.
    # 76 steps * 346k blocks * 35 triples = ~900M ops. Too slow.
    
    # Better Initial: Use the randomized greedy from the original code, but faster?
    # Or use a simple heuristic:
    # Pick a random uncovered triple, pick a random block containing it.
    # This is O(1) per step if we have triple_to_blocks.
    # We don't have triple_to_blocks precomputed.
    
    # Let's build triple_to_blocks now.
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx, tris in enumerate(block_triples_idx):
        for t_idx in tris:
            triple_to_blocks[t_idx].append(b_idx)
    
    # Greedy Initialization using triple_to_blocks
    random.seed(42)
    while np.any(uncovered):
        # Pick a random uncovered triple
        unc_idx = np.where(uncovered)[0]
        t_idx = random.choice(unc_idx)
        
        # Pick a random block covering this triple
        # To be better, pick the block that covers the most uncovered triples among those covering t_idx
        # This is a "Greedy" step.
        candidates = triple_to_blocks[t_idx]
        
        best_b_idx = -1
        max_gain = -1
        # Sample if candidates is large? 5985 is manageable.
        for b_idx in candidates:
            gain = 0
            for tt in block_triples_idx[b_idx]:
                if uncovered[tt]:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_b_idx = b_idx
                
        # Add best block
        current_blocks.append(best_b_idx)
        for tt in block_triples_idx[best_b_idx]:
            if uncovered[tt]:
                coverage[tt] += 1
                uncovered[tt] = False
            else:
                coverage[tt] += 1
        
        if time.time() - start_time > time_limit * 0.2:
            break
    
    # If time out, force finish
    if np.any(uncovered):
        while np.any(uncovered):
            unc_idx = np.where(uncovered)[0]
            t_idx = unc_idx[0]
            candidates = triple_to_blocks[t_idx]
            # Just take first
            b_idx = candidates[0]
            current_blocks.append(b_idx)
            for tt in block_triples_idx[b_idx]:
                coverage[tt] += 1
                if coverage[tt] == 1:
                    uncovered[tt] = False

    # 3. Simulated Annealing
    T = 10.0
    T_min = 0.1
    alpha = 0.995
    M = 100 # Penalty for uncovered
    
    best_blocks = list(current_blocks)
    best_cost = len(current_blocks) # + M * sum(uncovered)
    
    # We track cost = len(blocks) + M * num_uncovered
    # Since we start valid, num_uncovered = 0.
    
    while time.time() - start_time < time_limit:
        # Pick move
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove Block
            if not current_blocks:
                continue
            b_idx_pos = random.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[b_idx_pos]
            
            # Check coverage
            min_cov = np.min([coverage[tt] for tt in block_triples_idx[b_idx]])
            
            if min_cov > 1:
                # Safe removal
                for tt in block_triples_idx[b_idx]:
                    coverage[tt] -= 1
                current_blocks.pop(b_idx_pos)
                # Always accept improvement
            else:
                # Calculate delta U
                delta_u = 0
                for tt in block_triples_idx[b_idx]:
                    if coverage[tt] == 1:
                        delta_u += 1
                
                delta_cost = M * delta_u - 1
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Apply
                    for tt in block_triples_idx[b_idx]:
                        coverage[tt] -= 1
                    current_blocks.pop(b_idx_pos)
                    if np.any(coverage == 0):
                        # Update uncovered mask? 
                        # We just use coverage array. 
                        pass

        elif move_type < 0.8:
            # Swap Point
            if not current_blocks:
                continue
            b_idx_pos = random.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[b_idx_pos]
            block = all_blocks[b_idx]
            
            old_point = random.choice(list(block))
            new_point = random.randint(0, v - 1)
            
            if new_point == old_point or new_point in block:
                continue
            
            new_block_list = list(block)
            new_block_list.remove(old_point)
            new_block_list.append(new_point)
            new_block_tuple = tuple(sorted(new_block_list))
            
            new_b_idx = block_index_map.get(new_block_tuple)
            if new_b_idx is None:
                continue
            
            # Calculate delta
            old_tris = block_triples_idx[b_idx]
            new_tris = block_triples_idx[new_b_idx]
            
            # We need to know which triples lose/gain coverage
            # This is complex with numpy.
            # Simple check:
            # Temporarily remove old, add new?
            # Or calculate delta_u directly.
            
            # Triples in old but not new:
            old_set = set(old_tris)
            new_set = set(new_tris)
            
            lost = old_set - new_set
            gained = new_set - old_set
            
            delta_u = 0
            for tt in lost:
                if coverage[tt] == 1:
                    delta_u += 1
            # Gained triples reduce uncovered count if they were uncovered
            # But coverage[tt] > 0 for all currently covered.
            # If a triple was uncovered (coverage 0), and we gain it, delta_u decreases.
            # But we started with a valid solution? 
            # If we are in an invalid state (during SA), coverage can be 0.
            
            # Correct Delta U calculation:
            # U_old = count(coverage == 0)
            # U_new = count(coverage' == 0)
            # This is hard to compute incrementally without applying.
            
            # Let's just apply and check?
            # No, that breaks SA logic.
            
            # Approximation:
            # If we are valid (no uncovered), delta_u is just number of lost triples with coverage 1.
            # If we are invalid, it's complex.
            
            # Let's assume we stay mostly valid.
            # If delta_u > 0, it's a worsening move.
            
            if delta_u == 0:
                # Apply swap
                for tt in lost:
                    coverage[tt] -= 1
                for tt in gained:
                    coverage[tt] += 1
                current_blocks[b_idx_pos] = new_b_idx
            else:
                delta_cost = M * delta_u
                if random.random() < np.exp(-delta_cost / T):
                    for tt in lost:
                        coverage[tt] -= 1
                    for tt in gained:
                        coverage[tt] += 1
                    current_blocks[b_idx_pos] = new_b_idx

        else:
            # Add Block
            # Pick random block
            # Better: Pick block covering a random uncovered triple
            unc_idx = np.where(coverage == 0)[0]
            if len(unc_idx) > 0:
                t_idx = random.choice(unc_idx)
                candidates = triple_to_blocks[t_idx]
                if candidates:
                    b_idx = random.choice(candidates)
                    # Check if already in current_blocks? 
                    # If yes, adding duplicate is useless.
                    # We allow duplicates in SA state? No, blocks must be distinct.
                    # Check if b_idx is in current_blocks.
                    if b_idx not in current_blocks:
                        # Calculate delta_u
                        # It covers t_idx, so at least 1 uncovered is resolved.
                        # Delta_u = - (number of uncovered triples in this block)
                        # Delta_cost = M * delta_u + 1
                        gain = 0
                        for tt in block_triples_idx[b_idx]:
                            if coverage[tt] == 0:
                                gain += 1
                        
                        delta_cost = M * (-gain) + 1
                        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                            for tt in block_triples_idx[b_idx]:
                                coverage[tt] += 1
                            current_blocks.append(b_idx)
            
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Track best
        num_unc = np.sum(coverage == 0)
        current_cost = len(current_blocks) + M * num_unc
        if current_cost < best_cost:
            best_cost = current_cost
            best_blocks = list(current_blocks)

    # Reconstruct blocks from best_blocks
    # Ensure validity
    final_coverage = np.zeros(num_triples, dtype=np.int32)
    for b_idx in best_blocks:
        for tt in block_triples_idx[b_idx]:
            final_coverage[tt] += 1
            
    # If not valid, repair
    if np.any(final_coverage == 0):
        # Greedy repair
        uncovered = np.where(final_coverage == 0)[0]
        while len(uncovered) > 0:
            t_idx = uncovered[0]
            candidates = triple_to_blocks[t_idx]
            # Find best candidate
            best_b = -1
            max_gain = -1
            for b_idx in candidates:
                if b_idx in best_blocks:
                    continue
                gain = 0
                for tt in block_triples_idx[b_idx]:
                    if final_coverage[tt] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_b = b_idx
            if best_b != -1:
                best_blocks.append(best_b)
                for tt in block_triples_idx[best_b]:
                    final_coverage[tt] += 1
            else:
                # Should not happen
                break
            uncovered = np.where(final_coverage == 0)[0]
            
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
