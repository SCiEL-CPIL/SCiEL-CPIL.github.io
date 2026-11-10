import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_id = {s: i for i, s in enumerate(all_tsubsets)}
    
    # Precompute for each t-subset, the candidate blocks (k-subsets containing it)
    # For v=19, k=7, t=4: C(15,3)=455 candidates per subset
    subset_candidates = []
    for sub in all_tsubsets:
        sub_set = set(sub)
        remaining = [x for x in range(v) if x not in sub_set]
        num_add = k - t
        candidates = []
        for add_combo in itertools.combinations(remaining, num_add):
            block = tuple(sorted(sub + add_combo))
            candidates.append(block)
        subset_candidates.append(candidates)
    
    # Precompute for each possible block, the list of t-subset IDs it covers
    # We'll build this lazily using a cache
    block_to_tsubsets = {}
    
    def get_block_tsubsets(block):
        """Get the list of t-subset IDs covered by a block."""
        block_tuple = tuple(sorted(block))
        if block_tuple not in block_to_tsubsets:
            subs = list(itertools.combinations(block_tuple, t))
            ids = [tsubset_to_id[s] for s in subs]
            block_to_tsubsets[block_tuple] = ids
        return block_to_tsubsets[block_tuple]
    
    # Greedy construction with randomized tie-breaking
    def greedy_construct():
        uncovered = np.ones(n_tsubsets, dtype=np.uint8)
        uncovered_count = n_tsubsets
        blocks = []
        block_set = set()
        
        # For efficiency, maintain a list of uncovered subset IDs
        uncovered_ids = list(range(n_tsubsets))
        
        while uncovered_count > 0:
            # Pick a random uncovered subset
            idx = random.randrange(len(uncovered_ids))
            t_id = uncovered_ids[idx]
            target = all_tsubsets[t_id]
            
            # Get candidate blocks for this subset
            candidates = subset_candidates[t_id]
            
            # Find best gain
            best_gain = -1
            best_cands = []
            
            for block in candidates:
                if block in block_set:
                    continue
                # Calculate gain: number of currently uncovered t-subsets in this block
                block_subs = get_block_tsubsets(block)
                gain = 0
                for sid in block_subs:
                    if uncovered[sid]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [block]
                elif gain == best_gain:
                    best_cands.append(block)
            
            if not best_cands:
                # All candidates already used - shouldn't happen in valid covering
                # Pick any block containing the target
                chosen = random.choice(candidates)
            else:
                chosen = random.choice(best_cands)
            
            blocks.append(list(chosen))
            block_set.add(chosen)
            
            # Update uncovered
            for sid in get_block_tsubsets(chosen):
                if uncovered[sid]:
                    uncovered[sid] = 0
                    uncovered_count -= 1
                    # Remove from uncovered_ids (swap-remove)
                    # Find position of sid in uncovered_ids
                    # For efficiency, we could maintain a position map, but let's keep it simple
                    # Actually, let's use a different approach: just rebuild periodically or use a set
            
            # Rebuild uncovered_ids occasionally for correctness
            if len(uncovered_ids) > 100 or uncovered_count % 500 == 0:
                uncovered_ids = [i for i in range(n_tsubsets) if uncovered[i]]
        
        return blocks
    
    # Local search: remove redundant blocks
    def remove_redundant(blocks):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not blocks:
            return blocks
        
        n_blocks = len(blocks)
        # For each t-subset, count how many blocks cover it
        coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsubsets_list = []
        
        for block in blocks:
            tsubs = get_block_tsubsets(tuple(sorted(block)))
            block_tsubsets_list.append(tsubs)
            for sid in tsubs:
                coverage_count[sid] += 1
        
        # A block is redundant if all its t-subsets are covered by other blocks
        # i.e., for every t-subset in the block, coverage_count > 1
        # But we need to be careful: removing one block might make another redundant
        # So we do iterative removal
        
        changed = True
        current_blocks = list(blocks)
        current_tsubsets = block_tsubsets_list
        
        while changed and len(current_blocks) > 0:
            changed = False
            # Recompute coverage
            coverage_count = np.zeros(n_tsubsets, dtype=np.int332)
            for tsubs in current_tsubsets:
                for sid in tsubs:
                    coverage_count[sid] += 1
            
            # Find a redundant block
            for i in range(len(current_blocks)):
                redundant = True
                for sid in current_tsubsets[i]:
                    if coverage_count[sid] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    current_blocks.pop(i)
                    current_tsubsets.pop(i)
                    changed = True
                    break
        
        return current_blocks
    
    # Simulated annealing
    def simulated_annealing(initial_blocks, time_limit):
        """Improve the solution using simulated annealing."""
        current_blocks = list(initial_blocks)
        current_len = len(current_blocks)
        best_blocks = list(current_blocks)
        best_len = current_len
        
        # Precompute block t-subsets
        def get_block_tsubs(block):
            return get_block_tsubsets(tuple(sorted(block)))
        
        current_tsubs = [get_block_tsubs(b) for b in current_blocks]
        
        # Compute initial coverage
        coverage = np.zeros(n_tsubsets, dtype=np.int32)
        for tsubs in current_tsubs:
            for sid in tsubs:
                coverage[sid] += 1
        
        n_uncovered = np.sum(coverage == 0)
        # Cost = number of blocks + penalty for uncovered
        def cost(blocks, n_uncov):
            return len(blocks) + 1000 * n_uncov
        
        current_cost = cost(current_blocks, n_uncovered)
        best_cost = current_cost
        
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        rng = np.random.RandomState(42)
        
        while time.time() < time_limit:
            if T < T_min:
                break
            
            # Choose a move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a random block
                # Pick a random t-subset, then a random candidate block for it
                t_id = rng.randint(0, n_tsubsets - 1)
                cands = subset_candidates[t_id]
                new_block = tuple(sorted(random.choice(cands)))
                
                if new_block in [tuple(sorted(b)) for b in current_blocks]:
                    continue
                
                # Compute gain
                new_tsubs = get_block_tsubsets(new_block)
                gain = 0
                for sid in new_tsubs:
                    if coverage[sid] == 0:
                        gain += 1
                
                new_cost = current_cost + 1 - 1000 * gain
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / T):
                    # Accept
                    current_blocks.append(list(new_block))
                    current_tsubs.append(new_tsubs)
                    for sid in new_tsubs:
                        coverage[sid] += 1
                    n_uncovered -= gain
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        best_len = len(current_blocks)
                        
            elif move_type < 0.7:
                # Remove a block if possible
                if not current_blocks:
                    continue
                idx = rng.randint(0, len(current_blocks) - 1)
                block_tsubs = current_tsubs[idx]
                
                # Check if removing this block leaves any t-subset uncovered
                would_uncover = 0
                for sid in block_tsubs:
                    if coverage[sid] <= 1:
                        would_uncover += 1
                
                if would_uncover > 0:
                    # Can't remove without uncovering
                    new_cost = current_cost - 1 + 1000 * would_uncover
                    delta_cost = new_cost - current_cost
                    if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / T):
                        # Accept (this makes things worse but might help escape local minima)
                        pass
                    continue
                
                # Safe to remove
                for sid in block_tsubs:
                    coverage[sid] -= 1
                current_blocks.pop(idx)
                current_tsubs.pop(idx)
                current_cost -= 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = list(current_blocks)
                    best_len = len(current_blocks)
                    
            else:
                # Swap: replace a block with another
                if not current_blocks:
                    continue
                idx = rng.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                old_tsubs = current_tsubs[idx]
                
                # Pick a new block that covers the same "critical" t-subsets
                # Find t-subsets uniquely covered by this block
                unique_sids = [sid for sid in old_tsubs if coverage[sid] == 1]
                
                if not unique_sids:
                    # No unique coverage, just pick a random different block
                    new_block = tuple(sorted(random.choice(random.choice(subset_candidates))))
                    if new_block == tuple(sorted(old_block)):
                        continue
                else:
                    # Pick a block that covers all unique_sids
                    # Intersect candidate sets
                    candidate_sets = [set(subset_candidates[tsubset_to_id[all_tsubsets[sid]]]) for sid in unique_sids]
                    # This is expensive, so just pick a random block containing all unique_sids
                    # For simplicity, skip this and just try random
                    new_block = tuple(sorted(random.choice(random.choice(subset_candidates))))
                    if new_block == tuple(sorted(old_block)):
                        continue
                
                new_tsubs = get_block_tsubsets(new_block)
                
                # Compute cost change
                # Remove old block's coverage
                for sid in old_tsubs:
                    coverage[sid] -= 1
                # Add new block's coverage
                for sid in new_tsubs:
                    coverage[sid] += 1
                
                new_n_uncovered = np.sum(coverage == 0)
                new_cost = len(current_blocks) + 1000 * new_n_uncovered
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or rng.random() < np.exp(-delta_cost / T):
                    # Accept
                    current_blocks[idx] = list(new_block)
                    current_tsubs[idx] = new_tsubs
                    current_cost = new_cost
                    n_uncovered = new_n_uncovered
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        best_len = len(current_blocks)
                else:
                    # Reject: revert
                    for sid in new_tsubs:
                        coverage[sid] -= 1
                    for sid in old_tsubs:
                        coverage[sid] += 1
            
            T *= cooling_rate
        
        return best_blocks
    
    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy constructions and pick the best
    while time.time() < deadline - 30.0:
        blocks = greedy_construct()
        # Remove redundant blocks
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: just do one greedy
        best_blocks = greedy_construct()
        best_blocks = remove_redundant(best_blocks)
        best_len = len(best_blocks)
    
    # If we have time, try simulated annealing to improve
    if time.time() < deadline - 5.0:
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time > 10.0:
            improved = simulated_annealing(best_blocks, deadline - 5.0)
            if len(improved) < best_len:
                best_blocks = improved
                best_len = len(improved)
    
    return best_blocks
