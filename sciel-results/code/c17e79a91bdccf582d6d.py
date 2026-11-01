import itertools
import random
import time
import math
from collections import Counter

def construct(v, k, t):
    # Precomputation
    # Map t-subsets to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for every possible k-block, the list of t-subset indices it covers
    # There are C(v, k) blocks. For v=19, k=7, this is C(19,7) = 50388.
    # Each block covers C(7,4) = 35 subsets.
    # Total operations: 50388 * 35 ~ 1.7M. This is fast enough to precompute.
    block_to_subs = {}
    for block in itertools.combinations(range(v), k):
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(t_sub_index[sub])
        block_to_subs[block] = subs
        
    # List of all possible blocks for random selection
    all_blocks = list(block_to_subs.keys())
    
    # Helper to generate a greedy initial cover
    def greedy_initial():
        random.seed(42) # Deterministic for reproducibility of start
        uncovered = [True] * num_t_subsets
        blocks = []
        
        # Greedy: pick block covering most uncovered
        # To speed up, we don't re-scan all blocks every time if possible, 
        # but for v=19, scanning 50k blocks * 150 iterations is 7.5M ops, which is fine.
        # However, updating "uncovered" status for all blocks is expensive.
        # Alternative: Randomized Greedy. Pick random block, if it covers any new, keep.
        # Let's do a simple greedy with random tie-breaking.
        
        # Faster greedy: 
        # Maintain a set of uncovered indices.
        # For each step, find block with max intersection with uncovered.
        # This is O(N_blocks * C(k,t)). 
        # 50388 * 35 = 1.7M per step. 
        # If we need ~150 steps, that's 250M ops. Might be too slow in Python.
        
        # Let's use a randomized approach for initialization:
        # Add random blocks until covered.
        while True:
            uncovered_count = sum(1 for u in uncovered if u)
            if uncovered_count == 0:
                break
            
            # Pick a random uncovered subset
            # Find its index
            # We need a list of currently uncovered indices.
            # Rebuilding this list is O(num_t_subsets) = 3876.
            unc_list = [i for i, u in enumerate(uncovered) if u]
            if not unc_list:
                break
            target_idx = random.choice(unc_list)
            
            # Find a block that covers this target_idx
            # We can precompute a reverse map: subset_idx -> list of blocks covering it
            # Let's create that map now if we don't have it.
            
            # Actually, let's just pick a random block and see if it helps.
            # If we pick 100 random blocks, we are likely to cover most.
            # This "Random Fill" is very fast.
            # Let's try: Add 200 random blocks. Then clean up.
            pass
        
        # Simpler Initialization: 
        # 1. Generate a set of random blocks until coverage is complete.
        # 2. Clean up.
        blocks = []
        uncovered = [True] * num_t_subsets
        # Estimate needed blocks: Lower bound ~131. Let's start with 200 random.
        # To ensure coverage, we can be smarter.
        # Let's use a greedy heuristic that is fast:
        # Pick a random uncovered subset. Pick a random block containing it.
        
        # Precompute reverse map: subset_idx -> list of blocks containing it
        # This takes 1.7M memory ops, acceptable.
        sub_to_blocks = [[] for _ in range(num_t_subsets)]
        for b, subs in block_to_subs.items():
            for s in subs:
                sub_to_blocks[s].append(b)
                
        while True:
            # Find any uncovered subset
            unc_list = [i for i, u in enumerate(uncovered) if u]
            if not unc_list:
                break
            target = random.choice(unc_list)
            # Pick a random block covering this target
            candidates = sub_to_blocks[target]
            if not candidates:
                # Should not happen
                break
            new_block = random.choice(candidates)
            blocks.append(new_block)
            # Update uncovered
            for s in block_to_subs[new_block]:
                uncovered[s] = False
                
        return blocks

    # Simulated Annealing
    def simulated_annealing(initial_blocks, time_limit=110):
        # State
        current_blocks = [tuple(b) for b in initial_blocks]
        coverage_counts = [0] * num_t_subsets
        
        # Initialize coverage
        for b in current_blocks:
            for s in block_to_subs[b]:
                coverage_counts[s] += 1
                
        def get_energy(blocks, cov):
            n_blocks = len(blocks)
            n_uncovered = sum(1 for c in cov if c == 0)
            # We want to minimize blocks. Uncovered is a penalty.
            # If valid, energy = n_blocks.
            # If invalid, energy = n_blocks + large_penalty * n_uncovered.
            # Let's use a dynamic penalty or just track best valid.
            # For SA acceptance, we need a scalar.
            # E = n_blocks * 1.0 + n_uncovered * 10.0
            # This encourages reducing blocks, but heavily penalizes uncovered.
            return n_blocks + 10.0 * n_uncovered

        current_energy = get_energy(current_blocks, coverage_counts)
        best_blocks = [tuple(b) for b in current_blocks]
        best_energy = current_energy
        best_valid_energy = float('inf') # Energy of best valid solution (just n_blocks)
        
        if sum(1 for c in coverage_counts if c == 0) == 0:
            best_valid_energy = len(current_blocks)
            
        T = 10.0 # Initial temperature
        T_min = 0.01
        cooling_rate = 0.999 # Adjust based on time
        
        start_time = time.time()
        
        # Precompute random block indices for speed? No, random.choice is fine.
        
        while time.time() - start_time < time_limit:
            # Choose move
            move_type = random.random()
            if move_type < 0.3:
                # Remove a block
                if len(current_blocks) <= 1:
                    T *= cooling_rate
                    continue
                idx = random.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                # Calculate delta
                # Remove old_block from coverage
                delta_uncovered = 0
                for s in block_to_subs[old_block]:
                    if coverage_counts[s] == 1:
                        delta_uncovered += 1
                delta_blocks = -1
                delta_E = delta_blocks + 10.0 * delta_uncovered
                
                # Apply or revert
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Apply
                    for s in block_to_subs[old_block]:
                        coverage_counts[s] -= 1
                    current_blocks.pop(idx)
                    current_energy += delta_E
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        # Check if valid
                        # We can check validity faster if we track uncovered count
                        # Let's just check if any coverage is 0
                        # This is O(num_t_subsets) = 3876. Acceptable.
                        is_valid = all(c > 0 for c in coverage_counts)
                        if is_valid:
                            if len(current_blocks) < best_valid_energy:
                                best_valid_energy = len(current_blocks)
                                best_blocks = [tuple(b) for b in current_blocks]
                else:
                    pass # Reject, do nothing
                    
            elif move_type < 0.7:
                # Add a block
                # Pick a random block
                new_block = tuple(random.choice(all_blocks))
                # If it's already in the list, adding it increases count but doesn't change coverage
                # We should avoid adding duplicates if possible, or treat them as distinct?
                # Blocks must be distinct in the final output. 
                # In SA state, we can allow duplicates temporarily? 
                # No, "blocks must be distinct". 
                # So if new_block is in current_blocks, this move is invalid/no-op.
                if new_block in current_blocks:
                    T *= cooling_rate
                    continue
                    
                delta_blocks = 1
                delta_uncovered = 0
                for s in block_to_subs[new_block]:
                    if coverage_counts[s] == 0:
                        delta_uncovered -= 1 # Uncovered decreases
                delta_E = delta_blocks + 10.0 * delta_uncovered
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Apply
                    for s in block_to_subs[new_block]:
                        coverage_counts[s] += 1
                    current_blocks.append(new_block)
                    current_energy += delta_E
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        is_valid = all(c > 0 for c in coverage_counts)
                        if is_valid:
                            if len(current_blocks) < best_valid_energy:
                                best_valid_energy = len(current_blocks)
                                best_blocks = [tuple(b) for b in current_blocks]
                else:
                    pass

            else:
                # Swap a block
                if len(current_blocks) <= 1:
                    T *= cooling_rate
                    continue
                idx = random.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                new_block = tuple(random.choice(all_blocks))
                
                if new_block == old_block:
                    T *= cooling_rate
                    continue
                    
                # Check if new_block is already in list (at another index)
                # If so, swapping makes a duplicate. We should avoid duplicates in state.
                # Simple check: if new_block is in current_blocks and current_blocks.index(new_block) != idx
                # This is O(N). N is ~150. OK.
                if new_block in current_blocks:
                    # If it exists elsewhere, swapping creates duplicate. 
                    # We can just skip or treat as invalid.
                    T *= cooling_rate
                    continue

                # Calculate delta
                # Remove old, Add new
                delta_blocks = 0
                delta_uncovered = 0
                
                # Temporarily remove old to see effect on uncovered?
                # Or calculate directly.
                # A subset s becomes uncovered if:
                # 1. It was covered ONLY by old_block (count 1) AND new_block does NOT cover it.
                # 2. It was uncovered (count 0) AND new_block DOES cover it.
                
                # Let's iterate over subsets of old_block and new_block.
                # Subsets of old_block:
                old_subs = block_to_subs[old_block]
                new_subs = block_to_subs[new_block]
                
                # We need to know which subsets are in both, only old, only new.
                # Sets are faster for intersection.
                old_set = set(old_subs)
                new_set = set(new_subs)
                
                # Subsets only in old: old_set - new_set
                # Subsets only in new: new_set - old_set
                # Subsets in both: old_set & new_set (count stays same)
                
                only_old = old_set - new_set
                only_new = new_set - old_set
                
                # Effect of removing old_block:
                for s in only_old:
                    if coverage_counts[s] == 1:
                        # It will become uncovered
                        delta_uncovered += 1
                    # If count > 1, it stays covered by others, no change in uncovered count
                    # Note: We are removing old_block, so count decreases by 1.
                    # If count was 1, it becomes 0 (uncovered).
                    # If count was >1, it becomes >0 (covered).
                    
                # Effect of adding new_block:
                for s in only_new:
                    # Note: We haven't updated coverage_counts yet.
                    # The current count is BEFORE the swap.
                    # If s is in only_new, it was NOT covered by old_block.
                    # So its count is the same as before removing old_block (which didn't affect it).
                    # If count is 0, it was uncovered. Adding new_block makes it covered.
                    if coverage_counts[s] == 0:
                        delta_uncovered -= 1
                        
                # Subsets in both: count decreases by 1 (remove old) and increases by 1 (add new). Net 0.
                # So no change in uncovered status for these.
                
                delta_E = delta_blocks + 10.0 * delta_uncovered
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Apply
                    # Remove old
                    for s in old_subs:
                        coverage_counts[s] -= 1
                    # Add new
                    for s in new_subs:
                        coverage_counts[s] += 1
                    current_blocks[idx] = new_block
                    current_energy += delta_E
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        is_valid = all(c > 0 for c in coverage_counts)
                        if is_valid:
                            if len(current_blocks) < best_valid_energy:
                                best_valid_energy = len(current_blocks)
                                best_blocks = [tuple(b) for b in current_blocks]
                else:
                    pass
                    
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_blocks

    # Main Execution
    start_time = time.time()
    
    # 1. Generate Initial Cover
    initial_blocks = greedy_initial()
    
    # 2. Run SA
    # Leave 5 seconds for cleanup
    time_limit = max(5, 120 - (time.time() - start_time) - 5)
    sa_blocks = simulated_annealing(initial_blocks, time_limit)
    
    # 3. Final Cleanup (Remove Redundant Blocks)
    # Convert to list of lists
    final_blocks = [list(b) for b in sa_blocks]
    
    # Clean up
    coverage_count = Counter()
    for b in final_blocks:
        for sub in itertools.combinations(b, t):
            coverage_count[sub] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for b in final_blocks:
            redundant = True
            for sub in itertools.combinations(b, t):
                if coverage_count[sub] <= 1:
                    redundant = False
                    break
            if redundant:
                changed = True
                for sub in itertools.combinations(b, t):
                    coverage_count[sub] -= 1
                    if coverage_count[sub] == 0:
                        del coverage_count[sub]
            else:
                new_blocks.append(b)
        final_blocks = new_blocks
        
    return final_blocks
