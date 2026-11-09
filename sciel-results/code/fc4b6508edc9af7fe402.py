import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute block to t-subset mappings
    # For fast access, we store the list of t-subset indices for each block
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Convert to numpy arrays for faster iteration in some contexts, 
    # but lists are often faster for iteration in Python if the list is small.
    # block_tsubs[i] is a list of ints.
    # tsub_blocks[j] is a list of ints.

    def run_greedy_indices(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        # block_gain[i] = number of uncovered t-subsets covered by block i
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            
            if randomize:
                candidates = np.where(block_gain == max_gain)[0]
                best_block_idx = random.choice(candidates.tolist())
            else:
                best_block_idx = np.argmax(block_gain)
                
            selected_indices.append(best_block_idx)
            
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    for b_idx in tsub_blocks[si]:
                        block_gain[b_idx] -= 1
                        
        return selected_indices

    def anneal_solution(initial_indices, time_limit=5.0):
        """
        Improved Simulated Annealing.
        """
        if not initial_indices:
            return initial_indices
            
        M = 1000 # Penalty for uncovered t-subsets
        T = 1.0
        alpha = 0.95
        T_min = 1e-4
        
        # State
        selected = list(initial_indices)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        
        for b_idx in selected:
            for si in block_tsubs[b_idx]:
                cov[si] += 1
                
        uncovered_count = np.sum(cov == 0)
        current_cost = len(selected) + M * uncovered_count
        
        best_cost = current_cost
        best_selected = list(selected)
        
        start = time.time()
        
        while T > T_min and (time.time() - start) < time_limit and time.time() < DEADLINE:
            # Generate Move
            # 30% Remove, 30% Add, 40% Swap
            move_type = random.random()
            
            if move_type < 0.3 and selected:
                # Remove
                b_idx = random.choice(selected)
                delta_uncovered = 0
                # Check if any t-subset becomes uncovered
                for si in block_tsubs[b_idx]:
                    if cov[si] == 1:
                        delta_uncovered += 1
                delta_cost = -1 + M * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    for si in block_tsubs[b_idx]:
                        cov[si] -= 1
                    uncovered_count += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
                        
            elif move_type < 0.6 and uncovered_count > 0:
                # Add: Pick a random uncovered t-subset, then a random block covering it
                # Find an uncovered t-subset
                # To avoid O(N_t) scan every time, we could maintain a list, but N_t is 11628, which is small.
                # However, scanning 11k elements is fast in numpy.
                uncovered_indices = np.where(cov == 0)[0]
                if len(uncovered_indices) == 0:
                    continue
                target_si = random.choice(uncovered_indices.tolist())
                candidates = tsub_blocks[target_si]
                if not candidates:
                    continue
                b_idx = random.choice(candidates)
                
                if b_idx in selected_set:
                    continue # Block already selected, so adding it does nothing? 
                             # Wait, if it's already selected, cov[target_si] should be > 0. 
                             # So this case shouldn't happen if cov is consistent.
                             
                # Calculate Delta Cost
                # Newly covered t-subsets
                newly_covered = 0
                for si in block_tsubs[b_idx]:
                    if cov[si] == 0:
                        newly_covered += 1
                
                delta_cost = 1 - M * newly_covered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected.append(b_idx)
                    selected_set.add(b_idx)
                    for si in block_tsubs[b_idx]:
                        cov[si] += 1
                    uncovered_count -= newly_covered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
            else:
                # Swap: Pick a selected block b1, pick a block b2 covering one of b1's t-subsets
                if not selected:
                    continue
                b1_idx = random.choice(selected)
                b1_subs = block_tsubs[b1_idx]
                if not b1_subs:
                    continue
                target_si = random.choice(b1_subs)
                candidates = tsub_blocks[target_si]
                if not candidates:
                    continue
                b2_idx = random.choice(candidates)
                
                if b2_idx in selected_set:
                    # If b2 is already selected, this is a no-op swap (remove b1, add b2 which is already there? No, we remove b1 first)
                    # If b2 is already selected, removing b1 and adding b2 (which is already there) is just removing b1.
                    # This reduces to a Remove move. We can handle it or skip.
                    # Let's just skip to avoid complexity.
                    continue
                
                # Calculate Delta Cost for removing b1 and adding b2
                # 1. Simulate removal of b1
                #    Count how many t-subsets in b1 have cov == 1 (will become uncovered)
                #    Count how many t-subsets in b1 have cov > 1 (will remain covered)
                # 2. Simulate addition of b2
                #    Count how many t-subsets in b2 are currently uncovered (cov == 0) 
                #    Note: If b2 covers a t-subset that b1 uniquely covers, it might stay covered.
                
                # Efficient calculation:
                # Let's just iterate through b1 and b2.
                # Since |b1_subs| and |b2_subs| are 56, this is fast.
                
                # First, determine which t-subsets become uncovered if we remove b1.
                # These are si in b1 where cov[si] == 1.
                # Let U_b1 = {si in b1 | cov[si] == 1}
                # If we remove b1, these become uncovered.
                # Then we add b2.
                # Any si in U_b1 that is also in b2 will be covered again by b2.
                # Any si in b2 that is currently uncovered (cov[si] == 0) will be covered.
                # Note: si in b2 with cov[si] > 0 are already covered.
                
                # Delta Uncovered:
                # 1. Uncovered increase by |U_b1|
                # 2. Uncovered decrease by |U_b1 \cap b2| (these are re-covered by b2)
                # 3. Uncovered decrease by |{si in b2 | cov[si] == 0} \ U_b1| 
                #    Wait, if si in b2 and cov[si]==0, it was uncovered. Adding b2 covers it.
                #    If si is in U_b1, it was covered by b1, so cov[si] was 1. So it's not in {si | cov[si]==0}.
                #    So the sets are disjoint?
                #    Yes, because U_b1 has cov==1, and the other set has cov==0.
                
                newly_uncovered_from_b1 = 0
                for si in b1_subs:
                    if cov[si] == 1:
                        newly_uncovered_from_b1 += 1
                # This is |U_b1|
                
                # How many of these are re-covered by b2?
                re_covered_by_b2 = 0
                b2_set = set(block_tsubs[b2_idx])
                for si in b1_subs:
                    if cov[si] == 1 and si in b2_set:
                        re_covered_by_b2 += 1
                        
                # How many currently uncovered are covered by b2?
                newly_covered_by_b2 = 0
                for si in block_tsubs[b2_idx]:
                    if cov[si] == 0:
                        newly_covered_by_b2 += 1
                        
                delta_uncovered = newly_uncovered_from_b1 - re_covered_by_b2 - newly_covered_by_b2
                delta_cost = -1 + 1 + M * delta_uncovered # -1 for removing b1, +1 for adding b2
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    # Remove b1
                    selected.remove(b1_idx)
                    selected_set.discard(b1_idx)
                    for si in b1_subs:
                        cov[si] -= 1
                        
                    # Add b2
                    selected.append(b2_idx)
                    selected_set.add(b2_idx)
                    for si in block_tsubs[b2_idx]:
                        cov[si] += 1
                        
                    uncovered_count += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
            
            T *= alpha
        
        # Repair: Ensure validity
        # Greedily add blocks to cover any remaining uncovered t-subsets
        # Use a more efficient repair
        uncovered_tsubs = [i for i in range(N_t) if cov[i] == 0]
        while uncovered_tsubs:
            # Find block covering most uncovered
            best_b = -1
            max_cov = 0
            # Sample some blocks if list is too long?
            # For now, check all? Might be slow.
            # But N_blocks is 75k. N_t is 11k.
            # This repair phase should be fast if SA left few uncovered.
            # Optimization: Only check blocks that cover at least one uncovered t-subset.
            candidate_blocks = set()
            for si in uncovered_tsubs[:10]: # Check top 10 uncovered
                candidate_blocks.update(tsub_blocks[si])
            
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                count = 0
                for si in block_tsubs[b_idx]:
                    if cov[si] == 0:
                        count += 1
                if count > max_cov:
                    max_cov = count
                    best_b = b_idx
            
            if best_b == -1:
                break # Should not happen if problem is solvable
                
            selected.append(best_b)
            selected_set.add(best_b)
            for si in block_tsubs[best_b]:
                if cov[si] == 0:
                    cov[si] = 1
            # Re-calculate uncovered list
            uncovered_tsubs = [i for i in range(N_t) if cov[i] == 0]
            
        # Final Local Search: Reduction
        improved = True
        while improved:
            improved = False
            for b_idx in list(selected):
                can_remove = True
                for si in block_tsubs[b_idx]:
                    if cov[si] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    for si in block_tsubs[b_idx]:
                        cov[si] -= 1
                    improved = True
                    break # Restart loop
                    
        # Additional Polish: Try to swap blocks to enable further removals
        # This is a 2-opt style search.
        # For each pair (b1, b2) where b1 is selected and b2 is not,
        # if swapping b1 with b2 allows removal of b1 or b2 or another block, do it.
        # This is O(N^2) which is too slow for 75k blocks.
        # We will skip this for now as SA should have done most of the work.
        
        return selected

    # Main Loop
    best_sol_idx = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy
    sol_idx = run_greedy_indices(randomize=False)
    sol_idx = anneal_solution(sol_idx, time_limit=10.0)
    best_sol_idx = sol_idx
    best_len = len(sol_idx)
    
    # 2. Randomized Restarts
    restart_count = 0
    while time.time() < DEADLINE - 15:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy_indices(randomize=True, seed=seed)
        # Limit time per restart
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break
        time_limit = min(5.0, remaining - 10.0) # Leave 10s for repair/final checks
        if time_limit < 0.5:
            break
            
        cand_idx = anneal_solution(cand_idx, time_limit=time_limit)
        
        if len(cand_idx) < best_len:
            best_sol_idx = cand_idx
            best_len = len(cand_idx)
            
        restart_count += 1

    return [all_blocks[i] for i in best_sol_idx]
