# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Stochastic Local Search for covering design C(v, k, t).
    1. Initialize with a greedy covering.
    2. Refine using Simulated Annealing:
       - State: List of k-subsets (blocks).
       - Move: Pick a random block, pick a random point in it, replace with a random point not in it.
       - Objective: Maximize coverage (minimize uncovered t-subsets).
       - Acceptance: Always accept improvements. Accept worsenings with probability exp(-delta/T).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Pre-compute all t-subsets for fast coverage counting
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Coverage count array: count[i] = number of blocks covering all_t_subsets[i]
    coverage = [0] * n_t
    uncovered_count = n_t
    
    def apply_block(block):
        """Update coverage arrays for adding a block."""
        nonlocal uncovered_count
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            if coverage[idx] == 0:
                uncovered_count -= 1
            coverage[idx] += 1

    def remove_block(block):
        """Update coverage arrays for removing a block."""
        nonlocal uncovered_count
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            coverage[idx] -= 1
            if coverage[idx] == 0:
                uncovered_count += 1

    # --- Phase 1: Greedy Initialization ---
    # We need a valid covering to start. We'll use a simple greedy set cover.
    # This might be slow if done naively, so we limit it or use a faster heuristic.
    # Given v=24, k=7, t=3, n_t = 2024. Greedy is fast enough for init.
    
    # To ensure we have a valid start, we run a greedy cover.
    # However, standard greedy might take a bit. Let's do a "randomized greedy" 
    # or just a standard greedy but stop if we hit time? 
    # Actually, for v=24, greedy is very fast (<0.1s).
    
    uncovered_set = set(range(n_t))
    blocks = []
    
    # Helper to get index of t-subset from block
    # We need to efficiently find which t-subsets a new block covers.
    
    # Greedy Loop
    while uncovered_set and time.time() < deadline - 5.0:
        # Pick the first uncovered t-subset (or random one for diversity? Let's stick to deterministic for init stability)
        # Actually, randomizing the initial cover can help avoid specific bad local minima.
        # But we need a valid cover. Let's just do standard greedy.
        
        # To speed up greedy, we can pick the t-subset that is "hardest" to cover? 
        # Or just the first one.
        target_t = next(iter(uncovered_set))
        target_points = all_t_subsets[target_t]
        
        # Build a block containing target_points
        # Start with the points in target_t
        current_block = list(target_points)
        
        # Greedily add points that cover the most uncovered t-subsets
        while len(current_block) < k:
            best_point = None
            best_gain = -1
            
            # Candidate points: those not in current_block
            candidates = [p for p in range(v) if p not in current_block]
            
            # Calculate gain for each candidate
            # Gain = number of uncovered t-subsets that become covered
            # A t-subset is covered by (current_block + [x]) if it contains x and (t-1) points from current_block
            # AND was previously uncovered.
            
            # Optimization: Precompute for current_block which t-subsets are "almost covered" (missing 1 point)
            # This is complex. For v=24, k=7, t=3, the inner loop is small.
            # Number of candidates <= 24.
            # For each candidate, check all t-subsets? No, that's 2024 * 24 * 7 ~ 350k ops per block.
            # 76 blocks * 350k ~ 26M ops. Might be too slow in Python for init.
            
            # Alternative Init: Random blocks until valid?
            # Or use the existing seed's logic which was 1.0 (valid).
            # The seed program is:
            # uncovered = set(itertools.combinations(range(v), t))
            # while uncovered:
            #   block = list(min(uncovered))
            #   while len(block) < k:
            #     def gain(x): ...
            #   blocks.append(block)
            #   uncovered.difference_update(...)
            
            # Let's reuse the seed's init logic but ensure it's fast enough.
            # The seed's gain calculation:
            # sum(1 for rest in itertools.combinations(block, t-1) if tuple(sorted(rest + (x,))) in uncovered)
            # This is efficient because it only checks subsets formed by the current block.
            
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(current_block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if t_subset_to_idx[ts] in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                # Should not happen if uncovered is not empty and v is large enough
                break
            current_block.append(best_point)
        
        if len(current_block) < k:
            # Fill with random points if stuck (shouldn't happen)
            while len(current_block) < k:
                current_block.append(random.randint(0, v-1))
        
        current_block.sort()
        blocks.append(current_block)
        
        # Update uncovered_set
        for ts in itertools.combinations(current_block, t):
            idx = t_subset_to_idx[ts]
            uncovered_set.discard(idx)
            coverage[idx] += 1 # Keep coverage array in sync for the SA phase
            
            if coverage[idx] == 1:
                 # It was uncovered, now covered
                 pass
            # Note: coverage array in seed init wasn't used, but we need it for SA.
            # The logic above:
            # 1. We added block to `blocks`.
            # 2. We updated `uncovered_set`.
            # 3. We updated `coverage` incrementally.
            # But wait, `uncovered_set` removal logic:
            # If I add a block, some t-subsets become covered.
            # I should check if coverage[idx] was 0 before incrementing.
            
            # Let's restart init coverage logic to be safe and clean.
            # The loop above is getting messy. Let's just run the seed's greedy, 
            # then recompute coverage from scratch for the SA phase. It's O(N_blocks * C(k,t)).
            # 76 * 35 = 2660 ops. Negligible.
            
            # So, just run seed greedy to get `blocks`.
            break # Exit inner while? No, exit outer while?
        
        # The structure above is flawed for a single pass. 
        # Let's just copy the seed's greedy exactly for init.
        
    # RE-INITIALIZATION STRATEGY:
    # 1. Run the exact seed greedy to get a valid `blocks` list.
    # 2. Compute `coverage` and `uncovered_count` from `blocks`.
    # 3. Run SA.
    
    # Reset
    blocks = []
    uncovered_set = set(range(n_t))
    
    while uncovered_set and time.time() < deadline - 5.0:
        target_t = next(iter(uncovered_set))
        target_points = all_t_subsets[target_t]
        current_block = list(target_points)
        
        while len(current_block) < k:
            best_point = None
            best_gain = -1
            candidates = [p for p in range(v) if p not in current_block]
            
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(current_block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if t_subset_to_idx[ts] in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            current_block.append(best_point)
        
        if len(current_block) < k:
            while len(current_block) < k:
                current_block.append(random.randint(0, v-1))
        
        current_block.sort()
        blocks.append(current_block)
        
        for ts in itertools.combinations(current_block, t):
            uncovered_set.discard(t_subset_to_idx[ts])
            
    if not uncovered_set:
        # We have a valid covering from greedy
        pass
    else:
        # If greedy didn't finish in time (unlikely for v=24), we return what we have? 
        # No, we must return a valid covering. 
        # The seed program is guaranteed to finish quickly. 
        # If we are here, it means time.time() > deadline - 5.0.
        # We should probably just return the seed result if we haven't started SA yet.
        # But for now, assume greedy finishes.
        
        # Force complete if stuck (should not happen)
        while uncovered_set:
             target_t = next(iter(uncovered_set))
             target_points = all_t_subsets[target_t]
             current_block = list(target_points)
             while len(current_block) < k:
                 current_block.append(random.randint(0, v-1))
             current_block.sort()
             blocks.append(current_block)
             for ts in itertools.combinations(current_block, t):
                 uncovered_set.discard(t_subset_to_idx[ts])
             if time.time() > deadline:
                 break

    # Compute initial coverage state for SA
    coverage = [0] * n_t
    uncovered_count = n_t
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            coverage[idx] += 1
            if coverage[idx] == 1:
                uncovered_count -= 1

    best_blocks = blocks[:]
    best_uncovered = uncovered_count
    best_time = time.time()

    # --- Phase 2: Simulated Annealing ---
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Number of blocks is fixed for now. 
    # Move: Pick a block i, pick a point p in block_i, replace with q not in block_i.
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Pick random block
        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]
        
        # Pick a point to remove
        p_idx = random.randint(0, k - 1)
        p = old_block[p_idx]
        
        # Pick a point to add
        # Candidates: points not in old_block
        # To be efficient, we can just pick a random point from 0..v-1 and check if it's in the block.
        # Or maintain a set.
        # Since v=24, checking membership is fast.
        q = random.randint(0, v - 1)
        if q in old_block:
            continue # Retry if collision (rare)
            
        # Construct new block
        # Remove p, add q, sort
        new_block_set = set(old_block)
        new_block_set.remove(p)
        new_block_set.add(q)
        new_block = tuple(sorted(new_block_set))
        
        # Calculate delta (change in uncovered_count)
        # Delta = (new_uncovered_count - old_uncovered_count)
        # We want to minimize uncovered_count.
        # So we accept if new_uncovered < old_uncovered (delta < 0)
        
        # Efficient calculation:
        # 1. Remove old_block from coverage
        # 2. Add new_block to coverage
        # 3. Check change in uncovered_count
        
        # To avoid modifying global arrays and reverting on rejection, we can calculate delta directly.
        # Delta = (t-subsets lost when removing old_block) - (t-subsets gained when adding new_block)
        # A t-subset is "lost" if it was covered ONLY by old_block (coverage[idx] == 1) and is in old_block.
        # A t-subset is "gained" if it was NOT covered by any block (coverage[idx] == 0) and is in new_block.
        
        delta = 0
        
        # Subsets in old_block that might be lost
        for ts in itertools.combinations(old_block, t):
            idx = t_subset_to_idx[ts]
            if coverage[idx] == 1:
                delta += 1 # This subset becomes uncovered
        
        # Subsets in new_block that might be gained
        for ts in itertools.combinations(new_block, t):
            idx = t_subset_to_idx[ts]
            if coverage[idx] == 0:
                delta -= 1 # This subset becomes covered
        
        # Note: Subsets in intersection of old and new blocks are handled correctly?
        # If a subset is in both, it's covered by old (count>=1) and covered by new.
        # Removing old: if count==1, it becomes 0.
        # Adding new: if count==0, it becomes 1.
        # If it was in both, and count==1 (only old covered it), then:
        #   Remove old -> count 0.
        #   Add new -> count 1.
        #   Net change 0.
        #   My delta logic:
        #   Lost: coverage==1 -> +1
        #   Gained: coverage==0 -> -1
        #   Wait. The "Gained" check uses the CURRENT coverage (before move).
        #   If coverage==1 (only old covered it), it is NOT 0, so it is not "gained".
        #   So delta = +1 (lost) - 0 (gained) = +1.
        #   But actually, after move, it is covered by new. So it should not be lost.
        #   The issue is that "Gained" should check if it will be covered by the NEW block, 
        #   given the state AFTER removing the old block.
        
        # Correct Logic:
        # State S0.
        # S1 = S0 - old_block.
        # S2 = S1 + new_block.
        
        # Delta = Uncovered(S2) - Uncovered(S0)
        
        # Let's just apply and revert. It's safer and v is small.
        # Cost: O(C(k,t)) ~ 35 ops. 150s / 35 ops ~ 4M ops/s. Python can do ~10M-50M simple ops/s.
        # So we can do ~1M-2M moves.
        
        # Apply Move
        for ts in itertools.combinations(old_block, t):
            idx = t_subset_to_idx[ts]
            coverage[idx] -= 1
            if coverage[idx] == 0:
                uncovered_count += 1
        
        for ts in itertools.combinations(new_block, t):
            idx = t_subset_to_idx[ts]
            coverage[idx] += 1
            if coverage[idx] == 1:
                uncovered_count -= 1
        
        new_uncovered = uncovered_count
        delta = new_uncovered - best_uncovered # No, delta should be vs current state?
        # Actually, we compare new_uncovered with the uncovered_count BEFORE the move.
        # Let's store old_uncovered before the move.
        
        # I messed up the flow. Let's restart the move logic inside the loop properly.
        
        # Break out to rewrite the loop cleanly.
        break # Just to exit the placeholder loop
        
    # REWRITING SA LOOP CLEANLY
    
    current_uncovered = uncovered_count
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]
        
        p_idx = random.randint(0, k - 1)
        p = old_block[p_idx]
        
        q = random.randint(0, v - 1)
        if q in old_block:
            continue
            
        new_block_set = set(old_block)
        new_block_set.remove(p)
        new_block_set.add(q)
        new_block = tuple(sorted(new_block_set))
        
        # Apply
        old_uncovered = current_uncovered
        
        for ts in itertools.combinations(old_block, t):
            idx = t_subset_to_idx[ts]
            coverage[idx] -= 1
            if coverage[idx] == 0:
                current_uncovered += 1
                
        for ts in itertools.combinations(new_block, t):
            idx = t_subset_to_idx[ts]
            coverage[idx] += 1
            if coverage[idx] == 1:
                current_uncovered -= 1
        
        delta = current_uncovered - old_uncovered
        
        if delta <= 0:
            # Improvement or neutral
            blocks[i] = list(new_block)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_blocks = blocks[:]
                best_time = time.time()
        else:
            # Worse
            if random.random() < pow(2.71828, -delta / T):
                blocks[i] = list(new_block)
            else:
                # Revert
                for ts in itertools.combinations(new_block, t):
                    idx = t_subset_to_idx[ts]
                    coverage[idx] -= 1
                    if coverage[idx] == 0:
                        current_uncovered += 1
                for ts in itertools.combinations(old_block, t):
                    idx = t_subset_to_idx[ts]
                    coverage[idx] += 1
                    if coverage[idx] == 1:
                        current_uncovered -= 1
                        
        T *= cooling
        
    # Return best found
    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
