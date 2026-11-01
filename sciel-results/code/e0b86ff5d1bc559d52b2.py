import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N = len(all_blocks)
    
    all_targets = list(itertools.combinations(points, t))
    M = len(all_targets)
    
    target_index = {target: i for i, target in enumerate(all_targets)}
    
    # Precompute subsets for each block (list of target indices)
    block_subsets = []
    for block in all_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(target_index[sub])
        block_subsets.append(subs)
        
    # Precompute target_to_blocks for greedy initialization
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for j in subs:
            target_to_blocks[j].append(i)
            
    random.seed(42)
    
    # --- 1. Greedy Initialization ---
    cov = np.zeros(M, dtype=np.int32)
    selected = set()
    
    def add_block(i):
        selected.add(i)
        for s in block_subsets[i]:
            cov[s] += 1
            
    def remove_block(i):
        selected.remove(i)
        for s in block_subsets[i]:
            cov[s] -= 1

    # Greedy Set Cover
    uncovered = set(range(M))
    while uncovered:
        # Pick a random uncovered target
        t_idx = random.choice(list(uncovered))
        candidates = target_to_blocks[t_idx]
        
        best_block = -1
        max_new = -1
        
        for b in candidates:
            if b in selected:
                continue
            # Count how many subsets in b are currently uncovered
            new_cov_count = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    new_cov_count += 1
            if new_cov_count > max_new:
                max_new = new_cov_count
                best_block = b
                
        if best_block != -1:
            add_block(best_block)
            # Update uncovered
            for s in block_subsets[best_block]:
                if cov[s] == 1: # Just became covered
                    uncovered.discard(s)
        else:
            break
            
    # Remove redundant blocks in initial solution
    def remove_redundant():
        removed = True
        while removed:
            removed = False
            for i in list(selected):
                redundant = True
                for s in block_subsets[i]:
                    if cov[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(i)
                    removed = True
                    
    remove_redundant()
    
    current_best_size = len(selected)
    best_selected = set(selected)
    best_cov = cov.copy()
    
    # --- 2. Simulated Annealing ---
    # Cost function: lambda * uncovered + selected_count
    # We want to minimize this.
    # Let's choose lambda large enough so that coverage is prioritized.
    # If we add a block covering 1 new target, cost change is 1 - lambda.
    # If lambda > 1, we prefer adding blocks that cover new targets.
    # However, we also want to remove blocks.
    # If we remove a block that is redundant, cost change is -1.
    # If we remove a block that breaks coverage of 1 target, cost change is -1 + lambda.
    # If lambda is large, we avoid breaking coverage.
    
    LAMBDA = 10.0 # Penalty for uncovered targets
    start_time = time.time()
    time_limit = 110.0
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    # We don't know exact number of steps, so use time-based or step-based with dynamic T?
    # Let's use a fixed number of iterations estimated by time.
    # Or just decrease T over time.
    
    # Current state
    cov = best_cov.copy()
    selected = set(best_selected)
    
    # Helper to get current cost
    def get_cost():
        uncovered_count = np.sum(cov == 0)
        return LAMBDA * uncovered_count + len(selected)
    
    current_cost = get_cost()
    best_cost = current_cost
    
    # Precompute random indices for speed? No, just use random.
    
    # To speed up delta calculation, we can precompute the list of targets for each block.
    # We already have block_subsets.
    
    while time.time() - start_time < time_limit:
        # Determine temperature
        # Simple linear decay based on time? Or exponential?
        # Let's do exponential decay of temperature over the duration.
        # We need a reference time. Let's say T decays from T_start to T_end over time_limit.
        # T = T_end + (T_start - T_end) * exp(-k * elapsed)
        # Or simply T = T_start * (T_end/T_start)^(elapsed/time_limit)
        elapsed = time.time() - start_time
        T = T_end + (T_start - T_end) * (1.0 - elapsed / time_limit)
        if T < T_end:
            T = T_end
            
        # Choose move: Add or Remove
        # Probability of remove should be higher if we have many blocks?
        # Or just 50/50.
        if random.random() < 0.5 and selected:
            # Remove move
            victim = random.choice(list(selected))
            # Calculate delta cost
            # Removing victim:
            # -1 block
            # + lambda * (number of targets that become uncovered)
            # A target becomes uncovered if cov[target] == 1 and target is in victim.
            newly_uncovered = 0
            for s in block_subsets[victim]:
                if cov[s] == 1:
                    newly_uncovered += 1
            delta_cost = -1.0 + LAMBDA * newly_uncovered
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                remove_block(victim)
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_selected = set(selected)
                    best_cov = cov.copy()
                    
        else:
            # Add move
            # Pick a random block not in selected
            # Efficient way: pick random index, if in selected, try again?
            # N is 75k. Selected is ~300. Collision is rare.
            b = random.randint(0, N - 1)
            while b in selected:
                b = random.randint(0, N - 1)
            
            # Calculate delta cost
            # Adding b:
            # +1 block
            # - lambda * (number of targets that become covered)
            # A target becomes covered if cov[target] == 0 and target is in b.
            newly_covered = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    newly_covered += 1
            delta_cost = 1.0 - LAMBDA * newly_covered
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                add_block(b)
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_selected = set(selected)
                    best_cov = cov.copy()
                    
        # Occasionally, try to remove redundant blocks to clean up
        # This helps escape local minima where redundant blocks stick around
        if random.random() < 0.01:
             # Try to remove one redundant block if exists
             # This is O(|selected| * 56). 
             # We can just do a full remove_redundant pass occasionally?
             # That might be too slow if done every step.
             # Let's just rely on the SA moves. Removing a redundant block has delta_cost = -1 (if it doesn't break coverage).
             # So SA will naturally find and remove them if T is low enough.
             pass
            
    # Restore best solution
    selected = best_selected
    
    blocks = []
    for i in selected:
        blocks.append(list(all_blocks[i]))
        
    # Ensure distinct and sorted
    unique_blocks = []
    seen = set()
    for b in blocks:
        b.sort()
        t_b = tuple(b)
        if t_b not in seen:
            seen.add(t_b)
            unique_blocks.append(b)
            
    return unique_blocks
