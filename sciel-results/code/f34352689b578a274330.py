import itertools
import random
import time

def construct(v, k, t):
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    # State variables
    blocks = []
    counts = [0] * num_subsets
    uncovered_count = num_subsets
    
    best_blocks = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115
    
    def update_counts(block, delta):
        nonlocal uncovered_count
        for s in itertools.combinations(block, t):
            idx = subset_to_idx[s]
            old = counts[idx]
            new = old + delta
            counts[idx] = new
            if old == 0 and new > 0:
                uncovered_count -= 1
            elif old > 0 and new == 0:
                uncovered_count += 1

    def is_redundant(block):
        for s in itertools.combinations(block, t):
            if counts[subset_to_idx[s]] <= 1:
                return False
        return True

    def remove_redundant():
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(blocks[i]):
                    update_counts(blocks[i], -1)
                    del blocks[i]
                    changed = True
                    if time.time() > deadline:
                        return

    def greedy_add_block():
        # Pick a random uncovered subset
        while True:
            idx = random.randint(0, num_subsets - 1)
            if counts[idx] == 0:
                break
        
        s0 = all_subsets[idx]
        block = list(s0)
        
        while len(block) < k:
            best_p = None
            max_gain = -1
            block_set = set(block)
            
            for p in range(v):
                if p in block_set:
                    continue
                
                gain = 0
                new_block = block + [p]
                for s in itertools.combinations(new_block, t):
                    if counts[subset_to_idx[s]] == 0:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_p = p
            
            if best_p is None:
                break
            block.append(best_p)
        
        block.sort()
        blocks.append(block)
        update_counts(block, 1)

    def greedy_construct():
        while uncovered_count > 0:
            if time.time() > deadline:
                return
            greedy_add_block()

    # Initial Construction
    greedy_construct()
    remove_redundant()
    
    if len(blocks) < best_size:
        best_size = len(blocks)
        best_blocks = [b[:] for b in blocks]
        
    current_blocks = [b[:] for b in blocks]
    current_size = len(blocks)
    
    # ILS Loop
    while time.time() < deadline:
        # Copy state
        trial_blocks = [b[:] for b in current_blocks]
        trial_counts = counts[:]
        trial_uncovered = uncovered_count
        
        # Perturb: Remove 1-3 blocks
        num_remove = random.randint(1, 3)
        if len(trial_blocks) > num_remove:
            indices_to_remove = random.sample(range(len(trial_blocks)), num_remove)
            for idx in sorted(indices_to_remove, reverse=True):
                b = trial_blocks[idx]
                for s in itertools.combinations(b, t):
                    s_idx = subset_to_idx[s]
                    trial_counts[s_idx] -= 1
                    if trial_counts[s_idx] == 0:
                        trial_uncovered += 1
                del trial_blocks[idx]
        else:
            # Restart if too small to perturb
            greedy_construct()
            remove_redundant()
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = [b[:] for b in blocks]
            current_blocks = [b[:] for b in blocks]
            current_size = len(blocks)
            continue
        
        # Repair: Add blocks to cover uncovered
        # We need to modify trial_blocks and trial_counts
        # Re-using logic from greedy_add_block but on trial state
        while trial_uncovered > 0:
            if time.time() > deadline:
                break
            
            # Pick random uncovered
            while True:
                idx = random.randint(0, num_subsets - 1)
                if trial_counts[idx] == 0:
                    break
            
            s0 = all_subsets[idx]
            block = list(s0)
            
            while len(block) < k:
                best_p = None
                max_gain = -1
                block_set = set(block)
                
                for p in range(v):
                    if p in block_set:
                        continue
                    
                    gain = 0
                    new_block = block + [p]
                    for s in itertools.combinations(new_block, t):
                        if trial_counts[subset_to_idx[s]] == 0:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_p = p
                
                if best_p is None:
                    break
                block.append(best_p)
            
            block.sort()
            trial_blocks.append(block)
            
            for s in itertools.combinations(block, t):
                s_idx = subset_to_idx[s]
                trial_counts[s_idx] += 1
                if trial_counts[s_idx] == 1:
                    trial_uncovered -= 1
        
        # Remove redundant from trial
        # We need a local remove_redundant that works on trial state
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(trial_blocks) - 1, -1, -1):
                # Check redundancy using trial_counts
                redundant = True
                for s in itertools.combinations(trial_blocks[i], t):
                    if trial_counts[subset_to_idx[s]] <= 1:
                        redundant = False
                        break
                if redundant:
                    b = trial_blocks[i]
                    for s in itertools.combinations(b, t):
                        s_idx = subset_to_idx[s]
                        trial_counts[s_idx] -= 1
                        if trial_counts[s_idx] == 0:
                            trial_uncovered += 1
                    del trial_blocks[i]
                    changed = True
        
        if len(trial_blocks) < best_size:
            best_size = len(trial_blocks)
            best_blocks = [b[:] for b in trial_blocks]
            
        if len(trial_blocks) < current_size:
            current_blocks = trial_blocks
            current_size = len(trial_blocks)
            # Update global state to current best for next iteration?
            # Or just keep current_blocks as the base for next perturbation.
            # Yes, keep current_blocks as the base.
            # But we also need to update `counts` and `uncovered_count` 
            # if we want to use them for something? 
            # No, we copy from current_blocks each time.
            # But `counts` and `uncovered_count` in the outer scope are stale.
            # We should update them to match current_blocks to avoid confusion,
            # or just rely on the copy step.
            # The copy step `trial_counts = counts[:]` uses the OUTER `counts`.
            # So we MUST update outer `counts` and `uncovered_count` when we accept a new current.
            
            # Update outer state
            blocks = current_blocks
            # Recalculate counts? Or just copy from trial?
            # trial_counts is the correct one for trial_blocks.
            counts = trial_counts
            uncovered_count = trial_uncovered

    return best_blocks
