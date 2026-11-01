import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask
    masks = [0] * num_k_subsets
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        masks[i] = mask
        
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # Strategy:
    # 1. Fast vectorized greedy to get a good initial solution quickly.
    # 2. Prune redundant blocks.
    # 3. Local Search (WalkSAT-style) to minimize block count further.
    
    # --- Phase 1: Fast Greedy Construction ---
    # We use a simple loop but with optimized gain updates.
    # Since num_k_subsets is ~75k and num_t_subsets is ~11628, bit operations are fast.
    
    def greedy_construct(randomize=True, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # gain[i] = number of uncovered t-subsets covered by k-subset i
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i]) # Initially all uncovered
        
        while uncovered:
            if time.time() > deadline:
                return None, False
            
            # Find max gain
            max_gain = 0
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                return current_blocks, False # Should not happen for valid parameters
            
            # Random choice among max gain candidates
            if randomize and candidates:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0] if candidates else -1
            
            if chosen_idx == -1:
                return current_blocks, False
                
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Efficient gain update using t_to_k
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                for i in t_to_k[idx_j]:
                    if i not in used_blocks:
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0 # Safety

        return current_blocks, True

    def greedy_construct_scarcity(randomize=True, seed=None):
        """
        Greedy construction using dynamic scarcity weights.
        For each uncovered t-subset j, compute its degree (number of unused blocks covering it).
        A block's score is the sum over its covered uncovered t-subsets of 1/degree(j).
        This prioritizes covering rare configurations.
        """
        if seed is not None:
            random.seed(seed)
        
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # degree[j] = number of unused blocks that cover t-subset j
        # Initialize with total counts
        degree = [0] * num_t_subsets
        for i in range(num_k_subsets):
            mask = masks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                degree[j] += 1
                m ^= lsb
        
        # score[i] for each block i
        score = [0.0] * num_k_subsets
        
        # Initial score calculation
        for i in range(num_k_subsets):
            s = 0.0
            m = masks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if degree[j] > 0:
                    s += 1.0 / degree[j]
                m ^= lsb
            score[i] = s
        
        while uncovered:
            if time.time() > deadline:
                return None, False
            
            # Find max score among unused blocks
            max_score = 0.0
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if score[i] > max_score:
                    max_score = score[i]
                    candidates = [i]
                elif abs(score[i] - max_score) < 1e-9 and score[i] > 0:
                    candidates.append(i)
            
            if max_score == 0.0:
                # Fallback to simple gain if scarcity fails
                return greedy_construct(randomize, seed)
            
            # Random choice among max score candidates
            if randomize and candidates:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0] if candidates else -1
            
            if chosen_idx == -1:
                return current_blocks, False
                
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and degrees
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Update degrees for t-subsets covered by chosen block
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                degree[idx_j] -= 1
            
            # Update scores for all unused blocks
            # A block's score changes only if it covers a t-subset whose degree changed.
            # We iterate over the t-subsets whose degree changed and adjust scores of blocks covering them.
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                
                old_deg = degree[idx_j] + 1
                new_deg = degree[idx_j]
                
                # For each block i that covers idx_j and is unused, update score[i]
                # score[i] -= 1/old_deg + 1/new_deg (if new_deg > 0)
                # Actually: old contribution was 1/old_deg, new is 1/new_deg (or 0 if new_deg==0)
                for i in t_to_k[idx_j]:
                    if i in used_blocks:
                        continue
                    # Remove old contribution
                    score[i] -= 1.0 / old_deg
                    # Add new contribution
                    if new_deg > 0:
                        score[i] += 1.0 / new_deg

        return current_blocks, True

    def prune_redundant(block_indices):
        """
        Remove blocks that are redundant.
        Returns a new list of block indices.
        """
        if not block_indices:
            return []
        
        # Count coverage
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        changed = True
        current_blocks = list(block_indices)
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                # Check if removing idx keeps all counts >= 1
                mask = masks[idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    # Remove idx
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(idx)
            current_blocks = new_blocks
        return current_blocks

    # --- Phase 2: Local Search (WalkSAT-style for Set Cover) ---
    # Objective: Minimize number of blocks.
    # Constraint: All t-subsets must be covered.
    # Move: Remove a block if it doesn't break coverage. Add a block if it helps.
    # We focus on "Remove" moves primarily, as greedy tends to overshoot.
    # If removal breaks coverage, we can try to "Swap" or "Add" then "Remove" others.
    # Simplest effective local search: Try to remove each block one by one. If it fails, keep it.
    # This is basically prune_redundant but applied iteratively with random order.
    
    def local_search_improve(block_indices, max_iters=1000):
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        current_size = len(current_blocks)
        
        # Precompute coverage counts
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        # Helper to check if a block is removable
        def is_removable(idx):
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if cover_counts[j] == 1:
                    return False
                m ^= lsb
            return True
        
        # Helper to remove a block
        def remove_block(idx):
            nonlocal current_blocks
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] -= 1
                m ^= lsb
            current_blocks.remove(idx)
            
        # Helper to add a block
        def add_block(idx):
            nonlocal current_blocks
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
            current_blocks.append(idx)

        # WalkSAT-like loop
        # We try to remove blocks. If we get stuck, we might try to add a block that covers
        # many "critical" t-subsets (those covered exactly once) to enable removal of multiple blocks?
        # That's complex. Let's stick to random order removal + occasional random swaps if stuck.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle blocks to break symmetry
            random.shuffle(current_blocks)
            
            for idx in current_blocks[:]:  # Iterate over a copy
                if time.time() > deadline:
                    break
                if is_removable(idx):
                    remove_block(idx)
                    improved = True
                    current_size -= 1
                    
        return current_blocks

    # Main Loop: Run multiple restarts
    # We interleave different strategies:
    # 1. Pure greedy (deterministic)
    # 2. Randomized greedy
    # 3. Randomized greedy + Local Search
    # 4. Scarcity-weighted greedy
    
    num_restarts = 20000  # Limit by time, not count
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose strategy based on restart index
        # 0: Standard Greedy + LS
        # 1: Randomized Greedy + LS
        # 2: Scarcity Greedy + LS
        # 3: Scarcity Greedy (no LS, to save time for more restarts)
        strategy = restart % 4
        
        if strategy == 0:
            blocks, success = greedy_construct(randomize=False, seed=restart)
            use_ls = True
        elif strategy == 1:
            blocks, success = greedy_construct(randomize=True, seed=restart)
            use_ls = True
        elif strategy == 2:
            blocks, success = greedy_construct_scarcity(randomize=True, seed=restart)
            use_ls = True
        else:
            blocks, success = greedy_construct_scarcity(randomize=True, seed=restart)
            use_ls = False
        
        if not success or blocks is None:
            continue
            
        if use_ls:
            blocks = local_search_improve(blocks)
            
        # Final prune just in case
        blocks = prune_redundant(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [k_subsets[i] for i in blocks]
            
            # If we hit a very good solution, we might want to spend more time on local search
            # But for now, just continue to find even better ones.
            
    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
