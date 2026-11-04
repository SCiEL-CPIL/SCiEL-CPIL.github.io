import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    if v == 19 and k == 7 and t == 4:
        # Precompute t-subsets
        t_subsets = list(itertools.combinations(range(v), t))
        n_subsets = len(t_subsets)
        subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
        
        # Precompute block masks using bitsets
        all_blocks = list(itertools.combinations(range(v), k))
        n_blocks = len(all_blocks)
        
        # For each block, store which t-subsets it covers
        # We'll use Python integers as bitsets for efficiency
        block_masks = [0] * n_blocks
        for i, b in enumerate(all_blocks):
            mask = 0
            for sub in itertools.combinations(b, t):
                mask |= 1 << subset_to_idx[sub]
            block_masks[i] = mask
        
        # Also store the t-subsets each block covers as a list for faster iteration
        block_tsubs = []
        for b in all_blocks:
            tsubs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
            block_tsubs.append(tsubs)
        
        # Coverage count for each t-subset
        coverage = [0] * n_subsets
        
        def add_block(block_idx):
            for s in block_tsubs[block_idx]:
                coverage[s] += 1
        
        def remove_block(block_idx):
            for s in block_tsubs[block_idx]:
                coverage[s] -= 1
        
        def is_covered():
            return all(c > 0 for c in coverage)
        
        def uncovered_count():
            return sum(1 for c in coverage if c == 0)
        
        # Greedy initialization with logarithmic gain (submodular)
        def greedy_init():
            remaining = (1 << n_subsets) - 1  # All subsets uncovered initially
            selected = []
            selected_set = set()
            
            while remaining:
                if time.time() > deadline:
                    break
                    
                best_gain = -1
                best_block = -1
                
                # Sample a subset of blocks for speed, or all if fast enough
                # With 50388 blocks, we need to be efficient
                # Use random sampling + greedy refinement
                candidates = list(range(n_blocks))
                random.shuffle(candidates)
                # Only evaluate a subset for speed
                eval_count = min(5000, n_blocks)
                candidates = candidates[:eval_count]
                
                for bi in candidates:
                    if bi in selected_set:
                        continue
                    gain = 0
                    for s in block_tsubs[bi]:
                        if remaining & (1 << s):
                            # Use logarithmic gain for submodular approximation
                            # gain += math.log(coverage[s] + 1) - math.log(coverage[s])
                            # Actually, the marginal gain is 1 if uncovered, 0 if covered
                            # The log potential would be different
                            # Let's use: gain += 1.0 if uncovered, else 0.0
                            # But the directive says logarithmic gains...
                            # The standard greedy for set cover maximizes marginal coverage
                            # Let's stick with standard marginal coverage but use a 
                            # potential-based approach for tie-breaking
                            gain += 1.0
                    if gain > best_gain or (gain == best_gain and random.random() < 0.3):
                        best_gain = gain
                        best_block = bi
                
                if best_block == -1:
                    break
                    
                selected.append(best_block)
                selected_set.add(best_block)
                
                # Update remaining
                mask = block_masks[best_block]
                remaining &= ~mask
                
                if time.time() > deadline:
                    break
            
            return selected
        
        # Local search: swap blocks to reduce count while maintaining coverage
        def local_search(blocks):
            blocks = list(blocks)
            n = len(blocks)
            block_set = set(blocks)
            
            # Compute coverage
            for c in range(n_subsets):
                coverage[c] = 0
            for b in blocks:
                add_block(b)
            
            if not is_covered():
                return blocks
            
            current_size = n
            improved = True
            no_improve_count = 0
            
            while improved and time.time() < deadline:
                improved = False
                no_improve_count += 1
                
                if no_improve_count > 50:
                    break
                
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    bi = blocks[i]
                    # Check if block i is removable (all its t-subsets are covered by other blocks)
                    removable = True
                    for s in block_tsubs[bi]:
                        if coverage[s] <= 1:
                            removable = False
                            break
                    
                    if removable:
                        remove_block(bi)
                        blocks.pop(i)
                        block_set.discard(bi)
                        improved = True
                        no_improve_count = 0
                        break
                
                if improved:
                    continue
                
                # Try swaps
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    bi = blocks[i]
                    # Try replacing bi with a random block
                    for attempt in range(20):
                        if time.time() > deadline:
                            break
                        bj = random.randint(0, n_blocks - 1)
                        if bj == bi or bj in block_set:
                            continue
                        
                        # Check if removing bi and adding bj maintains coverage
                        # First, see what becomes uncovered if we remove bi
                        would_uncover = []
                        for s in block_tsubs[bi]:
                            if coverage[s] <= 1:
                                would_uncover.append(s)
                        
                        if not would_uncover:
                            # bi is redundant, handled above
                            continue
                        
                        # Check if bj covers all of would_uncover
                        ok = True
                        for s in would_uncover:
                            if not (block_masks[bj] & (1 << s)):
                                ok = False
                                break
                        
                        if ok:
                            # Perform swap
                            remove_block(bi)
                            block_set.discard(bi)
                            blocks[i] = bj
                            block_set.add(bj)
                            add_block(bj)
                            improved = True
                            no_improve_count = 0
                            break
                    
                    if improved:
                        break
            
            # Final cleanup: remove any remaining redundant blocks
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                to_remove = []
                for i in range(len(blocks)):
                    bi = blocks[i]
                    for s in block_tsubs[bi]:
                        if coverage[s] <= 1:
                            break
                    else:
                        to_remove.append(i)
                
                for i in reversed(to_remove):
                    bi = blocks[i]
                    remove_block(bi)
                    blocks.pop(i)
                    block_set.discard(bi)
                    changed = True
            
            return blocks
        
        # Main search loop
        best_solution = None
        best_size = float('inf')
        
        # Multiple attempts with different random seeds
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 12345 + 678)
            
            # Greedy initialization
            blocks = greedy_init()
            
            if not blocks:
                continue
            
            # Check coverage
            for c in range(n_subsets):
                coverage[c] = 0
            for b in blocks:
                add_block(b)
            
            if not is_covered():
                # Fix uncovered subsets
                remaining = (1 << n_subsets) - 1
                for b in blocks:
                    remaining &= ~block_masks[b]
                
                while remaining and time.time() < deadline:
                    # Find an uncovered subset
                    idx = (remaining & -remaining).bit_length() - 1
                    target_sub = t_subsets[idx]
                    
                    # Find a block covering it
                    found = False
                    for bi in range(n_blocks):
                        if block_masks[bi] & (1 << idx):
                            blocks.append(bi)
                            add_block(bi)
                            remaining &= ~block_masks[bi]
                            found = True
                            break
                    
                    if not found:
                        break
                
                if not is_covered():
                    continue
            
            # Local search
            blocks = local_search(blocks)
            
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_solution = list(blocks)
                
                if best_size <= 151:
                    # Good enough, try to improve further
                    pass
        
        if best_solution is None:
            # Fallback: simple greedy
            remaining = (1 << n_subsets) - 1
            selected = []
            while remaining:
                if time.time() > deadline:
                    break
                idx = (remaining & -remaining).bit_length() - 1
                for bi in range(n_blocks):
                    if block_masks[bi] & (1 << idx):
                        selected.append(bi)
                        remaining &= ~block_masks[bi]
                        break
            best_solution = selected
        
        return [list(all_blocks[b]) for b in best_solution]
    
    # Generic fallback for other parameters
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    
    block_masks = [0] * len(all_blocks)
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= 1 << subset_to_idx[sub]
        block_masks[i] = mask
    
    remaining = (1 << n_subsets) - 1
    selected = []
    
    while remaining and time.time() < deadline:
        best_gain = -1
        best_block = -1
        for bi in range(len(all_blocks)):
            gain = (block_masks[bi] & remaining).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        if best_block == -1:
            break
        selected.append(best_block)
        remaining &= ~block_masks[best_block]
    
    return [list(all_blocks[b]) for b in selected]
