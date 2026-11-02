"""Improved covering design C(19,8,5) using bitsets, greedy + local search."""
import itertools
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # For each point, precompute which t-subsets it participates in
    point_t_subs = defaultdict(list)
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_t_subs[p].append(i)
    
    # For each pair of points, precompute t-subsets containing both
    # Actually, let's precompute for each (t-1)-subset, which points complete it to a t-subset
    # And for each point x and (t-1)-subset s, check if s∪{x} is a t-subset
    
    # For block construction: given a partial block B, adding point x covers
    # t-subsets that contain x and (t-1) points from B
    # Precompute: for each (t-1)-subset, the list of points that complete it
    t_minus_1_subs = list(itertools.combinations(range(v), t - 1))
    t1_sub_index = {sub: i for i, sub in enumerate(t_minus_1_subs)}
    
    # For each (t-1)-subset s, points that with s form a t-subset
    t1_complete = {}
    for s in t_minus_1_subs:
        pts = []
        for p in range(v):
            if p not in s:
                full = tuple(sorted(s + (p,)))
                if full in t_sub_index:
                    pts.append(p)
        t1_complete[s] = pts
    
    # Uncovered t-subsets as a bitset
    uncovered = np.ones(num_t, dtype=np.uint8)
    blocks = []
    
    def block_covers_indices(block):
        """Return list of t-subset indices covered by this block."""
        idxs = []
        for sub in itertools.combinations(block, t):
            idxs.append(t_sub_index[tuple(sorted(sub))])
        return idxs
    
    def try_add_block(block):
        """Add block, update uncovered."""
        nonlocal uncovered
        for idx in block_covers_indices(block):
            uncovered[idx] = 0
        blocks.append(sorted(block))
    
    # Phase 1: Greedy construction
    while np.any(uncovered):
        if time.time() > deadline - 30:
            break
        
        # Find the most constrained uncovered t-subset (fewest candidate blocks)
        # For simplicity, pick first uncovered
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        
        # Pick an uncovered t-subset to start a new block
        # Choose one that appears in fewest potential blocks (heuristic)
        # For now, just pick the first
        start_sub = all_t_subs[unc_idx[0]]
        block = list(start_sub)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        remaining_k = k - len(block)
        for _ in range(remaining_k):
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many new uncovered t-subsets this point would cover
                # A t-subset is covered if it contains x and (t-1) points from block
                gain = 0
                for s in itertools.combinations(block, t - 1):
                    full = tuple(sorted(s + (x,)))
                    if full in t_sub_index:
                        idx = t_sub_index[full]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is not None and best_gain > 0:
                block.append(best_point)
            elif best_point is not None:
                block.append(best_point)
            else:
                # Just add any available point
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
        
        block.sort()
        try_add_block(block)
    
    # Phase 2: Local search to reduce block count
    if time.time() < deadline - 10 and blocks:
        # Try to remove blocks that are redundant
        # Try to swap blocks for better coverage
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try removing each block and checking if coverage is still complete
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                # Temporarily remove block i
                removed = blocks.pop(i)
                # Check if all t-subsets are still covered
                still_covered = np.ones(num_t, dtype=np.uint8)
                for b in blocks:
                    for idx in block_covers_indices(b):
                        still_covered[idx] = 0
                
                if not np.any(still_covered):
                    # Block was redundant, keep it removed
                    improved = True
                    break
                else:
                    # Restore
                    blocks.insert(i, removed)
            
            # Try swapping: replace one block with another that covers more uncovered
            if not improved:
                for i in range(len(blocks)):
                    if time.time() > deadline - 5:
                        break
                    # Compute what block i covers
                    covered_by_i = set(block_covers_indices(blocks[i]))
                    
                    # Try to find a replacement block that covers a superset
                    # or at least covers the same and allows other improvements
                    # For simplicity, try random replacements
                    # Actually, let's try: for each uncovered t-subset, see if we can
                    # modify block i to cover it while maintaining size k
                    
                    # Try adding one point and removing one to cover an uncovered subset
                    unc_idx = np.where(uncovered)[0]
                    for _ in range(min(20, len(unc_idx))):
                        if time.time() > deadline - 5:
                            break
                        target_idx = unc_idx[np.random.randint(len(unc_idx))]
                        target_sub = all_t_subs[target_idx]
                        
                        # Try to modify blocks[i] to include target_sub
                        current = set(blocks[i])
                        target_set = set(target_sub)
                        
                        # Points to add: in target but not in current
                        to_add = target_set - current
                        # Points to remove: in current but not in target
                        to_remove = current - target_set
                        
                        if len(to_add) <= len(to_remove) and len(to_add) <= 1:
                            new_block = list(current)
                            for p in to_add:
                                new_block.append(p)
                            for p in to_remove:
                                new_block.remove(p)
                            if len(new_block) == k and len(set(new_block)) == k:
                                new_block.sort()
                                # Check if this new block covers more uncovered
                                new_covered = set(block_covers_indices(new_block))
                                if new_covered - covered_by_i:
                                    blocks[i] = new_block
                                    improved = True
                                    break
            
            # Update uncovered after any changes
            uncovered = np.ones(num_t, dtype=np.uint8)
            for b in blocks:
                for idx in block_covers_indices(b):
                    uncovered[idx] = 0
    
    # Final verification and return
    # Verify all t-subsets are covered
    final_uncovered = np.ones(num_t, dtype=np.uint8)
    for b in blocks:
        for idx in block_covers_indices(b):
            final_uncovered[idx] = 0
    
    # If not all covered, do emergency greedy
    if np.any(final_uncovered) and time.time() < deadline - 2:
        while np.any(final_uncovered):
            unc_idx = np.where(final_uncovered)[0]
            if len(unc_idx) == 0:
                break
            start_sub = all_t_subs[unc_idx[0]]
            block = list(start_sub)
            for _ in range(k - len(block)):
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for s in itertools.combinations(block, t - 1):
                        full = tuple(sorted(s + (x,)))
                        if full in t_sub_index:
                            idx = t_sub_index[full]
                            if final_uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is not None:
                    block.append(best_point)
                else:
                    for x in range(v):
                        if x not in block:
                            block.append(x)
                            break
            block.sort()
            blocks.append(block)
            for idx in block_covers_indices(block):
                final_uncovered[idx] = 0
    
    return blocks
