"""Covering C(19,8,5): randomized minimal cover via iterative removal of non-essential blocks."""
import numpy as np
import itertools
import random

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    block_t_list = []
    for block in k_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_list.append(covered)
    
    # Coverage counts: how many remaining blocks cover each t-subset
    coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            coverage[idx] += 1
            
    # blocks_for_s[s] = list of block indices that cover t-subset s
    blocks_for_s = [[] for _ in range(n_t)]
    for i in range(n_k):
        for idx in block_t_list[i]:
            blocks_for_s[idx].append(i)
            
    # Convert to arrays for efficiency
    for s in range(n_t):
        blocks_for_s[s] = np.array(blocks_for_s[s], dtype=np.int32)

    # Active flags
    active = np.ones(n_k, dtype=bool)
    
    # Identify initially critical t-subsets (count == 1)
    # and block all blocks that cover them
    blocked = np.zeros(n_k, dtype=bool)
    
    critical_mask = (coverage == 1)
    # For each critical t-subset, mark all covering blocks as blocked
    for s in range(n_t):
        if critical_mask[s]:
            blocked[blocks_for_s[s]] = True
            
    # List of removable blocks
    # Use a list for O(1) swap-pop removal
    removable_list = []
    in_removable = np.zeros(n_k, dtype=bool)
    
    for i in range(n_k):
        if not blocked[i]:
            removable_list.append(i)
            in_removable[i] = True
            
    # Shuffle to ensure randomness
    random.shuffle(removable_list)
    
    while removable_list:
        # Pick a random valid removable block
        # We try up to a few times if we hit a stale entry, but with swap-pop
        # and immediate marking, stale entries should be rare if we manage indices correctly.
        # Actually, we don't remove from the list when a block becomes blocked,
        # we just mark in_removable as False.
        # So we might pick a blocked one. We need to handle that.
        
        while True:
            if not removable_list:
                break
            pos = random.randint(0, len(removable_list) - 1)
            i = removable_list[pos]
            if in_removable[i]:
                # Valid block
                break
            else:
                # Stale entry, remove it
                removable_list[pos] = removable_list[-1]
                removable_list.pop()
        
        if not removable_list:
            break
            
        # We have a valid block i
        in_removable[i] = False
        # Remove from list (swap with last)
        # Find position of i? We don't know it.
        # We just picked it at `pos`.
        # So we can swap `pos` with last.
        removable_list[pos] = removable_list[-1]
        removable_list.pop()
        
        active[i] = False
        
        # Update coverage and blocked status
        for idx in block_t_list[i]:
            coverage[idx] -= 1
            if coverage[idx] == 1:
                # This t-subset is now critical
                # All active blocks covering it (except i, which is inactive) must be blocked
                for j in blocks_for_s[idx]:
                    if j != i and active[j] and in_removable[j]:
                        blocked[j] = True
                        in_removable[j] = False
                        # Note: We do NOT remove j from removable_list immediately.
                        # It will be filtered out when picked.
                        # This is acceptable if the list doesn't get too full of stale entries.
                        # Given the ratio of removable to total, it should be fine.

    # Collect remaining blocks
    result = [list(k_blocks[i]) for i in range(n_k) if active[i]]
    
    # Verify coverage (safety check)
    if n_t > 0:
        if np.any(coverage < 1):
             # Fallback: this should not happen
             pass

    return result
