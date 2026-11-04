"""Covering C(19,8,5): optimized iterative removal of non-essential blocks."""
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
        
    # Coverage counts
    coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            coverage[idx] += 1
            
    # blocks_for_s[s] = list of block indices that cover t-subset s
    blocks_for_s = [[] for _ in range(n_t)]
    for i in range(n_k):
        for idx in block_t_list[i]:
            blocks_for_s[idx].append(i)
            
    # Convert to numpy arrays for faster iteration? No, lists of ints are fine for small lists.
    # But we need to check if a block is active.
    
    active = np.ones(n_k, dtype=bool)
    blocked = np.zeros(n_k, dtype=bool)
    
    # Initialize removable list
    removable = []
    for i in range(n_k):
        # A block is removable if it doesn't cover any critical t-subset?
        # No, a block is removable if removing it doesn't leave any t-subset uncovered.
        # i.e., for all t in block, coverage[t] > 1.
        # If coverage[t] == 1 for any t in block, the block is "essential" (blocked).
        is_removable = True
        for idx in block_t_list[i]:
            if coverage[idx] == 1:
                is_removable = False
                break
        if is_removable:
            removable.append(i)
            
    random.shuffle(removable)
    
    # Map block index to its position in removable list for O(1) removal
    # This is tricky with swap-pop because positions change.
    # Alternative: Use a set for O(1) lookup, but we need to iterate.
    # Or, just use a list and mark as "dead" and clean up periodically?
    # Given the size, let's try to maintain a list and a dictionary of positions.
    # Actually, simpler: Just use a list and when we remove an item, we swap with last.
    # We need to know where `i` is.
    
    # Let's use a different strategy:
    # Maintain a list `removable`.
    # When a block `j` becomes blocked, we need to remove it from `removable`.
    # If we don't remove it immediately, we might pick it later.
    # We can use a `valid` flag array.
    
    valid_in_removable = np.zeros(n_k, dtype=bool)
    for i in removable:
        valid_in_removable[i] = True
        
    while removable:
        # Pick a random valid block
        # We might have stale entries.
        while True:
            if not removable:
                break
            pos = random.randint(0, len(removable) - 1)
            i = removable[pos]
            if valid_in_removable[i]:
                break
            else:
                # Swap and pop stale entry
                removable[pos] = removable[-1]
                removable.pop()
                
        if not removable:
            break
            
        # Remove i from removable
        valid_in_removable[i] = False
        removable[pos] = removable[-1]
        removable.pop()
        
        active[i] = False
        
        # Update coverage
        for idx in block_t_list[i]:
            coverage[idx] -= 1
            if coverage[idx] == 1:
                # This t-subset is now critical.
                # All other active blocks covering it must be blocked.
                for j in blocks_for_s[idx]:
                    if j != i and active[j] and valid_in_removable[j]:
                        # j becomes blocked
                        blocked[j] = True
                        valid_in_removable[j] = False
                        # Note: j is still in `removable` list, but marked invalid.
                        # It will be cleaned up when picked.
                        
    # Collect remaining blocks
    result = [list(k_blocks[i]) for i in range(n_k) if active[i]]
    
    return result
