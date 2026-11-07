"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_n = P['n']
_adj_masks = [0] * _n
for _u, _v in P['edges']:
    _adj_masks[_u] |= (1 << _v)
    _adj_masks[_v] |= (1 << _u)

# Precompute neighbors list for faster iteration? 
# Or just use masks.

best_count = 0
best_mask = 0

def construct():
    global best_count, best_mask
    best_count = 0
    best_mask = 0
    
    n = _n
    # degs[v] is the current degree of v in the induced subgraph defined by mask
    # We only care about degs[v] if v is in mask.
    degs = [0] * n
    # freq[d] is the count of vertices in mask with degree d
    # Max degree is n-1 = 16
    freq = [0] * n
    
    def dfs(idx, mask, size, distinct_count):
        global best_count, best_mask
        
        # Pruning
        # Max possible distinct degrees in final set is at most final size.
        # Final size <= size + (n - idx)
        upper = size + (n - idx)
        if upper <= best_count:
            return
            
        if idx == n:
            if size >= 8 and distinct_count > best_count:
                best_count = distinct_count
                best_mask = mask
            return
        
        # Option 1: Exclude vertex idx
        dfs(idx + 1, mask, size, distinct_count)
        
        # Option 2: Include vertex idx
        # Calculate new degree of idx
        # degs[idx] = count of neighbors of idx in mask
        new_deg = (_adj_masks[idx] & mask).bit_count()
        
        # Update degrees of existing vertices in mask that are neighbors of idx
        # We need to iterate over neighbors of idx that are in mask
        neighbors_mask = _adj_masks[idx] & mask
        # Iterate through set bits in neighbors_mask
        # To avoid modifying degs/freq while iterating, we collect changes?
        # Or just iterate and update.
        
        # We must save old state to backtrack
        # Since we are in a recursive function, we can just modify and then revert.
        
        # Revert list for neighbors
        changed_vertices = []
        
        # Iterate through neighbors
        m = neighbors_mask
        while m:
            # Extract lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            m -= lsb
            
            old_deg = degs[v]
            new_v_deg = old_deg + 1
            
            # Update freq
            freq[old_deg] -= 1
            if freq[old_deg] == 0:
                distinct_count -= 1
            freq[new_v_deg] += 1
            if freq[new_v_deg] == 1:
                distinct_count += 1
            
            degs[v] = new_v_deg
            changed_vertices.append((v, old_deg))
            
        # Add idx to mask
        degs[idx] = new_deg
        freq[new_deg] += 1
        if freq[new_deg] == 1:
            distinct_count += 1
            
        new_mask = mask | (1 << idx)
        new_size = size + 1
        
        dfs(idx + 1, new_mask, new_size, distinct_count)
        
        # Backtrack
        # Revert idx
        freq[new_deg] -= 1
        if freq[new_deg] == 0:
            distinct_count -= 1 # Not used after return, but good for consistency
        degs[idx] = 0
        
        # Revert neighbors
        for v, old_deg in changed_vertices:
            curr_deg = degs[v]
            freq[curr_deg] -= 1
            if freq[curr_deg] == 0:
                distinct_count -= 1
            freq[old_deg] += 1
            if freq[old_deg] == 1:
                distinct_count += 1
            degs[v] = old_deg

    dfs(0, 0, 0, 0)
    
    if best_count == 0:
        return list(range(8))
        
    return [v for v in range(n) if (best_mask >> v) & 1]

# EVOLVE-BLOCK-END
