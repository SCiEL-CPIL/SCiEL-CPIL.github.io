"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

def construct():
    sys.setrecursionlimit(10000)
    
    N = 256
    # weights[i] corresponds to number i+1
    weights = [1.0 / (i + 1) for i in range(N)]
    
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            # Check if a, b form a 3-AP with some c in [1, 256]
            conflict = False
            
            # Case 1: 2a = b + c => c = 2a - b
            c = 2 * a - b
            if 1 <= c <= 256:
                conflict = True
                
            # Case 2: 2b = a + c => c = 2b - a
            if not conflict:
                c = 2 * b - a
                if 1 <= c <= 256:
                    conflict = True
                    
            # Case 3: 2c = a + b => c = (a + b) / 2
            if not conflict:
                if (a + b) % 2 == 0:
                    c = (a + b) // 2
                    if 1 <= c <= 256:
                        conflict = True
            
            if conflict:
                conflict_mask[i] |= (1 << j)
    
    # Greedy to find initial best
    def greedy():
        selected = []
        avail = (1 << N) - 1
        for i in range(N):
            if (avail >> i) & 1:
                selected.append(i)
                # Remove i and its conflicts from avail
                avail &= ~(conflict_mask[i] | (1 << i))
        s = 0.0
        for i in selected:
            s += weights[i]
        return s, selected

    best_val, best_set_indices = greedy()
    # Convert indices to 1-based values for return
    best_set = [i + 1 for i in best_set_indices]
    
    def dfs2(idx, available_mask, current_val, avail_sum, chosen_mask):
        nonlocal best_val, best_set
        
        # Pruning
        if current_val + avail_sum <= best_val:
            return
            
        if idx == N:
            if current_val > best_val:
                best_val = current_val
                # Convert chosen_mask to list of 1-based integers
                lst = []
                m = chosen_mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    lst.append(idx_bit + 1)
                    m ^= lsb
                best_set = lst
            return
        
        # If idx is not available, we must exclude it
        if not ((available_mask >> idx) & 1):
            dfs2(idx + 1, available_mask, current_val, avail_sum, chosen_mask)
            return
        
        # Option 1: Include idx (Heuristic: try include first for high weights)
        remove_mask = available_mask & (conflict_mask[idx] | (1 << idx))
        
        # Calculate sum of removed weights
        removed_sum = 0.0
        rm = remove_mask
        while rm:
            lsb = rm & -rm
            idx_rm = lsb.bit_length() - 1
            removed_sum += weights[idx_rm]
            rm ^= lsb
            
        new_avail_mask = available_mask & ~remove_mask
        new_avail_sum = avail_sum - removed_sum
        new_chosen_mask = chosen_mask | (1 << idx)
        new_current_val = current_val + weights[idx]
        
        dfs2(idx + 1, new_avail_mask, new_current_val, new_avail_sum, new_chosen_mask)
        
        # Option 2: Exclude idx
        new_avail_mask_ex = available_mask & ~(1 << idx)
        new_avail_sum_ex = avail_sum - weights[idx]
        
        dfs2(idx + 1, new_avail_mask_ex, current_val, new_avail_sum_ex, chosen_mask)

    initial_mask = (1 << N) - 1
    initial_sum = sum(weights)
    
    dfs2(0, initial_mask, 0.0, initial_sum, 0)
    
    return best_set
# EVOLVE-BLOCK-END
