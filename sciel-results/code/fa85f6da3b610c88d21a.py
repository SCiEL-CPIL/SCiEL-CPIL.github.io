"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import sys

def construct():
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * (MAX_M + 1) for _ in range(MAX_M + 1)]
    for i in range(2, MAX_M + 1):
        for j in range(2, MAX_M + 1):
            gcd_table[i][j] = math.gcd(i, j)
            
    # Precompute incompatibility masks
    mask_incompat = {}
    for m in moduli:
        mask_incompat[m] = {}
        divisors = [g for g in range(1, m + 1) if m % g == 0]
        for g in divisors:
            mask_incompat[m][g] = [0] * g
            for rem in range(g):
                mask = 0
                for r in range(m):
                    if r % g == rem:
                        mask |= (1 << r)
                mask_incompat[m][g][rem] = mask

    # Determine processing order
    scores = {}
    for m in moduli:
        scores[m] = sum(gcd_table[m][m2] for m2 in moduli if m2 != m)
    
    order = sorted(moduli, key=lambda m: -scores[m])
    
    # Initial masks
    init_masks = [ (1 << m) - 1 for m in order ]
    
    # Precompute clear masks
    clear_masks = [None] * n
    for i in range(n):
        m_i = order[i]
        clear_masks[i] = [None] * m_i
        for r in range(m_i):
            effects = []
            for j in range(i + 1, n):
                m_j = order[j]
                g = gcd_table[m_i][m_j]
                rem = r % g
                if g in mask_incompat[m_j]:
                    cm = mask_incompat[m_j][g][rem]
                    effects.append((j, cm))
            clear_masks[i][r] = effects

    best_size = 0
    best_solution = []
    current_path = []
    
    def solve(idx, masks, current_count, remaining_count):
        nonlocal best_size, best_solution
        
        if current_count + remaining_count <= best_size:
            return
            
        if idx == n:
            if current_count > best_size:
                best_size = current_count
                best_solution = list(current_path)
            return
        
        m = order[idx]
        mask = masks[idx]
        
        if mask == 0:
            # If mask is 0, this modulus cannot be in the set.
            # remaining_count already excludes this modulus (as it's 0).
            # So we pass remaining_count unchanged.
            solve(idx + 1, masks, current_count, remaining_count)
            return
        
        # Try each residue in mask
        r = 0
        while mask:
            lsb = mask & (-mask)
            r = lsb.bit_length() - 1
            mask_without_lsb = mask & (mask - 1)
            
            current_path.append((m, r))
            
            new_masks = masks[:]
            # remaining_count includes current mask (which is non-zero).
            # So initial new_remaining is remaining_count - 1.
            new_remaining = remaining_count - 1
            
            effects = clear_masks[idx][r]
            for j, cm in effects:
                if new_masks[j] != 0:
                    new_masks[j] &= ~cm
                    if new_masks[j] == 0:
                        new_remaining -= 1
            
            solve(idx + 1, new_masks, current_count + 1, new_remaining)
            
            current_path.pop()
            
            mask = mask_without_lsb

    solve(0, init_masks, 0, n)
    
    if not best_solution:
        return [[0, 2]]
        
    result = [[r, m] for m, r in best_solution]
    return result

# EVOLVE-BLOCK-END
