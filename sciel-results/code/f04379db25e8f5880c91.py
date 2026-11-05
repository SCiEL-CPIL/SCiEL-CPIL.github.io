"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)  # 127 moduli
    
    # Precompute gcd for all pairs
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # Precompute incompatible residue masks
    # For each modulus m2, for each divisor g of m2, and each rem in 0..g-1:
    #   mask_incompat[m2][g][rem] = bitset of residues r2 in 0..m2-1 with r2 % g == rem
    mask_incompat = {}
    for m2 in moduli:
        mask_incompat[m2] = {}
        for g in range(1, m2 + 1):
            if m2 % g != 0:
                continue
            mask_incompat[m2][g] = {}
            for rem in range(g):
                mask = 0
                for r2 in range(m2):
                    if r2 % g == rem:
                        mask |= (1 << r2)
                mask_incompat[m2][g][rem] = mask
    
    # Greedy to get initial lower bound
    def greedy_lower_bound():
        assignment = {}
        available = {m: (1 << m) - 1 for m in moduli}
        
        # Order: most constrained first (moduli with many large gcds)
        order = sorted(moduli, key=lambda m: -sum(gcd_table[(m, m2)] for m2 in moduli if m2 != m))
        
        for m in order:
            mask = available[m]
            if mask == 0:
                continue
            # Try each residue, pick the one that causes least damage
            best_r = None
            best_damage = float('inf')
            tmp = mask
            while tmp:
                lsb = tmp & (-tmp)
                r = lsb.bit_length() - 1
                damage = 0
                for m2 in order:
                    if m2 == m:
                        continue
                    mask2 = available[m2]
                    if mask2 == 0:
                        continue
                    g = gcd_table[(m, m2)]
                    rem = r % g
                    incompat = mask_incompat[m2][g][rem]
                    removed = mask2 & incompat
                    damage += bin(removed).count('1')
                if damage < best_damage:
                    best_damage = damage
                    best_r = r
                tmp &= tmp - 1
            
            if best_r is not None:
                assignment[m] = best_r
                # Update available for others
                for m2 in order:
                    if m2 == m:
                        continue
                    mask2 = available[m2]
                    if mask2 == 0:
                        continue
                    g = gcd_table[(m, m2)]
                    rem = best_r % g
                    incompat = mask_incompat[m2][g][rem]
                    available[m2] = mask2 & ~incompat
        
        return assignment
    
    best_assign = greedy_lower_bound()
    best_size = len(best_assign)
    best_solution = [(m, best_assign[m]) for m in best_assign]
    
    # Branch and bound
    start_time = time.time()
    TIMEOUT = 55  # seconds, leave margin for 60s limit
    
    # We'll use iterative deepening on the target size
    # Or just B&B with pruning
    
    # For B&B, we maintain:
    # - current list of assigned (m, r)
    # - available masks for each modulus
    # - set of moduli already processed
    
    # Process moduli in a fixed order (most constrained first)
    order = sorted(moduli, key=lambda m: -sum(gcd_table[(m, m2)] for m2 in moduli if m2 != m))
    order_index = {m: i for i, m in enumerate(order)}
    
    # Precompute for each position in order, the list of moduli that come after
    # For fast upper bound: count of remaining moduli with non-zero mask
    
    def bb(assignment, masks, idx, current_count):
        nonlocal best_size, best_solution, start_time
        
        # Check timeout
        if time.time() - start_time > TIMEOUT:
            return
        
        if idx >= n:
            if current_count > best_size:
                best_size = current_count
                best_solution = list(assignment)
            return
        
        # Upper bound: current_count + number of moduli from idx onward with non-zero mask
        ub = current_count
        for j in range(idx, n):
            m2 = order[j]
            if masks[m2] != 0:
                ub += 1
        if ub <= best_size:
            return
        
        m = order[idx]
        mask = masks[m]
        
        if mask == 0:
            # Skip this modulus
            bb(assignment, masks, idx + 1, current_count)
            return
        
        # Option 1: Skip modulus m
        bb(assignment, masks, idx + 1, current_count)
        
        # Option 2: Try each residue
        tmp = mask
        while tmp:
            lsb = tmp & (-tmp)
            r = lsb.bit_length() - 1
            tmp &= tmp - 1
            
            # Assign r to m
            new_masks = dict(masks)
            new_masks[m] = 0
            
            # Update masks for future moduli
            feasible = True
            for j in range(idx + 1, n):
                m2 = order[j]
                mask2 = new_masks[m2]
                if mask2 == 0:
                    continue
                g = gcd_table[(m, m2)]
                rem = r % g
                incompat = mask_incompat[m2][g][rem]
                new_masks[m2] = mask2 & ~incompat
            
            bb(assignment + [(m, r)], new_masks, idx + 1, current_count + 1)
    
    # Initial masks
    init_masks = {m: (1 << m) - 1 for m in moduli}
    
    # Run B&B
    bb([], init_masks, 0, 0)
    
    # Convert best_solution to required format
    result = [[r, m] for m, r in best_solution]
    
    # Ensure nonempty
    if not result:
        # Fallback: just pick one modulus
        result = [[0, 2]]
    
    return result

# EVOLVE-BLOCK-END
