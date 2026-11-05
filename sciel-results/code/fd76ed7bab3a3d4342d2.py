"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0]*n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(n_moduli):
            gcd_table[i][j] = math.gcd(moduli[i], moduli[j])
            
    # Precompute incompatibility masks
    # masks[i][j][r_i] is a bitset of residues r_j in modulus j 
    # that are incompatible with residue r_i in modulus i.
    
    masks = [[[0]*moduli[i] for _ in range(n_moduli)] for i in range(n_moduli)]
    
    for i in range(n_moduli):
        m_i = moduli[i]
        for j in range(n_moduli):
            if i == j:
                continue
            m_j = moduli[j]
            g = gcd_table[i][j]
            for r_i in range(m_i):
                rem = r_i % g
                mask = 0
                for r_j in range(m_j):
                    if r_j % g == rem:
                        mask |= (1 << r_j)
                masks[i][j][r_i] = mask
                
    # Greedy initialization to get a good lower bound
    def greedy_lower_bound():
        available = [(1 << moduli[i]) - 1 for i in range(n_moduli)]
        chosen = []
        # Sort moduli by number of residues (ascending) to pick constrained ones first?
        # Or just iterate.
        # Let's try picking the modulus with the fewest available residues at each step.
        active = set(range(n_moduli))
        while active:
            min_count = float('inf')
            min_idx = -1
            for i in active:
                if available[i] != 0:
                    c = bin(available[i]).count("1")
                    if c < min_count:
                        min_count = c
                        min_idx = i
            if min_idx == -1:
                break
            
            # Pick a residue. Let's pick the one that removes the fewest options from others.
            m_i = moduli[min_idx]
            best_r = -1
            best_cost = float('inf')
            mask = available[min_idx]
            while mask:
                lsb = mask & (-mask)
                r = lsb.bit_length() - 1
                mask_without = mask & (mask - 1)
                
                cost = 0
                for j in active:
                    if j == min_idx or available[j] == 0:
                        continue
                    incompat = masks[min_idx][j][r]
                    # Count how many bits are removed
                    removed = available[j] & incompat
                    cost += bin(removed).count("1")
                
                if cost < best_cost:
                    best_cost = cost
                    best_r = r
                mask = mask_without
            
            if best_r == -1:
                break
                
            chosen.append((min_idx, best_r))
            available[min_idx] = 0
            active.remove(min_idx)
            
            for j in active:
                if available[j] == 0:
                    continue
                incompat = masks[min_idx][j][best_r]
                available[j] &= ~incompat
                if available[j] == 0:
                    # This modulus is now blocked, but we keep it in active with 0 mask
                    # so it won't be picked.
                    pass
                    
        return len(chosen), chosen

    initial_size, initial_sol = greedy_lower_bound()
    best_size = initial_size
    best_solution = initial_sol
    
    def solve(available, current_size, chosen):
        nonlocal best_size, best_solution
        
        # Upper bound
        ub = current_size
        for i in range(n_moduli):
            if available[i] != 0:
                ub += 1
                
        if ub <= best_size:
            return
            
        # Find MRV
        min_count = float('inf')
        min_idx = -1
        for i in range(n_moduli):
            if available[i] != 0:
                c = bin(available[i]).count("1")
                if c < min_count:
                    min_count = c
                    min_idx = i
                    
        if min_idx == -1:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(chosen)
            return
            
        m_i = moduli[min_idx]
        mask = available[min_idx]
        
        # Branch
        while mask:
            lsb = mask & (-mask)
            r = lsb.bit_length() - 1
            mask_without = mask & (mask - 1)
            
            new_available = list(available)
            new_available[min_idx] = 0
            
            for j in range(n_moduli):
                if j == min_idx or new_available[j] == 0:
                    continue
                incompat = masks[min_idx][j][r]
                new_available[j] &= ~incompat
                
            chosen.append((min_idx, r))
            solve(new_available, current_size + 1, chosen)
            chosen.pop()
            
            mask = mask_without

    init_available = [(1 << moduli[i]) - 1 for i in range(n_moduli)]
    solve(init_available, 0, [])
    
    result = []
    for idx, r in best_solution:
        m = moduli[idx]
        result.append([r, m])
        
    return result

# EVOLVE-BLOCK-END
