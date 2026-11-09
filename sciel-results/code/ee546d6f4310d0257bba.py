import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mods = len(moduli)
    
    # Precompute gcd matrix: gcd[i][j] = gcd(moduli[i], moduli[j])
    # Index i corresponds to moduli[i]
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_mods):
            gcd_mat[i][j] = gcd(moduli[i], moduli[j])
    
    # For each modulus i, and for each existing modulus j, 
    # the forbidden residues for i given j are those r where r % gcd(i,j) == r_j % gcd(i,j)
    # We need to find a residue r_i in [0, m_i-1] that is NOT forbidden by any existing j.
    
    # Strategy: Branch and Bound / Backtracking with good ordering
    # We want to maximize the number of moduli included.
    # We can model this as a Maximum Independent Set on a conflict graph?
    # No, the conflict depends on the specific residues chosen.
    # However, if gcd(m_i, m_j) == 1, they are always disjoint.
    # If gcd(m_i, m_j) > 1, they are disjoint iff r_i % g != r_j % g.
    
    # Let's use a heuristic search.
    # Order moduli by "difficulty". Moduli with many small prime factors (high gcd with many others) might be harder?
    # Or maybe moduli that are powers of 2?
    # Let's try sorting by number of divisors or just value.
    
    # A strong heuristic:
    # 1. Try to include all moduli that are coprime to everything else? No, that's not right.
    # 2. We can view this as assigning residues.
    
    # Let's try a randomized local search or simulated annealing style approach.
    # Or a simple greedy with multiple restarts and different orderings.
    
    # Actually, let's look at the structure.
    # If we pick moduli that are all powers of 2, say 2, 4, 8, ..., 128.
    # gcd(2,4)=2. r_2 % 2 != r_4 % 2.
    # r_2 can be 0 or 1. r_4 can be 0,1,2,3.
    # If r_2=0, r_4 cannot be 0,2. Must be 1 or 3.
    # If r_4=1, r_8 cannot be 1,3,5,7. Must be 0,2,4,6.
    # It seems we can always find a residue.
    # In fact, for a chain of moduli where m_{i+1} is a multiple of m_i, we can always extend?
    # Not necessarily, but likely.
    
    # Let's try to find a large set.
    # The maximum possible size is 127 (all moduli).
    # Can we achieve 127?
    # If we can, the score is 127/seed.
    # The seed score is 2.5714. If seed value is V, V/seed = 2.5714.
    # We don't know V, but we want to maximize V.
    
    # Let's try a backtracking algorithm with bitsets for speed.
    # State: current set of (residue, modulus) pairs.
    # To add a new modulus m, we need to check if there exists r in [0, m-1]
    # such that for all existing (r_j, m_j), r % gcd(m, m_j) != r_j % gcd(m, m_j).
    
    # Optimization:
    # For a fixed m, the constraints from existing pairs define a set of forbidden residues modulo g = gcd(m, m_j).
    # The union of these forbidden sets in [0, m-1] must not cover all residues.
    
    # We can represent the forbidden residues for a candidate modulus m as a bitset of size m.
    # But m is up to 128, so bitset is 128 bits, which fits in 2 64-bit ints or just a Python int.
    
    def can_add(m, current_pairs):
        # current_pairs: list of (r_j, m_j)
        # Return (True, r) if possible, else (False, None)
        # forbidden_mask: bit r is 1 if r is forbidden
        forbidden_mask = 0
        
        for r_j, m_j in current_pairs:
            g = gcd(m, m_j)
            # r is forbidden if r % g == r_j % g
            # This is a pattern of residues.
            # We can iterate through the forbidden residues and set bits.
            # Since m <= 128, this is fast.
            r_j_mod_g = r_j % g
            for r in range(r_j_mod_g, m, g):
                forbidden_mask |= (1 << r)
        
        # Find an available residue
        # Check if forbidden_mask has all bits 0..m-1 set
        # full_mask = (1 << m) - 1
        if forbidden_mask != (1 << m) - 1:
            # Find the lowest set bit in ~forbidden_mask within range m
            available_mask = ~forbidden_mask & ((1 << m) - 1)
            # Get index of lowest set bit
            # available_mask & -available_mask gives the lowest set bit
            low_bit = available_mask & (-available_mask)
            r = low_bit.bit_length() - 1
            return (True, r)
        else:
            return (False, None)

    # Ordering of moduli
    # We want to place "hard" moduli first or "easy" ones first?
    # Usually, placing restrictive ones first is better for backtracking.
    # Moduli with many divisors share more gcds > 1 with others.
    # Let's sort by number of divisors descending.
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                count += 1
                if i != n//i:
                    count += 1
        return count
    
    moduli_sorted = sorted(moduli, key=lambda x: -num_divisors(x))
    
    # Let's try a few orderings
    orderings = [
        moduli_sorted,
        sorted(moduli, reverse=True),
        sorted(moduli),
        list(reversed(moduli_sorted)),
    ]
    
    best_size = 0
    best_result = []
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        # Greedy with backtracking limit?
        # Simple greedy first to get a baseline.
        current_pairs = []
        for m in order:
            ok, r = can_add(m, current_pairs)
            if ok:
                current_pairs.append((r, m))
        
        if len(current_pairs) > best_size:
            best_size = len(current_pairs)
            best_result = current_pairs[:]
            
        # If we got all 127, we can stop.
        if best_size == 127:
            break

    # Now, try to improve using local search / swaps?
    # Or try random restarts with the greedy approach.
    
    import random
    random.seed(42)
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Random order
        rand_order = moduli[:]
        random.shuffle(rand_order)
        
        current_pairs = []
        for m in rand_order:
            ok, r = can_add(m, current_pairs)
            if ok:
                current_pairs.append((r, m))
        
        if len(current_pairs) > best_size:
            best_size = len(current_pairs)
            best_result = current_pairs[:]
            
            if best_size == 127:
                break

    # Convert to required format
    final_result = [[r, m] for r, m in best_result]
    
    if not final_result:
        final_result = [[1, 2]]
        
    return final_result
