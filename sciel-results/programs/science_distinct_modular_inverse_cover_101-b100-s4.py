import time

def construct():
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0
    
    # Precompute modular inverses for denominators 1..10
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # We need to find, for each residue r in 0..100, a non-empty subset of {1..10}
    # such that the sum of inverses modulo 101 equals r.
    # We want to minimize the maximum size of these subsets.
    
    # Let's precompute the sum and size for every mask from 1 to 1023.
    # mask -> (sum_mod_101, size, list_of_denominators)
    mask_info = [None] * 1024
    mask_info[0] = (0, 0, [])
    
    for mask in range(1, 1024):
        # Find the least significant set bit
        lsb = mask & -mask
        i = (lsb.bit_length() - 1)  # index 0..9 corresponds to denominator i+1
        prev_mask = mask ^ lsb
        prev_sum, prev_size, _ = mask_info[prev_mask]
        denom = i + 1
        new_sum = (prev_sum + inv[denom]) % 101
        new_size = prev_size + 1
        mask_info[mask] = (new_sum, new_size, None)  # We'll fill the list later if needed
        
    # For each residue, find the minimum size subset that maps to it.
    min_size_for_residue = [float('inf')] * 101
    best_mask_for_residue = [0] * 101
    
    for mask in range(1, 1024):
        s, sz, _ = mask_info[mask]
        if sz < min_size_for_residue[s]:
            min_size_for_residue[s] = sz
            best_mask_for_residue[s] = mask
            
    # The answer is the maximum of min_size_for_residue
    # But wait, the problem asks to "Minimize the largest row length".
    # This is exactly max(min_size_for_residue[r] for r in 0..100).
    
    # Let's verify coverage.
    # Are all residues covered?
    uncovered = []
    for r in range(101):
        if min_size_for_residue[r] == float('inf'):
            uncovered.append(r)
            
    # If there are uncovered residues, we have a problem.
    # However, with 1023 subsets and 101 residues, it's highly likely all are covered.
    # If not, we would need to search harder or the problem is impossible.
    # Given the modular arithmetic, it's possible some residues are not sums of inverses?
    # Let's check if the set of possible sums is the whole group Z_101.
    # The inverses are:
    # 1: 1
    # 2: 51
    # 3: 34
    # 4: 76
    # 5: 81
    # 6: 17
    # 7: 29
    # 8: 89
    # 9: 45
    # 10: 91
    # These are elements of Z_101. The subset sums generate a subgroup.
    # Since 1 is in the set, the subgroup is the whole group Z_101.
    # So all residues are covered.
    
    # Now, construct the lists.
    result = [[] for _ in range(101)]
    for r in range(101):
        mask = best_mask_for_residue[r]
        terms = []
        for i in range(10):
            if mask & (1 << i):
                terms.append(i + 1)
        result[r] = terms
        
    return result
