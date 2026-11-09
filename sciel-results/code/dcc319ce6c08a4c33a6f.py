# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 16
    k = 4
    all_sets = list(combinations(range(1, n + 1), k))
    total = len(all_sets)

    # Convert to bitmask representation
    all_bits = [0] * total
    for i, s in enumerate(all_sets):
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        all_bits[i] = mask

    def is_sunflower_triple(a, b, c):
        """Check if three sets form a sunflower: A∩B == A∩C == B∩C"""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc

    def can_add_to_family(c_bit, family_bits):
        """Check if c_bit can be added to family_bits without forming a sunflower."""
        for i in range(len(family_bits)):
            a = family_bits[i]
            for j in range(i + 1, len(family_bits)):
                b = family_bits[j]
                if is_sunflower_triple(a, b, c_bit):
                    return False
        return True

    def is_valid_family(family_bits):
        """Check if the entire family is valid (no sunflower triples)."""
        for i in range(len(family_bits)):
            a = family_bits[i]
            for j in range(i + 1, len(family_bits)):
                b = family_bits[j]
                for k in range(j + 1, len(family_bits)):
                    c = family_bits[k]
                    if is_sunflower_triple(a, b, c):
                        return False
        return True

    best_family = []
    best_size = 0

    def greedy_once(time_limit):
        """One greedy pass with random ordering."""
        order = list(range(total))
        random.shuffle(order)
        
        family = []
        family_bits = []
        
        for idx in order:
            if time.time() > time_limit:
                break
            c_bit = all_bits[idx]
            if can_add_to_family(c_bit, family_bits):
                family.append(idx)
                family_bits.append(c_bit)
        
        return family, family_bits

    # Run multiple greedy passes
    while time.time() < deadline - 20:
        family, family_bits = greedy_once(deadline - 3)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            if best_size >= 200:
                break

    # Local search: try swaps to improve
    if best_size > 0 and time.time() < deadline - 5:
        current_family = best_family[:]
        current_set = set(current_family)
        current_bits = [all_bits[i] for i in current_family]
        
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            candidates = [i for i in range(total) if i not in current_set]
            random.shuffle(candidates)
            
            # Try up to 100 candidate additions with removals
            for new_idx in candidates[:100]:
                if time.time() > deadline - 3:
                    break
                new_bit = all_bits[new_idx]
                
                # Try removing each element and adding new_idx
                for remove_pos in range(len(current_family)):
                    # Build test family
                    test_family = current_family[:remove_pos] + current_family[remove_pos+1:] + [new_idx]
                    test_bits = [all_bits[i] for i in test_family]
                    
                    if is_valid_family(test_bits):
                        # Perform swap
                        current_family = test_family
                        current_set = set(current_family)
                        current_bits = test_bits
                        improved = True
                        break
                if improved:
                    break
    
    return [sorted(all_sets[i]) for i in best_family]

# EVOLVE-BLOCK-END
