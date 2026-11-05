# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 60s limit (or 90s)
    
    # All 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Use bitmask representation for speed
    # Map element 1..16 to bit 0..15
    mask_map = {i: 1 << (i-1) for i in range(1, 17)}
    
    def to_mask(s):
        m = 0
        for x in s:
            m |= mask_map[x]
        return m
    
    masks = [to_mask(s) for s in all_sets]
    mask_to_set = {to_mask(s): s for s in all_sets}
    
    # Precompute pairwise intersections between all masks?
    # C(16,4) = 1820. Pairs = 1820*1819/2 ~ 1.65M.
    # Storing these might be memory heavy or not needed if we compute on the fly.
    # However, checking sunflower condition:
    # For new set m, and existing a, b:
    # (a & b) == (a & m) == (b & m)
    
    # Optimization:
    # For a fixed new set m, and a fixed existing set a:
    # We need to check if there exists b in family (b != a) such that:
    # (a & b) == (a & m) and (b & m) == (a & m)
    # Let I = a & m.
    # Conditions:
    # 1. b must contain all elements of I (since a & b = I implies I subset of b, and I subset of a).
    #    Wait, a & b = I means:
    #    - x in I => x in a and x in b
    #    - x not in I => x not in a OR x not in b
    #    Since x in I implies x in a, the second condition for x not in I is:
    #    If x in a, then x not in b.
    #    So b must contain I, and b must be disjoint from (a \ I) = (a \ (a&m)) = a \ m.
    #    So b must satisfy: I subset b subset (U \ (a \ m)) = U \ (a & ~m).
    #    Also condition 2: b & m = I.
    #    This means b must contain I, and b must be disjoint from (m \ I) = (m \ (a&m)) = m \ a.
    #    
    #    So for a given pair (a, m), the forbidden sets b are those such that:
    #    I = a & m
    #    b contains I
    #    b is disjoint from (a \ m)
    #    b is disjoint from (m \ a)
    #    Note that (a \ m) and (m \ a) are disjoint.
    #    So b must be a subset of U \ ((a \ m) U (m \ a)) = U \ (a XOR m).
    #    And b must contain I.
    #    Also b must be a 4-set (since all sets are 4-sets).
    #    
    #    So the condition "b creates a sunflower with a and m" is equivalent to:
    #    b is a 4-set such that I subset b subset U \ (a XOR m).
    #    
    #    Let's check this logic.
    #    If b satisfies this, then:
    #    a & b: Since b subset U \ (a \ m), b is disjoint from a \ m.
    #    So a & b is a subset of a & m = I.
    #    Since I subset b, a & b contains I.
    #    Thus a & b = I.
    #    Similarly, b & m: Since b subset U \ (m \ a), b is disjoint from m \ a.
    #    So b & m is a subset of m & a = I.
    #    Since I subset b, b & m contains I.
    #    Thus b & m = I.
    #    So a & b = a & m = b & m = I.
    #    Yes, this is correct.
    
    # So, to check if m can be added to family F:
    # For each a in F:
    #   I = a & m
    #   Forbidden region for b: b contains I and b is disjoint from (a XOR m).
    #   We need to check if there is any b in F (b != a) that satisfies this.
    
    # How to check this fast?
    # We can precompute for each possible 4-set b, its mask.
    # But we are checking against the current family F.
    # F is a list of masks.
    # We can maintain a set of masks in F for O(1) lookup? No, we need to check if ANY b in F satisfies the condition.
    # The condition defines a specific set of allowed b's.
    # Specifically, b must be a 4-set subset of U \ (a XOR m) that contains I.
    # Let's call this set of candidates C(a, m).
    # We need to check if F intersects C(a, m) \ {a}.
    
    # Since |F| is small (< 250), and we do this for each a in F, the total cost is |F|^2 * (cost of checking if b is in C(a,m)).
    # Checking if b is in C(a,m) takes O(1) with bitmasks:
    # b contains I: (b & I) == I
    # b disjoint from (a XOR m): (b & (a ^ m)) == 0
    
    # So the check for a specific pair (a, b) with candidate m is:
    # I = a & m
    # (b & I) == I and (b & (a ^ m)) == 0
    
    # This is still O(|F|^2) per candidate m.
    # Total work: sum over candidates m of |F|^2.
    # If we iterate through all 1820 candidates, and F grows to ~200, that's 1820 * 200^2 = 72M operations.
    # Each operation is a few bitwise ops. This should be fast enough in Python if optimized?
    # 72M simple ops might take ~5-10 seconds.
    # We have 85 seconds. We can do multiple restarts.
    
    best_size = 0
    best_masks = []
    rng = random.Random(42)
    
    # Strategy 1: Greedy with random restarts
    # We will try several random orderings.
    # Also, we can try a "smart" ordering based on frequency or something?
    # For now, random restarts are a solid baseline.
    
    # Optimization: Precompute masks for all sets.
    # We already have `masks`.
    
    # To speed up the inner loop, we can use a list of masks for the current family.
    # And maybe a set of masks for O(1) membership?
    # But the check is not just membership, it's a property.
    # So we must iterate over all pairs in F.
    
    # Let's implement the greedy loop with the optimized check.
    
    def try_greedy(order):
        family = []
        family_set = set() # For quick duplicate check, though order is a permutation so no dups
        
        for m in order:
            # Check if m can be added
            valid = True
            n = len(family)
            
            for i in range(n):
                a = family[i]
                I = a & m
                xor_am = a ^ m
                
                for j in range(n):
                    if i == j:
                        continue
                    b = family[j]
                    # Check if b is in forbidden region for (a, m)
                    # b contains I and b disjoint from xor_am
                    if (b & I) == I and (b & xor_am) == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(m)
                family_set.add(m)
        
        return family
    
    # Try a few random orders
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random shuffle
        order = list(range(n_sets))
        rng.shuffle(order)
        
        # Map indices to masks
        # It's faster to shuffle the masks directly?
        # masks is a list of ints.
        shuffled_masks = masks[:]
        rng.shuffle(shuffled_masks)
        
        family = try_greedy(shuffled_masks)
        
        if len(family) > best_size:
            best_size = len(family)
            best_masks = family[:]
            
        if best_size >= 250:
            break
    
    # Local Search / Improvement
    # Try to swap elements or add/remove to improve?
    # Given the constraints, greedy is often good for these packing problems.
    # But let's try a simple local search on the best family found.
    # Try to replace a set in the family with a set not in the family to see if we can add more?
    # Or just try to add more sets to the best family found.
    
    if time.time() < deadline - 10:
        family = best_masks[:]
        family_set = set(family)
        
        # Try to add more sets
        # We can iterate through all sets not in family and try to add them.
        # If we can add one, we do it. Repeat until no more can be added.
        # This is a "maximal" greedy step.
        
        improved = True
        while improved:
            improved = False
            for m in masks:
                if m in family_set:
                    continue
                
                # Check if m can be added
                valid = True
                n = len(family)
                
                for i in range(n):
                    a = family[i]
                    I = a & m
                    xor_am = a ^ m
                    
                    for j in range(n):
                        if i == j:
                            continue
                        b = family[j]
                        if (b & I) == I and (b & xor_am) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    family.append(m)
                    family_set.add(m)
                    improved = True
        
        if len(family) > best_size:
            best_size = len(family)
            best_masks = family
    
    # Convert best_masks back to sorted lists
    result = []
    for m in best_masks:
        s = mask_to_set[m]
        result.append(sorted(s))
    
    return result

# EVOLVE-BLOCK-END
