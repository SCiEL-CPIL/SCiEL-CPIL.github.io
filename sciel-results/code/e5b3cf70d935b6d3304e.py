"""Rewritten program for large interval sets avoiding a prescribed subset sum."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1  # bits 0..200

    def reachable_bitset(elements):
        """Compute bitset of reachable subset sums up to TARGET."""
        r = 1  # bit 0 set (sum 0)
        for x in elements:
            r |= (r << x)
            r &= MASK
        return r

    def has_sum_200(elements):
        """Check if any subset of elements sums to exactly 200 using bitset."""
        r = 1
        for x in elements:
            r |= (r << x)
            r &= MASK
            if (r >> TARGET) & 1:
                return True
        return False

    def try_add(elements, x, bitset):
        """
        Try adding x to elements. 
        Returns (new_bitset, success) if adding x doesn't create sum 200.
        """
        # Check if sum 200 - x is reachable in current elements
        # If yes, then x + subset = 200.
        # Also, if x == 200 (not possible here), or x itself is 200.
        # Since x < 200, we just check if (200-x) is in bitset.
        # Wait, the subset can be empty? Yes, if x=200. But x<=100.
        # So we need a non-empty subset of 'elements' summing to 200-x?
        # Actually, the subset of 'elements' can be empty only if 200-x == 0, i.e., x=200.
        # If 200-x > 0, the subset must be non-empty.
        # The bitset includes 0. If 200-x == 0, bit 0 is set.
        # If 200-x > 0, and bit (200-x) is set, it means there is a subset summing to 200-x.
        # Is it possible that the subset is empty? Only if 200-x=0.
        # So checking bit (200-x) is sufficient.
        
        target_rem = TARGET - x
        if target_rem < 0:
            # x > 200, cannot be part of sum 200 with positive integers
            return bitset | (1 << x), True # Just add it, it doesn't form 200
        if (bitset >> target_rem) & 1:
            return None, False
        # Safe to add
        new_bitset = bitset | ((bitset << x) & MASK)
        # Check if 200 became reachable? 
        # If (new_bitset >> 200) & 1, then we formed 200.
        # This happens if x + subset = 200.
        # We already checked if subset sum 200-x exists.
        # So new_bitset should NOT have bit 200 set.
        # But just in case:
        if (new_bitset >> TARGET) & 1:
            return None, False
        return new_bitset, True

    # 1. Generate all triples in [51, 100] summing to 200
    upper_range = list(range(51, 101))
    triples = []
    for i in range(len(upper_range)):
        a = upper_range[i]
        for j in range(i + 1, len(upper_range)):
            b = upper_range[j]
            if a + b >= 200:
                break
            c = 200 - a - b
            if c > b and c <= 100:
                # Find index of c? We just need the values.
                # Ensure c is in upper_range (c > b >= 52, so c >= 53, which is in range)
                triples.append((a, b, c))
    
    # 2. Greedy Hitting Set to find maximal subset of upper_range with no 3-sum 200
    def greedy_maximal_subset_no_3sum(range_list, triples_list, seed):
        rng = random.Random(seed)
        S = set(range_list)
        active_triples = [t for t in triples_list]
        
        while active_triples:
            if time.time() > deadline:
                break
            # Count frequencies
            counts = {}
            for t in active_triples:
                for e in t:
                    counts[e] = counts.get(e, 0) + 1
            
            # Find max frequency
            max_count = max(counts.values())
            # Get all elements with max frequency
            candidates = [e for e, c in counts.items() if c == max_count]
            
            # Pick one (randomly for diversity)
            e_to_remove = rng.choice(candidates)
            S.remove(e_to_remove)
            
            # Remove all triples containing e_to_remove
            new_active = []
            for t in active_triples:
                if e_to_remove not in t:
                    new_active.append(t)
            active_triples = new_active
            
        return S

    # Run multiple greedy passes with different seeds
    best_upper_set = set()
    best_upper_size = 0
    
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        s = greedy_maximal_subset_no_3sum(upper_range, triples, seed)
        if len(s) > best_upper_size:
            best_upper_set = s
            best_upper_size = len(s)

    # 3. Try to add elements from [1, 50]
    # We will try multiple strategies for adding the lower elements
    best_final_set = set(best_upper_set)
    best_final_size = len(best_final_set)
    
    lower_range = list(range(1, 51))
    
    # Strategy A: Add in descending order
    for order_name, order in [("desc", sorted(lower_range, reverse=True)), 
                              ("asc", sorted(lower_range)),
                              ("rand1", list(lower_range))]:
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(42 if order_name == "rand1" else 0)
        if order_name == "rand1":
            rng.shuffle(order)
            
        current_set = set(best_upper_set)
        # Initialize bitset for current_set
        bitset = 1
        for x in current_set:
            bitset |= (bitset << x)
            bitset &= MASK
        
        for x in order:
            if x in current_set:
                continue
            new_bitset, success = try_add(list(current_set), x, bitset)
            if success:
                current_set.add(x)
                bitset = new_bitset
        
        if len(current_set) > best_final_size:
            best_final_set = current_set
            best_final_size = len(best_final_set)

    # Strategy B: Random orders
    for trial in range(500):
        if time.time() > deadline - 2:
            break
        rng = random.Random(trial)
        order = list(lower_range)
        rng.shuffle(order)
        
        current_set = set(best_upper_set)
        bitset = 1
        for x in current_set:
            bitset |= (bitset << x)
            bitset &= MASK
        
        for x in order:
            if x in current_set:
                continue
            new_bitset, success = try_add(list(current_set), x, bitset)
            if success:
                current_set.add(x)
                bitset = new_bitset
        
        if len(current_set) > best_final_size:
            best_final_set = current_set
            best_final_size = len(current_set)

    # Final verification
    best_list = list(best_final_set)
    if has_sum_200(best_list):
        # Fallback to a known valid set if something went wrong
        # [66, 100] is size 35 and valid
        best_list = list(range(66, 101))

    return best_list
# EVOLVE-BLOCK-END
