import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Bitset-based representation:
    # sums_by_card[c] is a Python int where bit s is set if sum s is achievable with exactly c elements
    # We only need bits up to 256*60 = 15360, so we need about 15361 bits
    
    MAX_SUM = 256 * 60  # Upper bound on any subset sum
    MASK = (1 << (MAX_SUM + 1)) - 1  # Mask to keep bits in range
    
    def is_valid_addition(x, sums_by_card, current_size):
        """
        Check if adding x is valid.
        x: the candidate value to add
        sums_by_card: list of bitsets, sums_by_card[c] has bit s set if sum s achievable with c elements
        current_size: current number of elements in the set
        
        When we add x, for each cardinality c from 1 to current_size+1:
          new_sums_c = (sums_by_card[c-1] << x) & MASK
          We need: new_sums_c & sums_by_card[c'] == 0 for all c' != c
        """
        # Precompute the new sums for each cardinality
        # But we only need to check collisions with existing sums
        
        # For each existing cardinality c (0 to current_size):
        #   The new sums that will be generated at cardinality c+1 are: sums_by_card[c] << x
        #   These must not collide with sums_by_card[c'] for any c' != c+1
        
        # More precisely:
        # After adding x, sums_by_card'[c] = sums_by_card[c] | (sums_by_card[c-1] << x) for c >= 1
        # sums_by_card'[0] = sums_by_card[0]
        
        # The constraint is: for all c1 != c2, sums_by_card'[c1] & sums_by_card'[c2] == 0
        
        # We can check this by:
        # 1. Compute all new_sums_c = sums_by_card[c-1] << x for c = 1..current_size+1
        # 2. Check that new_sums_c doesn't collide with any existing sums_by_card[c'] where c' != c
        # 3. Also check that new_sums_c doesn't collide with new_sums_c' for c' != c (but this shouldn't happen if existing sums are disjoint)
        
        for c in range(1, current_size + 2):
            if c > len(sums_by_card) - 1:
                break
            new_sums = (sums_by_card[c-1] << x) & MASK
            if new_sums == 0:
                continue
            # Check collision with all existing cardinalities except c
            for c2 in range(0, current_size + 1):
                if c2 == c:
                    continue
                if c2 < len(sums_by_card):
                    if new_sums & sums_by_card[c2]:
                        return False
        return True
    
    def add_element(x, sums_by_card, current_size):
        """
        Add element x and update sums_by_card.
        Returns new sums_by_card list.
        """
        new_sums = [s for s in sums_by_card]
        # Extend if needed
        while len(new_sums) < current_size + 2:
            new_sums.append(0)
        
        # Update in reverse order to avoid using updated values
        for c in range(current_size + 1, 0, -1):
            new_sums[c] = new_sums[c] | ((sums_by_card[c-1] << x) & MASK)
        
        return new_sums
    
    def remove_element(x, sums_by_card, current_size):
        """
        Remove element x from the set and recompute sums_by_card.
        This is expensive, so we'll use it sparingly.
        Actually, removal is hard with bitsets. We need to recompute from scratch.
        For now, let's just do a full recomputation when we need to remove.
        """
        # This is complex. For simulated annealing, we'll use a different approach:
        # Instead of remove+add, we'll just try to add elements greedily and use
        # a different strategy for improvement.
        # For now, return None to indicate this operation is not supported.
        return None
    
    # Phase 1: Greedy construction
    # Start with empty set
    sums_by_card = [1]  # sums_by_card[0] = bit 0 set (empty subset has sum 0)
    current_set = []
    
    # Try to add elements in increasing order
    for x in range(1, 257):
        if is_valid_addition(x, sums_by_card, len(current_set)):
            current_set.append(x)
            sums_by_card = add_element(x, sums_by_card, len(current_set))
    
    best_set = current_set[:]
    best_size = len(current_set)
    best_sums = sums_by_card[:]
    
    # Phase 2: Try to improve by adding more elements in different order
    # Try multiple greedy passes with different starting points
    random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Try a randomized greedy
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        sums_by_card = [1]
        current_set = []
        
        for x in candidates:
            if len(current_set) >= 60:
                break
            if is_valid_addition(x, sums_by_card, len(current_set)):
                current_set.append(x)
                sums_by_card = add_element(x, sums_by_card, len(current_set))
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            best_sums = sums_by_card[:]
    
    # Phase 3: Simulated annealing to escape local optima
    # We'll use a strategy of removing one element and trying to add two
    # Or just try random additions after removing random elements
    
    current_set = best_set[:]
    sums_by_card = best_sums[:]
    
    T = 10.0
    cooling = 0.999
    min_T = 0.001
    
    while time.time() < deadline - 1:
        if T < min_T:
            T = min_T
        
        if random.random() < 0.3 and len(current_set) > 1:
            # Try to remove a random element and add a different one
            idx = random.randint(0, len(current_set) - 1)
            x = current_set[idx]
            
            # Recompute sums_by_card without x
            # This requires knowing the original elements
            # Let's rebuild from the remaining elements
            remaining = [current_set[i] for i in range(len(current_set)) if i != idx]
            
            # Rebuild sums_by_card from scratch for remaining
            new_sums = [1]
            for elem in remaining:
                new_sums = add_element(elem, new_sums, len(new_sums) - 1)
            
            # Now try to add a new element
            added = False
            for _ in range(10):  # Try up to 10 random candidates
                y = random.randint(1, 256)
                if y not in remaining and is_valid_addition(y, new_sums, len(remaining)):
                    current_set = remaining + [y]
                    sums_by_card = add_element(y, new_sums, len(remaining))
                    added = True
                    break
            
            if not added:
                # Revert
                current_set = best_set[:]
                sums_by_card = best_sums[:]
        else:
            # Try to add a random element
            y = random.randint(1, 256)
            if y not in current_set and is_valid_addition(y, sums_by_card, len(current_set)):
                current_set.append(y)
                sums_by_card = add_element(y, sums_by_card, len(current_set))
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            best_sums = sums_by_card[:]
        
        T *= cooling
    
    # Final result
    result = [int(x) for x in best_set]
    
    return result

