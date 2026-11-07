import time
import sys
import random

def _validate_full(elements):
    """
    Full strict validation.
    1. All elements distinct, in 1..256, type int.
    2. All non-empty subset sums are unique.
    3. No two distinct non-empty subset sums divide one another.
    """
    if not elements:
        return True
    
    # Type and range check
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    if len(set(elements)) != len(elements):
        return False
        
    # Generate all subset sums
    sums = set([0])
    for x in elements:
        new_sums = set(s + x for s in sums)
        if sums & new_sums:
            return False
        sums |= new_sums
        
    non_empty_sums = [s for s in sums if s > 0]
    non_empty_sums.sort()
    n = len(non_empty_sums)
    max_sum = non_empty_sums[-1]
    
    for i in range(n):
        a = non_empty_sums[i]
        if a > max_sum // 2:
            break
        for j in range(i + 1, n):
            b = non_empty_sums[j]
            if b % a == 0:
                return False
    return True

def construct():
    deadline = time.time() + 45.0
    
    # Precompute divisibility relationships for all possible sums (1..4096)
    # max_sum = 30 * 256 = 7680, but we'll cap at a reasonable limit
    MAX_SUM = 4096
    
    # forbidden_pairs[a] = set of b such that a divides b and b != a
    # For efficiency, use a list of bitmasks or just check on the fly
    # Actually, let's precompute which values divide which
    # divisors_of[b] = list of a < b such that a | b
    divisors_of = [[] for _ in range(MAX_SUM + 1)]
    for a in range(1, MAX_SUM + 1):
        for b in range(2 * a, MAX_SUM + 1, a):
            divisors_of[b].append(a)
    
    # Strategy: Use bitset representation for subset sums
    # sums_bitset is a Python int where bit i is set if sum i is achievable
    # This allows fast collision detection: (sums_bitset << candidate) & sums_bitset != 0 means collision
    
    best_elements = []
    best_size = 0
    
    def try_build(order):
        """Try to build a valid set by adding elements in the given order."""
        current_elements = []
        sums_bitset = 1  # bit 0 set, representing sum 0
        current_sums = {0}  # Keep a set for divisibility checks
        
        for candidate in order:
            if time.time() > deadline:
                break
            
            # Check for collision: does shifting sums_bitset by candidate overlap with sums_bitset?
            shifted = sums_bitset << candidate
            if shifted & sums_bitset:
                continue  # Collision, skip this candidate
            
            # Check divisibility constraint
            # New sums are {s + candidate for s in current_sums}
            # We need to ensure no new sum divides or is divided by any existing sum
            # And no two new sums divide each other
            
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + candidate
                if ns > MAX_SUM:
                    continue
                new_sums.add(ns)
            
            if not new_sums:
                continue
            
            # Check: for each new sum ns, does it divide or get divided by any existing sum?
            # And do any two new sums divide each other?
            
            # First, check new sums against existing sums
            for ns in new_sums:
                # Check if ns divides any existing sum
                for es in current_sums:
                    if es > 0 and es > ns and es % ns == 0:
                        valid = False
                        break
                    if es > 0 and ns > es and ns % es == 0:
                        valid = False
                        break
                if not valid:
                    break
                
                # Check if ns divides any other new sum
                if valid:
                    for ns2 in new_sums:
                        if ns2 != ns:
                            if ns2 > ns and ns2 % ns == 0:
                                valid = False
                                break
                            if ns > ns2 and ns % ns2 == 0:
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                current_elements.append(candidate)
                current_sums |= new_sums
                sums_bitset |= shifted
        
        return current_elements
    
    # Candidate pool: focus on larger numbers to reduce divisibility issues
    # Large numbers make sums large, reducing the chance of one sum dividing another
    
    # Try different strategies:
    # 1. Descending order from high end
    # 2. Randomized orders with bias towards large numbers
    # 3. Start with a few large anchors, then fill in
    
    # Strategy 1: Descending
    order_desc = list(range(256, 0, -1))
    elems = try_build(order_desc)
    if len(elems) > best_size:
        best_size = len(elems)
        best_elements = elems
    
    # Strategy 2: Randomized with large number bias
    random.seed(42)
    
    for restart in range(500):
        if time.time() > deadline - 5.0:
            break
        
        # Generate order with bias towards larger numbers
        # Use a weighted shuffle: higher numbers have higher probability of being selected early
        order = list(range(1, 257))
        
        # Bias: sort by a random key that favors larger numbers
        # Or just shuffle and then move some large numbers to the front
        random.shuffle(order)
        
        # Ensure we try some large numbers early
        # Pick 5-10 random large numbers and put them at the start
        large_numbers = [x for x in range(128, 257)]
        random.shuffle(large_numbers)
        num_large = random.randint(3, 8)
        front = large_numbers[:num_large]
        rest = [x for x in order if x not in front]
        order = front + rest
        
        elems = try_build(order)
        if len(elems) > best_size:
            best_size = len(elems)
            best_elements = elems
    
    # Strategy 3: Local search - try to improve by swapping elements
    # For each element in best_elements, try replacing it with a different candidate
    if best_elements and time.time() < deadline - 10.0:
        for i in range(len(best_elements)):
            if time.time() > deadline - 5.0:
                break
            
            # Try swapping best_elements[i] with various candidates
            for candidate in range(1, 257):
                if candidate in best_elements:
                    continue
                
                new_elements = best_elements[:i] + [candidate] + best_elements[i+1:]
                
                # Quick check: is this valid?
                if _validate_full(new_elements):
                    # Same size, but different composition might allow further growth
                    # Try to extend from here
                    remaining = [x for x in range(1, 257) if x not in new_elements]
                    random.shuffle(remaining)
                    extended = try_build_from(new_elements, remaining)
                    if len(extended) > best_size:
                        best_size = len(extended)
                        best_elements = extended
                        break
    
    # Final validation
    if best_elements:
        if not _validate_full(best_elements):
            best_elements = [20, 21, 23]
    
    return list(best_elements)

def try_build_from(start_elements, remaining_candidates):
    """Try to extend a starting set with remaining candidates."""
    if not start_elements:
        return []
    
    # Build initial sums
    current_elements = list(start_elements)
    current_sums = {0}
    for x in current_elements:
        new_sums = set(s + x for s in current_sums)
        if current_sums & new_sums:
            return current_elements  # Invalid start, return as is
        current_sums |= new_sums
    
    # Add candidates in the given order
    for candidate in remaining_candidates:
        if time.time() > time.time() + 0.1:  # Simple time check placeholder
            break
        
        new_sums = set(s + candidate for s in current_sums)
        if current_sums & new_sums:
            continue
        
        # Check divisibility
        valid = True
        for ns in new_sums:
            if ns == 0:
                continue
            for es in current_sums:
                if es > 0 and es != ns:
                    if es > ns and es % ns == 0:
                        valid = False
                        break
                    if ns > es and ns % es == 0:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            current_elements.append(candidate)
            current_sums |= new_sums
    
    return current_elements

if __name__ == "__main__":
    print(construct())
