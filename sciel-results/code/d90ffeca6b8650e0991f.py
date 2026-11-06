"""Improved program for large four-term-progression-free integer sets."""
# EVOLVE-BLOCK-START
import time
import random

N = 512
NUM_ELEMS = 513  # bits 0..512, we use 1..512

def _precompute_conflict_masks():
    """
    For each x in 1..512, precompute a bitset mask of all elements y
    such that if y is in the set and x is added, a 4-AP is created.
    The mask has bit y set iff {y, x} is part of a 4-AP.
    """
    masks = [0] * (NUM_ELEMS)
    for x in range(1, N + 1):
        mask = 0
        # Case 1: x is the 4th term (largest)
        # x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a = x - 3 * d
            b = x - 2 * d
            c = x - d
            if a >= 1:
                # If a, b, c are all in the set, adding x creates a 4-AP
                # But for the mask, we need: if ANY of a, b, c is in the set
                # AND the other two are also in the set, then x creates a 4-AP.
                # Actually, the mask should represent: for which y, if y is in the set,
                # does it "block" x? No, that's not quite right.
                # A 4-AP is formed if all three other terms are present.
                # So the mask approach is: precompute for each x, the set of
                # triples (a,b,c) such that a,b,c,x form a 4-AP.
                # Then x can be added iff none of these triples are fully contained.
                # A simpler bitset approach: for each x, build a mask M_x such that
                # (bits & M_x) has fewer than 3 bits set from any valid triple.
                # This is complex. Let's use a different approach.
                pass
        
        # Let me reconsider. The bitset approach from the inspirations:
        # _can_add_bitset(bits, x) checks if adding x creates a 4-AP.
        # This can be done by checking all 4-APs that include x.
        # A 4-AP containing x: x can be at position 1, 2, 3, or 4.
        # For each case, we need the other 3 terms to all be in the set.
        # So we can precompute for each x a list of (a, b, c) where {a,b,c,x} is a 4-AP.
        # Then _can_add_bitset checks if any (a,b,c) are all in bits.
        
        # Let's build conflict_lists[x] = list of (a, b, c)
        # where {a,b,c,x} forms a 4-AP and a,b,c != x.
        
        pass
    
    # Actually, let's build conflict triples directly
    conflict_triples = [[] for _ in range(NUM_ELEMS)]
    for x in range(1, N + 1):
        # x is 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            conflict_triples[x].append((x + d, x + 2 * d, x + 3 * d))
        # x is 2nd term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            conflict_triples[x].append((x - d, x + d, x + 2 * d))
        # x is 3rd term: x-2d, x-d, x, x+d
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if x + d <= N:
                conflict_triples[x].append((x - 2 * d, x - d, x + d))
        # x is 4th term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            conflict_triples[x].append((x - 3 * d, x - 2 * d, x - d))
    
    return conflict_triples

def _can_add(bits, x, conflict_triples):
    """Check if adding x to the set represented by bits creates a 4-AP."""
    for (a, b, c) in conflict_triples[x]:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
            return False
    return True

def _greedy_construct(order, conflict_triples):
    """
    Greedy construction: iterate through elements in given order,
    add element if it doesn't create a 4-AP.
    Returns (bits, count).
    """
    bits = 0
    count = 0
    for x in order:
        if _can_add(bits, x, conflict_triples):
            bits |= (1 << x)
            count += 1
    return bits, count

def _local_search(bits, count, deadline, conflict_triples):
    """
    Local search: try to add elements. If blocked, try swaps.
    Returns (bits, count).
    """
    # Build set from bits for efficient lookup
    s_set = set()
    temp = bits
    for x in range(1, N + 1):
        if (temp >> x) & 1:
            s_set.add(x)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [x for x in range(1, N + 1) if x not in s_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if _can_add(bits, x, conflict_triples):
                bits |= (1 << x)
                s_set.add(x)
                count += 1
                improved = True
                break
        
        if not improved and time.time() < deadline:
            # Try 1-for-1 swaps
            for x in candidates:
                if time.time() > deadline:
                    break
                
                if _can_add(bits, x, conflict_triples):
                    continue
                
                # Find conflicts
                for (a, b, c) in conflict_triples[x]:
                    if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
                        # Try removing one of a, b, c
                        for y in (a, b, c):
                            if y not in s_set:
                                continue
                            new_bits = bits & ~(1 << y)
                            if _can_add(new_bits, x, conflict_triples):
                                bits = new_bits | (1 << x)
                                s_set.discard(y)
                                s_set.add(x)
                                # count stays the same
                                improved = True
                                break
                        if improved:
                            break
                
                if improved:
                    break
        
        # Try 1-for-2 swaps (remove 1, add 2)
        if not improved and time.time() < deadline:
            s_list = list(s_set)
            random.shuffle(s_list)
            for y in s_list:
                if time.time() > deadline:
                    break
                
                temp_bits = bits & ~(1 << y)
                added = 0
                new_set = set(s_set)
                new_set.discard(y)
                
                candidates = [x for x in range(1, N + 1) if x not in new_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if _can_add(temp_bits, x, conflict_triples):
                        temp_bits |= (1 << x)
                        new_set.add(x)
                        added += 1
                        if added >= 2:
                            bits = temp_bits
                            s_set = new_set
                            count += 1
                            improved = True
                            break
                
                if improved:
                    break
    
    return bits, count

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Precompute conflict triples
    conflict_triples = _precompute_conflict_masks()
    
    # Generate multiple greedy seeds
    best_bits = 0
    best_count = 0
    best_sets = []
    
    # Time budget for greedy seeds: ~40s
    greedy_deadline = time.time() + 40.0
    
    while time.time() < greedy_deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        bits, count = _greedy_construct(order, conflict_triples)
        
        if count > best_count:
            best_count = count
            best_bits = bits
        
        # Keep top few sets
        best_sets.append((bits, count))
        best_sets.sort(key=lambda x: x[1], reverse=True)
        if len(best_sets) > 5:
            best_sets.pop()
    
    # Local search on best sets
    # Use remaining time for local search
    search_deadline = deadline
    
    for s_bits, s_count in best_sets:
        if time.time() > search_deadline:
            break
        
        final_bits, final_count = _local_search(s_bits, s_count, search_deadline, conflict_triples)
        
        if final_count > best_count:
            best_count = final_count
            best_bits = final_bits
    
    # Convert bits to sorted list
    result = []
    temp = best_bits
    for x in range(1, N + 1):
        if (temp >> x) & 1:
            result.append(x)
    
    return result

# EVOLVE-BLOCK-END
