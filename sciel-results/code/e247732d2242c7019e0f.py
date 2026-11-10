"""Improved program: bitmask 4-AP check + local search with swaps/additions."""
import time
import random

N = 256
DEADLINE_MARGIN = 2.0  # seconds before the 90s limit

def _count_3ap(bits):
    """Count distinct 3-term APs with positive difference in the set represented by bits."""
    count = 0
    # For each pair (a, c) with a < c, check if (a+c)//2 is in the set and (a+c) is even
    # More efficient: iterate over differences
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if (bits >> (a - 1)) & 1 and (bits >> (b - 1)) & 1 and (bits >> (c - 1)) & 1:
                count += 1
    return count

def _has_4ap(bits):
    """Check if the set represented by bits contains a nonconstant 4-term AP."""
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if ((bits >> (a - 1)) & 1) and ((bits >> (b - 1)) & 1) and \
               ((bits >> (c - 1)) & 1) and ((bits >> (e - 1)) & 1):
                return True
    return False

def _greedy_with_order(order, deadline):
    """Greedy construction using a given ordering of elements."""
    bits = 0
    for x in order:
        if time.time() > deadline:
            break
        idx = x - 1
        # Check if adding x creates a 4-AP
        new_bits = bits | (1 << idx)
        if not _has_4ap(new_bits):
            bits = new_bits
    return bits

def _local_search(bits, deadline, max_rounds=50000):
    """Local search: try swaps and additions to increase 3-AP count."""
    best_bits = bits
    best_count = _count_3ap(bits)
    
    # Precompute list of bits set and unset
    set_list = [i + 1 for i in range(N) if (bits >> i) & 1]
    unset_list = [i + 1 for i in range(N) if not ((bits >> i) & 1)]
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try additions: add an element from unset_list
        for x in unset_list:
            idx = x - 1
            new_bits = bits | (1 << idx)
            if _has_4ap(new_bits):
                continue
            new_count = _count_3ap(new_bits)
            if new_count > best_count:
                bits = new_bits
                best_count = new_count
                improved = True
                # Update lists
                set_list.append(x)
                unset_list.remove(x)
                break
        
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        for rm in set_list:
            if time.time() > deadline:
                return best_bits, best_count
            rm_idx = rm - 1
            base_bits = bits & ~(1 << rm_idx)
            
            for x in unset_list:
                idx = x - 1
                new_bits = base_bits | (1 << idx)
                if _has_4ap(new_bits):
                    continue
                new_count = _count_3ap(new_bits)
                if new_count > best_count:
                    bits = new_bits
                    best_count = new_count
                    improved = True
                    # Update lists
                    set_list.remove(rm)
                    set_list.append(x)
                    unset_list.remove(x)
                    unset_list.append(rm)
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return bits, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    best_bits = 0
    best_count = 0
    
    # Multiple random restarts with different orderings
    random.seed(42)
    
    for restart in range(200):  # Upper bound on restarts; deadline controls actual count
        if time.time() > deadline:
            break
        
        # Generate a random ordering
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy construction
        bits = _greedy_with_order(order, deadline)
        
        if time.time() > deadline:
            if _count_3ap(bits) > best_count:
                best_bits = bits
                best_count = _count_3ap(bits)
            break
        
        # Local search
        bits, count = _local_search(bits, deadline)
        
        if count > best_count:
            best_bits = bits
            best_count = count
    
    # Final check
    if time.time() > deadline - 1.0:
        # Just return what we have
        pass
    
    result = sorted([i + 1 for i in range(N) if (best_bits >> i) & 1])
    return result if result else [1]
