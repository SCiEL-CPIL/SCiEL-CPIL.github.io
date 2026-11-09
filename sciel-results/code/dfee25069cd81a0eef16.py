import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity_bitset(sums, n):
    """
    Check if the bitsets in 'sums' represent a valid cardinality-separated set.
    sums[k] is an integer where bit s is 1 if sum s is achievable with k elements.
    n is the current size of the set.
    Valid iff sums[i] & sums[j] == 0 for all i != j.
    """
    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            if sums[i] & sums[j]:
                return False
    return True

def try_greedy_order(order, max_val=256):
    """
    Greedily build a set by trying elements in the given order.
    Returns the largest valid set found.
    """
    # sums[k] is a bitset: bit s set means sum s is achievable with exactly k elements
    # Start: empty subset gives sum 0 with cardinality 0
    sums = [0] * 62
    sums[0] = 1  # bit 0 set: sum 0 achievable with 0 elements
    
    chosen = []
    max_sum = 0  # track max possible sum to bound bitset operations
    
    for candidate in order:
        if candidate < 1 or candidate > max_val:
            continue
        if candidate in chosen:
            continue
        
        # Try adding candidate: new sums at cardinality k = old sums[k] | (old sums[k-1] shifted by candidate)
        # We need to check if any new sum at cardinality k collides with existing sums at cardinality j != k
        # Efficient check: for each k, new_sums[k] = sums[k] | ((sums[k-1] << candidate) if k > 0 else 0)
        # Collision check: new_sums[k] & sums[j] for j != k must be 0
        # But actually we just need: for all k != j, (sums[k] | ((sums[k-1] << candidate) if k>0 else 0)) & (sums[j] | ((sums[j-1] << candidate) if j>0 else 0)) == 0
        # Simpler: compute all new_sums, then check pairwise
        
        new_sums = sums[:]  # copy
        current_max_card = len(chosen)
        
        # Update sums in reverse order to avoid using updated values
        # Actually, let's compute the new sums properly
        # new_sums[k] = sums[k] | (sums[k-1] << candidate) for k >= 1
        # new_sums[0] = sums[0] (unchanged, still just bit 0)
        
        # Let's build new_sums from scratch
        ns = [0] * 62
        ns[0] = 1  # sum 0 always achievable with 0 elements
        
        for k in range(1, current_max_card + 2):
            # Without candidate: sums[k]
            # With candidate: sums[k-1] shifted by candidate
            val = sums[k]
            if k - 1 >= 0 and k - 1 <= current_max_card:
                val |= (sums[k-1] << candidate)
            ns[k] = val
        
        # Check validity: for all i != j, ns[i] & ns[j] == 0
        valid = True
        for i in range(current_max_card + 2):
            if ns[i] == 0:
                continue
            for j in range(i + 1, current_max_card + 3):
                if ns[j] == 0:
                    continue
                if ns[i] & ns[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            sums = ns
            chosen.append(candidate)
            max_sum += candidate
    
    return chosen

def local_search_improve(chosen, all_candidates, max_val=256):
    """
    Try to improve a valid set by swapping elements or adding new ones.
    Returns an improved set (possibly the same).
    """
    if not chosen:
        return chosen
    
    chosen_set = set(chosen)
    best = list(chosen)
    best_len = len(best)
    
    # First, try to add any candidate that fits
    for cand in all_candidates:
        if cand in chosen_set:
            continue
        test_set = list(best) + [cand]
        test_set_set = set(test_set)
        
        # Check validity
        sums = [0] * (len(test_set) + 2)
        sums[0] = 1
        for elem in test_set:
            ns = [0] * (len(test_set) + 2)
            ns[0] = 1
            for k in range(1, len(test_set) + 1):
                val = sums[k]
                if k - 1 >= 0:
                    val |= (sums[k-1] << elem)
                ns[k] = val
            sums = ns
        
        valid = True
        n = len(test_set)
        for i in range(n + 1):
            if sums[i] == 0:
                continue
            for j in range(i + 1, n + 2):
                if sums[j] == 0:
                    continue
                if sums[i] & sums[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best = test_set
            best_len = len(best)
            chosen_set.add(cand)
            # Continue trying to add more
    
    # Now try swaps: remove one element, try to add a different one
    for _ in range(5):  # limited iterations
        if is_done():
            break
        improved = False
        for i, elem in enumerate(list(best)):
            # Try removing elem
            reduced = best[:i] + best[i+1:]
            reduced_set = set(reduced)
            
            # Try adding each candidate not in reduced
            for cand in all_candidates:
                if cand in reduced_set:
                    continue
                test_set = reduced + [cand]
                
                # Check validity
                sums = [0] * (len(test_set) + 2)
                sums[0] = 1
                for elem2 in test_set:
                    ns = [0] * (len(test_set) + 2)
                    ns[0] = 1
                    for k in range(1, len(test_set) + 1):
                        val = sums[k]
                        if k - 1 >= 0:
                            val |= (sums[k-1] << elem2)
                        ns[k] = val
                    sums = ns
                
                valid = True
                n = len(test_set)
                for ii in range(n + 1):
                    if sums[ii] == 0:
                        continue
                    for jj in range(ii + 1, n + 2):
                        if sums[jj] == 0:
                            continue
                        if sums[ii] & sums[jj]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and len(test_set) > len(best):
                    best = test_set
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    best_set = []
    best_len = 0
    
    all_candidates = list(range(1, 257))
    
    # Strategy 1: Try ascending order
    order_asc = list(range(1, 257))
    result = try_greedy_order(order_asc)
    if len(result) > best_len:
        best_set = result
        best_len = len(result)
    
    # Strategy 2: Try descending order
    order_desc = list(range(256, 0, -1))
    result = try_greedy_order(order_desc)
    if len(result) > best_len:
        best_set = result
        best_len = len(result)
    
    # Strategy 3: Random restarts with different orderings
    random.seed(42)
    attempts = 0
    max_attempts = 5000
    
    while not is_done() and attempts < max_attempts:
        attempts += 1
        
        # Create a random ordering with some bias
        r = random.random()
        if r < 0.2:
            # Mostly ascending with some randomization
            order = list(range(1, 257))
            random.shuffle(order)
            # Re-sort but with some random swaps
            for _ in range(random.randint(0, 10)):
                i = random.randint(0, 255)
                j = random.randint(0, 255)
                order[i], order[j] = order[j], order[i]
        elif r < 0.4:
            # Random ordering
            order = list(range(1, 257))
            random.shuffle(order)
        elif r < 0.6:
            # Descending with randomization
            order = list(range(256, 0, -1))
            for _ in range(random.randint(0, 10)):
                i = random.randint(0, 255)
                j = random.randint(0, 255)
                order[i], order[j] = order[j], order[i]
        elif r < 0.8:
            # Weighted toward larger numbers
            order = list(range(1, 257))
            for i in range(len(order)):
                weight = (i + 1) ** 2
                if random.random() > weight / (256 ** 2):
                    continue
                # Keep in order
            random.shuffle(order)
        else:
            # Weighted toward smaller numbers
            order = list(range(1, 257))
            for i in range(len(order)):
                weight = (256 - i) ** 2
                if random.random() > weight / (256 ** 2):
                    continue
            random.shuffle(order)
        
        result = try_greedy_order(order)
        if len(result) > best_len:
            best_set = result
            best_len = len(result)
        
        # If we found a decent set, try local search
        if best_len >= 4 and not is_done():
            improved = local_search_improve(best_set, all_candidates)
            if len(improved) > best_len:
                best_set = improved
                best_len = len(best_set)
    
    # Final verification of best_set
    if best_set:
        # Verify validity
        sums = [0] * (len(best_set) + 2)
        sums[0] = 1
        for elem in best_set:
            ns = [0] * (len(best_set) + 2)
            ns[0] = 1
            for k in range(1, len(best_set) + 1):
                val = sums[k]
                if k - 1 >= 0:
                    val |= (sums[k-1] << elem)
                ns[k] = val
            sums = ns
        
        # Check pairwise
        valid = True
        n = len(best_set)
        for i in range(n + 1):
            if sums[i] == 0:
                continue
            for j in range(i + 1, n + 2):
                if sums[j] == 0:
                    continue
                if sums[i] & sums[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a known valid set
            best_set = list(range(128, 136))
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
