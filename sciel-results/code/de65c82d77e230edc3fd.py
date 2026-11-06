import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    required = [1, 2, 5, 11]
    required_set = set(required)
    best_set = set(required)
    best_size = len(best_set)
    
    def get_sums(S):
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def is_valid_addition(x, S, sums):
        for y in S:
            if (x + y) in sums:
                return False
        if (2 * x) in sums:
            return False
        return True

    def add_to_sums(x, S, sums):
        for y in S:
            sums.add(x + y)
        sums.add(2 * x)

    def count_valid_candidates(S, sums):
        """Count how many elements can be added to S."""
        count = 0
        for x in range(1, 285):
            if x not in S:
                if is_valid_addition(x, S, sums):
                    count += 1
        return count

    def find_blocking_elements(S, sums, candidates):
        """Find which elements in S block the most candidates (MRV-inspired)."""
        if not candidates:
            return []
        # For each non-required element, count how many candidates it blocks
        block_count = {}
        for a in S:
            if a in required_set:
                continue
            blocked = 0
            for x in candidates:
                # Check if a is the reason x can't be added
                # x can't be added because some sum collision exists
                # If removing a would allow x, then a blocks x
                S_temp = S - {a}
                sums_temp = get_sums(S_temp)
                if is_valid_addition(x, S_temp, sums_temp):
                    blocked += 1
            if blocked > 0:
                block_count[a] = blocked
        # Sort by block count descending
        sorted_blocks = sorted(block_count.items(), key=lambda kv: -kv[1])
        return sorted_blocks

    def greedy_fill(S, sums, deadline):
        """Greedily add as many elements as possible with MRV ordering."""
        while time.time() < deadline:
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            if not candidates:
                break
            # Use MRV-like heuristic: pick candidate that leaves most options
            # But for speed, just pick randomly from top candidates
            if len(candidates) > 20:
                # Sample from a subset for speed
                sample = random.sample(candidates, min(20, len(candidates)))
                x = random.choice(sample)
            else:
                x = random.choice(candidates)
            S.add(x)
            add_to_sums(x, S, sums)
        return S, sums

    def local_search_mrv(S, sums, deadline):
        """Local search with MRV-guided swaps."""
        improved = True
        stall_count = 0
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            
            if candidates:
                x = random.choice(candidates)
                S.add(x)
                add_to_sums(x, S, sums)
                improved = True
                stall_count = 0
                continue
            
            # Stuck - try MRV-guided swap
            non_fixed = [a for a in S if a not in required_set]
            if not non_fixed:
                break
            
            # Find which elements block the most potential additions
            # First, find all "almost valid" candidates (those that fail by one conflict)
            almost_candidates = []
            for x in range(1, 285):
                if x in S:
                    continue
                # Check if x is blocked by exactly one element
                blocking_elements = []
                for y in S:
                    if (x + y) in sums:
                        blocking_elements.append(y)
                if (2 * x) in sums:
                    # Find which element creates the 2x sum
                    for y in S:
                        if y + y == 2 * x:
                            blocking_elements.append(y)
                        elif y in S and x + y == 2 * x:  # y = x, but x not in S
                            pass
                if len(blocking_elements) == 1:
                    almost_candidates.append((x, blocking_elements[0]))
            
            if almost_candidates:
                # Pick a random almost-candidate and try to remove its blocker
                x, blocker = random.choice(almost_candidates)
                if blocker in required_set:
                    continue
                S_temp = S - {blocker}
                sums_temp = get_sums(S_temp)
                if is_valid_addition(x, S_temp, sums_temp):
                    S.remove(blocker)
                    S.add(x)
                    sums = get_sums(S)
                    improved = True
                    stall_count = 0
                    continue
            
            # If no almost-candidates, try random swap as fallback
            a = random.choice(non_fixed)
            b = random.randint(1, 284)
            if b in S:
                continue
            S_temp = S - {a}
            sums_temp = get_sums(S_temp)
            if is_valid_addition(b, S_temp, sums_temp):
                S.remove(a)
                S.add(b)
                sums = get_sums(S)
                improved = True
                stall_count = 0
                continue
            
            stall_count += 1
            if stall_count > 50:
                break
        
        return S, sums

    def local_search_2swap_mrv(S, sums, deadline):
        """Local search with MRV-guided 2-swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add
            candidates = []
            for x in range(1, 285):
                if x not in S:
                    if is_valid_addition(x, S, sums):
                        candidates.append(x)
            
            if candidates:
                x = random.choice(candidates)
                S.add(x)
                add_to_sums(x, S, sums)
                improved = True
                continue
            
            # Try 2-swap: find two elements to remove that together unblock an addition
            non_fixed = [a for a in S if a not in required_set]
            if len(non_fixed) < 2:
                break
            
            # Find candidates blocked by exactly two elements
            two_block_candidates = []
            for x in range(1, 285):
                if x in S:
                    continue
                blocking_elements = []
                for y in S:
                    if (x + y) in sums:
                        blocking_elements.append(y)
                if (2 * x) in sums:
                    for y in S:
                        if y + y == 2 * x:
                            if y not in blocking_elements:
                                blocking_elements.append(y)
                if len(blocking_elements) == 2:
                    two_block_candidates.append((x, tuple(sorted(blocking_elements))))
            
            if two_block_candidates:
                x, (b1, b2) = random.choice(two_block_candidates)
                if b1 in required_set or b2 in required_set:
                    continue
                S_temp = S - {b1, b2}
                sums_temp = get_sums(S_temp)
                if is_valid_addition(x, S_temp, sums_temp):
                    S.remove(b1)
                    S.remove(b2)
                    S.add(x)
                    sums = get_sums(S)
                    improved = True
                    continue
            
            # Fallback: random 2-swap
            a1, a2 = random.sample(non_fixed, 2)
            b1 = random.randint(1, 284)
            b2 = random.randint(1, 284)
            if b1 in S or b2 in S or b1 == b2:
                continue
            S_temp = S - {a1, a2}
            sums_temp = get_sums(S_temp)
            if not is_valid_addition(b1, S_temp, sums_temp):
                continue
            S_temp.add(b1)
            add_to_sums(b1, S_temp, sums_temp)
            if not is_valid_addition(b2, S_temp, sums_temp):
                continue
            S_temp.add(b2)
            add_to_sums(b2, S_temp, sums_temp)
            S = S_temp
            sums = sums_temp
            improved = True
        
        return S, sums

    def remove_and_refill(S, sums, deadline):
        """Remove a non-fixed element and greedily refill with MRV."""
        non_fixed = [a for a in S if a not in required_set]
        if not non_fixed:
            return S, sums
        
        # Pick element to remove based on which one, when removed, allows most additions
        best_remove = None
        best_gain = 0
        for a in non_fixed:
            S_temp = S - {a}
            sums_temp = get_sums(S_temp)
            gain = count_valid_candidates(S_temp, sums_temp)
            if gain > best_gain:
                best_gain = gain
                best_remove = a
        
        if best_remove is None:
            return S, sums
        
        S_temp = S - {best_remove}
        sums_temp = get_sums(S_temp)
        S_temp, sums_temp = greedy_fill(S_temp, sums_temp, deadline)
        if len(S_temp) > len(S):
            return S_temp, sums_temp
        return S, sums

    def random_greedy_with_mrv_swaps():
        S = set(required)
        sums = get_sums(S)
        
        # Phase 1: Greedy Add with MRV
        S, sums = greedy_fill(S, sums, deadline)
        
        # Phase 2: Local search with MRV-guided 1-swaps
        S, sums = local_search_mrv(S, sums, deadline)
        
        # Phase 3: Local search with MRV-guided 2-swaps
        S, sums = local_search_2swap_mrv(S, sums, deadline)
        
        # Phase 4: Remove and refill with MRV
        while time.time() < deadline:
            S_new, sums_new = remove_and_refill(S, sums, deadline)
            if len(S_new) > len(S):
                S, sums = S_new, sums_new
                # Continue local search after improvement
                S, sums = local_search_mrv(S, sums, deadline)
                S, sums = local_search_2swap_mrv(S, sums, deadline)
            else:
                break
        
        return S

    # Main Loop: Multiple Restarts
    while time.time() < deadline:
        candidate_set = random_greedy_with_mrv_swaps()
        if len(candidate_set) > best_size:
            best_size = len(candidate_set)
            best_set = candidate_set
            if best_size == 20:
                break

    return list(best_set)
