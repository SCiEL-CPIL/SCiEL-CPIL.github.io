import random
import time

def construct():
    N = 512
    
    # Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)

    def is_valid_set(S):
        """Check if set S contains any 4-AP"""
        for ap in all_aps:
            if all(x in S for x in ap):
                return False
        return True

    def can_add_element(x, S):
        """Check if x can be added to S without creating a 4-AP"""
        # For each AP containing x, check if the other 3 elements are in S
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            count_in_set = 0
            for elem in ap:
                if elem == x:
                    continue
                if elem in S:
                    count_in_set += 1
            if count_in_set == 3:
                return False
        return True

    def greedy_sorted():
        """Standard greedy construction sorted by value"""
        S = set()
        for x in range(1, N + 1):
            if can_add_element(x, S):
                S.add(x)
        return S

    # Generate Base-4 {0,1} set (Salem-Spencer style)
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)
    
    if not is_valid_set(base4_set):
        base4_set = set()

    # Get initial best
    S_greedy = greedy_sorted()
    
    seed_set = set(S_greedy)
    if len(base4_set) > len(seed_set) and is_valid_set(base4_set):
        seed_set = set(base4_set)
        
    best_set = set(seed_set)
    best_size = len(seed_set)

    start_time = time.time()
    
    # Local Search with multiple restarts and randomized candidate ordering
    for restart in range(2000):
        if time.time() - start_time > 55:
            break
            
        current_set = set(seed_set)
        candidates = [i for i in range(1, N + 1) if i not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if can_add_element(c, current_set):
                current_set.add(c)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Try swap-based local search: remove one element, try to add two
    for restart in range(500):
        if time.time() - start_time > 58:
            break
            
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        for i in range(len(elements)):
            if time.time() - start_time > 58:
                break
            remove_elem = elements[i]
            current_set.remove(remove_elem)
            
            # Try to add candidates that were previously blocked
            candidates = [j for j in range(1, N + 1) if j not in current_set]
            random.shuffle(candidates)
            
            added = []
            for c in candidates:
                if can_add_element(c, current_set):
                    current_set.add(c)
                    added.append(c)
            
            # If we added more than we removed, keep the improvement
            if len(added) > 1:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                break
            else:
                # Revert
                current_set.add(remove_elem)
                for a in added:
                    if a in current_set:
                        current_set.remove(a)

    return sorted(best_set)
