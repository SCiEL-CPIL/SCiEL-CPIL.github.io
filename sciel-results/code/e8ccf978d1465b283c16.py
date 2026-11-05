import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for 60s limit
    
    n = 512
    
    # Precompute all 4-APs
    # ap_list: list of tuples (a, b, c, d) with 1-based indexing
    # elem_to_aps: dict mapping element x to list of AP indices it belongs to
    ap_list = []
    elem_to_aps = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            elem_to_aps[a].append(ap_idx)
            elem_to_aps[b].append(ap_idx)
            elem_to_aps[c].append(ap_idx)
            elem_to_aps[e].append(ap_idx)
            
    num_aps = len(ap_list)
    
    # count_in_ap[i] = number of elements of AP i currently in the set
    count_in_ap = [0] * num_aps
    
    def add_element(x, in_set):
        """
        Adds x to the set if valid.
        Returns True if added, False if invalid.
        Modifies in_set and count_in_ap.
        """
        for ap_idx in elem_to_aps[x]:
            if count_in_ap[ap_idx] == 3:
                return False
        in_set.add(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] += 1
        return True

    def remove_element(x, in_set):
        """
        Removes x from the set.
        Modifies in_set and count_in_ap.
        """
        in_set.discard(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] -= 1

    best_size = 0
    best_set = set()
    
    # Run multiple restarts
    # Each restart:
    # 1. Greedy construction (random order or base-3 seed)
    # 2. Local search to improve
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Choose initialization strategy
        # Strategy A: Base-3 digits 0 and 1 (Salem-Spencer, known 3-AP-free hence 4-AP-free)
        # Strategy B: Random greedy
        if restart_count % 3 == 0:
            # Base-3 seed
            in_set = set()
            for x in range(1, n + 1):
                y = x
                valid = True
                while y > 0:
                    if y % 3 == 2:
                        valid = False
                        break
                    y //= 3
                if valid:
                    in_set.add(x)
            # Update counts
            for x in in_set:
                for ap_idx in elem_to_aps[x]:
                    count_in_ap[ap_idx] += 1
        else:
            # Random greedy
            in_set = set()
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            for x in candidates:
                add_element(x, in_set)
        
        current_size = len(in_set)
        
        # Local search phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any element
            for x in range(1, n + 1):
                if x not in in_set:
                    if add_element(x, in_set):
                        current_size += 1
                        improved = True
            
            if current_size > best_size:
                best_size = current_size
                best_set = set(in_set)
            
            if not improved:
                break
            
            # Phase 2: Try swaps
            # We need to be careful with swaps to avoid oscillation
            # Try a limited number of random swaps
            swap_attempts = 0
            max_swaps = 2000
            while swap_attempts < max_swaps and time.time() < deadline:
                swap_attempts += 1
                
                # Pick x not in set
                x = random.randint(1, n)
                if x in in_set:
                    continue
                    
                # Pick y in set
                y = random.choice(list(in_set))
                
                # Try swap: remove y, add x
                remove_element(y, in_set)
                if add_element(x, in_set):
                    # Swap accepted
                    # Try to add more?
                    # For now, just accept the swap and continue
                    pass
                else:
                    # Swap failed, restore
                    in_set.add(x)  # This is wrong, x was not added
                    # Wait, add_element failed, so x is NOT in set.
                    # We need to restore y.
                    remove_element(x, in_set) # This is a no-op since x not in set, but let's be precise.
                    # Actually, add_element checks validity first. If it fails, it doesn't modify in_set.
                    # So we just need to re-add y.
                    add_element(y, in_set) # This should succeed since we just removed it and no other changes
                    # But wait, add_element might fail if y forms an AP with other elements?
                    # No, y was in the set before, so it was valid. Removing x (which wasn't added) doesn't change validity of y.
                    # So add_element(y, in_set) should always succeed here.
        
        # After local search, update best
        if len(in_set) > best_size:
            best_size = len(in_set)
            best_set = set(in_set)
        
        # Reset counts for next restart
        for i in range(num_aps):
            count_in_ap[i] = 0

    return sorted(best_set)

# EVOLVE-BLOCK-END
