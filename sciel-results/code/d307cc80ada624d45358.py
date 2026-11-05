import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 512
    
    # 1. Precompute all 4-APs and mapping from element to AP indices
    # This allows O(1) validity checks via count_in_ap
    ap_list = []
    elem_to_aps = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
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
    
    # State variables for the current search
    in_set = set()
    count_in_ap = [0] * num_aps
    
    def add_element(x):
        """
        Adds x to the set if it doesn't complete any 4-AP.
        Returns True if added, False otherwise.
        """
        # Check if adding x completes any AP
        for ap_idx in elem_to_aps[x]:
            if count_in_ap[ap_idx] == 3:
                return False
        
        # Valid, so add
        in_set.add(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] += 1
        return True

    def remove_element(x):
        """
        Removes x from the set.
        """
        in_set.discard(x)
        for ap_idx in elem_to_aps[x]:
            count_in_ap[ap_idx] -= 1

    def reset_state():
        """
        Clears the current set and counters.
        """
        in_set.clear()
        for i in range(num_aps):
            count_in_ap[i] = 0

    def get_base3_set():
        """
        Returns the set of integers in [1, n] whose base-3 representation
        contains only digits 0 and 1. This is a known 4-AP-free set.
        """
        s = set()
        # Generate all numbers with digits 0/1 in base 3 up to n
        # Max power of 3 is 3^5 = 243, 3^6 = 729 > 512
        # So we can iterate through combinations of bits for positions 0..5
        for mask in range(1, 1 << 6): # 1 to 63 (6 bits)
            val = 0
            for i in range(6):
                if mask & (1 << i):
                    val += (3 ** i)
            if 1 <= val <= n:
                s.add(val)
        return s

    best_size = 0
    best_set = set()
    
    # We will run multiple restarts.
    # Each restart:
    # 1. Initialize (Random Greedy or Base-3 Greedy)
    # 2. Local Search (Swaps)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        reset_state()
        
        # Choose initialization strategy
        # Alternate between random greedy and base-3 greedy to explore different basins
        if restart_count % 2 == 0:
            # Base-3 initialization
            init_candidates = list(get_base3_set())
            # Add remaining elements in random order to "gap fill"
            remaining = [x for x in range(1, n + 1) if x not in init_candidates]
            random.shuffle(remaining)
            candidates = init_candidates + remaining
        else:
            # Random initialization
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            
        # Greedy Phase
        for x in candidates:
            if x not in in_set:
                if add_element(x):
                    pass # Added successfully
        
        current_size = len(in_set)
        
        # Update best if this greedy start is already good
        if current_size > best_size:
            best_size = current_size
            best_set = set(in_set)
            
        # Local Search Phase (Swaps)
        # We try to improve the set by swapping elements.
        # If we can add x, we do it.
        # If x cannot be added, we find a conflicting AP, remove a random element y from it, 
        # and try to add x again.
        
        # Limit steps per restart to ensure we try multiple restarts
        # With fast checks, we can do many steps.
        max_local_steps = 2000 
        steps = 0
        no_improvement_count = 0
        local_best_size = current_size
        
        while steps < max_local_steps and time.time() < deadline:
            steps += 1
            
            # Pick a random element x not in S
            # To avoid O(N) scan every time, we can just pick random 1..n
            # Probability of picking non-member is (n - size)/n
            # If size is ~200, prob is ~0.6. This is efficient enough.
            x = random.randint(1, n)
            if x in in_set:
                continue
                
            # Try to add x directly
            if add_element(x):
                current_size += 1
                if current_size > local_best_size:
                    local_best_size = current_size
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                continue
            
            # If x cannot be added, find a conflicting AP
            # We know add_element failed, so some AP containing x has count 3.
            conflicting_ap_idx = None
            for ap_idx in elem_to_aps[x]:
                if count_in_ap[ap_idx] == 3:
                    conflicting_ap_idx = ap_idx
                    break
                    
            if conflicting_ap_idx is None:
                # Should not happen if add_element logic is consistent
                continue
                
            ap = ap_list[conflicting_ap_idx]
            # The other 3 elements are in the set
            # Pick one to remove
            y = random.choice([t for t in ap if t != x])
            
            # Perform swap: remove y, try to add x
            remove_element(y)
            if add_element(x):
                # Swap successful. Size remains same.
                # This changes the configuration, potentially allowing future additions.
                no_improvement_count += 1
            else:
                # Swap failed, restore y
                add_element(y) # y was removed, so adding it back should be valid as it was before
                # Note: add_element checks validity. Since y was in the set before, 
                # and we only removed y, the state for y's APs is restored.
                # However, add_element increments counts. We must be careful.
                # Actually, remove_element decrements. add_element increments.
                # So calling add_element(y) restores the state exactly.
                
            # Early termination for this local search if stuck for too long
            if no_improvement_count > 500:
                break
                
        # Update global best after local search
        if current_size > best_size:
            best_size = current_size
            best_set = set(in_set)
            
    return sorted(best_set)

# EVOLVE-BLOCK-END
