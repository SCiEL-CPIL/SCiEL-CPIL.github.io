import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            ap3s.append((a, a + d, a + 2 * d))
    
    ap4s = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map element to AP indices
    elem_to_ap3 = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(ap3s):
        elem_to_ap3[a].append(i)
        elem_to_ap3[b].append(i)
        elem_to_ap3[c].append(i)
        
    elem_to_ap4 = [[] for _ in range(N + 1)]
    for i, (a, b, c, d) in enumerate(ap4s):
        elem_to_ap4[a].append(i)
        elem_to_ap4[b].append(i)
        elem_to_ap4[c].append(i)
        elem_to_ap4[d].append(i)
    
    # Convert to numpy arrays for faster operations
    ap3_array = np.array(ap3s)
    ap4_array = np.array(ap4s)
    
    # Bitmask representation: 4 uint64 words for 256 bits
    def make_mask():
        return np.zeros(4, dtype=np.uint64)
    
    def set_bit(mask, pos):
        mask[pos >> 6] |= (1 << (pos & 63))
    
    def clear_bit(mask, pos):
        mask[pos >> 6] &= ~(1 << (pos & 63))
    
    def get_bit(mask, pos):
        return (mask[pos >> 6] >> (pos & 63)) & 1
    
    def mask_to_set(mask):
        s = set()
        for pos in range(N):
            if get_bit(mask, pos):
                s.add(pos + 1)
        return s
    
    def set_to_mask(s):
        mask = make_mask()
        for x in s:
            set_bit(mask, x - 1)
        return mask
    
    def count_3aps(mask):
        """Count 3-APs using numpy vectorization with bitmask"""
        in_set = np.zeros(N + 1, dtype=bool)
        for pos in range(N):
            if get_bit(mask, pos):
                in_set[pos + 1] = True
        all_in = np.all(in_set[ap3_array], axis=1)
        return int(np.sum(all_in))
    
    def has_4ap(mask):
        """Check if mask contains any 4-AP"""
        in_set = np.zeros(N + 1, dtype=bool)
        for pos in range(N):
            if get_bit(mask, pos):
                in_set[pos + 1] = True
        all_in = np.all(in_set[ap4_array], axis=1)
        return bool(np.any(all_in))
    
    def is_4ap_conflict(mask, pos):
        """Check if adding pos creates a 4-AP"""
        for idx in elem_to_ap4[pos + 1]:
            a, b, c, d = ap4s[idx]
            others = [o for o in (a, b, c, d) if o != pos + 1]
            if all(get_bit(mask, o - 1) for o in others):
                return True
        return False
    
    def delta_3ap_add(mask, pos):
        """Delta in 3-AP count when adding pos"""
        delta = 0
        for idx in elem_to_ap3[pos + 1]:
            a, b, c = ap3s[idx]
            others = [o for o in (a, b, c) if o != pos + 1]
            if get_bit(mask, others[0] - 1) and get_bit(mask, others[1] - 1):
                delta += 1
        return delta
    
    def delta_3ap_remove(mask, pos):
        """Delta in 3-AP count when removing pos (negative)"""
        delta = 0
        for idx in elem_to_ap3[pos + 1]:
            a, b, c = ap3s[idx]
            others = [o for o in (a, b, c) if o != pos + 1]
            if get_bit(mask, others[0] - 1) and get_bit(mask, others[1] - 1):
                delta += 1
        return -delta
    
    def greedy_init():
        """Greedy initialization with random ordering"""
        mask = make_mask()
        order = list(range(N))
        random.shuffle(order)
        
        for pos in order:
            if is_4ap_conflict(mask, pos):
                continue
            delta = delta_3ap_add(mask, pos)
            if delta > 0 or (delta == 0 and random.random() < 0.15):
                set_bit(mask, pos)
        
        return mask
    
    def local_search(mask, time_limit):
        """Steepest-ascent local search with swaps"""
        start = time.time()
        cur_n3 = count_3aps(mask)
        improved = True
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # Try adding elements
            candidates = list(range(N))
            random.shuffle(candidates)
            
            for pos in candidates:
                if get_bit(mask, pos):
                    continue
                if is_4ap_conflict(mask, pos):
                    continue
                delta = delta_3ap_add(mask, pos)
                if delta > 0:
                    set_bit(mask, pos)
                    cur_n3 += delta
                    improved = True
            
            # Try swaps: remove x, add y
            set_positions = [p for p in range(N) if get_bit(mask, p)]
            unset_positions = [p for p in range(N) if not get_bit(mask, p)]
            
            if set_positions and unset_positions:
                # Try a limited number of swaps
                num_swap_trials = min(50, len(set_positions) * len(unset_positions))
                for _ in range(num_swap_trials):
                    x = random.choice(set_positions)
                    y = random.choice(unset_positions)
                    
                    # Check if swap is valid (no 4-AP after swap)
                    # Temporarily remove x
                    clear_bit(mask, x)
                    valid = not is_4ap_conflict(mask, y)
                    set_bit(mask, x)
                    
                    if not valid:
                        continue
                    
                    # Compute delta
                    rem_delta = delta_3ap_remove(mask, x)
                    # After removing x, check adding y
                    clear_bit(mask, x)
                    add_delta = delta_3ap_add(mask, y)
                    set_bit(mask, x)
                    
                    total_delta = add_delta + rem_delta
                    if total_delta > 0:
                        clear_bit(mask, x)
                        set_bit(mask, y)
                        cur_n3 += total_delta
                        improved = True
                        # Update lists
                        set_positions.remove(x)
                        set_positions.append(y)
                        unset_positions.remove(y)
                        unset_positions.append(x)
        
        return mask, cur_n3
    
    def iterated_local_search(time_budget):
        """Iterated local search with perturbations"""
        best_mask = None
        best_n3 = 0
        
        start = time.time()
        
        # Initial greedy + local search
        mask = greedy_init()
        mask, cur_n3 = local_search(mask, 2.0)
        
        if cur_n3 > best_n3:
            best_n3 = cur_n3
            best_mask = mask.copy()
        
        while time.time() - start < time_budget - 5.0:
            # Perturb: remove k random elements
            k = random.randint(2, 8)
            mask = best_mask.copy()
            
            set_positions = [p for p in range(N) if get_bit(mask, p)]
            
            if len(set_positions) < k:
                break
            
            to_remove = random.sample(set_positions, k)
            for pos in to_remove:
                clear_bit(mask, pos)
            
            # Local search
            mask, cur_n3 = local_search(mask, 1.0)
            
            if cur_n3 > best_n3:
                best_n3 = cur_n3
                best_mask = mask.copy()
        
        return best_mask, best_n3
    
    def simulated_annealing(mask, max_iters=30000):
        """Simulated annealing on bitmask representation"""
        cur_n3 = count_3aps(mask)
        best_mask = mask.copy()
        best_n3 = cur_n3
        
        T_start = 8.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        
        set_positions = [p for p in range(N) if get_bit(mask, p)]
        unset_positions = [p for p in range(N) if not get_bit(mask, p)]
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.3 and len(set_positions) < 200:
                # Add
                if not unset_positions:
                    continue
                y = random.choice(unset_positions)
                if is_4ap_conflict(mask, y):
                    continue
                
                delta = delta_3ap_add(mask, y)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    set_bit(mask, y)
                    cur_n3 += delta
                    set_positions.append(y)
                    unset_positions.remove(y)
                    
                    if cur_n3 > best_n3:
                        best_n3 = cur_n3
                        best_mask = mask.copy()
                        
            elif move_type < 0.5 and len(set_positions) > 50:
                # Remove
                x = random.choice(set_positions)
                delta = delta_3ap_remove(mask, x)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    clear_bit(mask, x)
                    cur_n3 += delta
                    set_positions.remove(x)
                    unset_positions.append(x)
                    
                    if cur_n3 > best_n3:
                        best_n3 = cur_n3
                        best_mask = mask.copy()
                        
            else:
                # Swap
                if not set_positions or not unset_positions:
                    continue
                x = random.choice(set_positions)
                y = random.choice(unset_positions)
                
                # Check validity
                clear_bit(mask, x)
                valid = not is_4ap_conflict(mask, y)
                set_bit(mask, x)
                
                if not valid:
                    continue
                
                rem_delta = delta_3ap_remove(mask, x)
                clear_bit(mask, x)
                add_delta = delta_3ap_add(mask, y)
                set_bit(mask, x)
                
                delta = add_delta + rem_delta
                if delta >= 0 or random.random() < np.exp(delta / T):
                    clear_bit(mask, x)
                    set_bit(mask, y)
                    cur_n3 += delta
                    set_positions.remove(x)
                    set_positions.append(y)
                    unset_positions.remove(y)
                    unset_positions.append(x)
                    
                    if cur_n3 > best_n3:
                        best_n3 = cur_n3
                        best_mask = mask.copy()
            
            T *= cooling_rate
        
        return best_mask, best_n3
    
    # Main search loop
    best_mask = None
    best_n3 = -1
    
    # Phase 1: ILS
    time_budget_ils = 40.0
    mask, n3 = iterated_local_search(time_budget_ils)
    if n3 > best_n3:
        best_n3 = n3
        best_mask = mask.copy()
    
    # Phase 2: Simulated annealing from best
    if time.time() - start_time < deadline - 10:
        sa_budget = min(30000, int((deadline - 10 - (time.time() - start_time)) * 1000))
        mask, n3 = simulated_annealing(best_mask, sa_budget)
        if n3 > best_n3:
            best_n3 = n3
            best_mask = mask.copy()
    
    # Phase 3: More ILS restarts if time remains
    while time.time() - start_time < deadline - 8:
        mask = greedy_init()
        mask, n3 = local_search(mask, 1.5)
        if n3 > best_n3:
            best_n3 = n3
            best_mask = mask.copy()
    
    # Final verification
    if time.time() - start_time > deadline - 2:
        return sorted(mask_to_set(best_mask))
    
    # Ensure no 4-APs
    if has_4ap(best_mask):
        # Fallback to base-3 construction
        mask = make_mask()
        for x in range(1, 257):
            y = x - 1
            valid = True
            while y > 0:
                if y % 3 == 2:
                    valid = False
                    break
                y //= 3
            if valid:
                set_bit(mask, x - 1)
        best_mask = mask
    
    return sorted(mask_to_set(best_mask))
