import time
import random

def construct():
    N = 512
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N
    # Use 0-indexed: element x maps to bit (x-1)
    aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(1, N - 3 * d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    
    # For each element index (0..N-1), store list of AP indices containing it
    aps_containing = [[] for _ in range(N)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        aps_containing[i0].append(idx)
        aps_containing[i1].append(idx)
        aps_containing[i2].append(idx)
        aps_containing[i3].append(idx)
    
    # For fast checking: given a mask, check if adding bit x (0-indexed) creates a 4-AP
    # We check all APs that contain x: if the other 3 bits are already set, adding x completes a 4-AP
    def can_add_bit(mask, x_idx):
        """Check if adding bit x_idx to mask creates a 4-AP. x_idx is 0-indexed."""
        for ap_idx in aps_containing[x_idx]:
            i0, i1, i2, i3 = aps[ap_idx]
            # Check if the other 3 positions are set in mask
            others = [i0, i1, i2, i3]
            others.remove(x_idx)
            if (mask >> others[0] & 1) and (mask >> others[1] & 1) and (mask >> others[2] & 1):
                return False
        return True
    
    def fast_greedy(order):
        """Greedy: add elements in given order if they don't create a 4-AP."""
        mask = 0
        for x in order:
            x_idx = x - 1  # 0-indexed
            if can_add_bit(mask, x_idx):
                mask |= (1 << x_idx)
        return mask
    
    def local_search(mask, rng, deadline):
        """Local search: try adding blocked elements, then 1-swaps, then 2-swaps."""
        S_size = bin(mask).count('1')
        blocked = [x for x in range(1, N + 1) if not (mask >> (x - 1) & 1)]
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
            
            improved = False
            
            # Phase 1: Try adding any blocked element
            rng.shuffle(blocked)
            for x in blocked:
                if time.time() >= deadline:
                    return mask, S_size
                x_idx = x - 1
                if can_add_bit(mask, x_idx):
                    mask |= (1 << x_idx)
                    S_size += 1
                    blocked.remove(x)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # Phase 2: Try 1-swaps (remove r, add a)
            if not blocked:
                break
            
            a = rng.choice(blocked)
            a_idx = a - 1
            
            S_list = [x for x in range(1, N + 1) if (mask >> (x - 1) & 1)]
            rng.shuffle(S_list)
            
            swap_found = False
            for r in S_list[:30]:
                if time.time() >= deadline:
                    return mask, S_size
                r_idx = r - 1
                mask_without_r = mask & ~(1 << r_idx)
                
                if can_add_bit(mask_without_r, a_idx):
                    # Perform swap
                    mask = mask_without_r | (1 << a_idx)
                    if a in blocked:
                        blocked.remove(a)
                    blocked.append(r)
                    swap_found = True
                    break
            
            if swap_found:
                no_improve_count = 0
                continue
            
            # Phase 3: Try 2-swaps (remove r1, r2, add a1, a2) - only if time permits
            if len(blocked) >= 2 and len(S_list) >= 2:
                a1, a2 = rng.sample(blocked, 2)
                a1_idx, a2_idx = a1 - 1, a2 - 1
                
                # Try removing 2 random S elements
                r_candidates = rng.sample(S_list, min(15, len(S_list)))
                
                two_swap_found = False
                for r1 in r_candidates:
                    if time.time() >= deadline:
                        return mask, S_size
                    r1_idx = r1 - 1
                    mask_w1 = mask & ~(1 << r1_idx)
                    
                    for r2 in r_candidates:
                        if r2 == r1:
                            continue
                        if time.time() >= deadline:
                            return mask, S_size
                        r2_idx = r2 - 1
                        mask_w2 = mask_w1 & ~(1 << r2_idx)
                        
                        # Check if both a1 and a2 can be added
                        if can_add_bit(mask_w2, a1_idx) and can_add_bit(mask_w2 | (1 << a1_idx), a2_idx):
                            # Perform 2-swap
                            mask = mask_w2 | (1 << a1_idx) | (1 << a2_idx)
                            for a in [a1, a2]:
                                if a in blocked:
                                    blocked.remove(a)
                            for r in [r1, r2]:
                                if r not in blocked:
                                    blocked.append(r)
                            two_swap_found = True
                            break
                    if two_swap_found:
                        break
                
                if two_swap_found:
                    no_improve_count = 0
                    continue
            
            no_improve_count += 1
        
        return mask, S_size
    
    # Main: multi-restart with different orderings
    rng = random.Random(42)
    best_mask = 0
    best_size = 0
    
    # Phase 1: Multiple random greedy restarts
    remaining_time = deadline - (time.time() - start_time)
    restart_budget = min(20, max(3, int(remaining_time // 3)))
    
    for restart in range(restart_budget):
        if time.time() >= deadline:
            break
        
        # Different ordering strategies
        if restart == 0:
            order = list(range(1, N + 1))  # Natural order
        elif restart == 1:
            order = list(range(N, 0, -1))  # Reverse order
        elif restart == 2:
            order = list(range(1, N + 1))
            rng.shuffle(order)  # Random
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        
        mask = fast_greedy(order)
        size = bin(mask).count('1')
        
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Local search on best mask
    if time.time() < deadline:
        ls_deadline = min(deadline, time.time() + (deadline - (time.time() - start_time)))
        best_mask, best_size = local_search(best_mask, rng, ls_deadline)
    
    # Phase 3: Additional local search passes if time remains
    if time.time() < deadline - 2:
        best_mask, best_size = local_search(best_mask, rng, deadline)
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1) & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
