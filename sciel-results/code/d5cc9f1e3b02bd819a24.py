import time
import random
import math

N = 256
DEADLINE = 85.0  # 90s limit, keep 5s safety margin

def _precompute():
    """
    Precompute 3-APs and 4-APs with bitmask representations for fast operations.
    """
    aps3 = []
    aps4 = []
    elem_to_aps3 = [[] for _ in range(N + 1)]
    elem_to_aps4 = [[] for _ in range(N + 1)]
    
    # Generate 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
            elem_to_aps3[a].append((1 << b) | (1 << c))
            elem_to_aps3[b].append((1 << a) | (1 << c))
            elem_to_aps3[c].append((1 << a) | (1 << b))
    
    # Generate 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps4.append((a, b, c, e))
            elem_to_aps4[a].append((1 << b) | (1 << c) | (1 << e))
            elem_to_aps4[b].append((1 << a) | (1 << c) | (1 << e))
            elem_to_aps4[c].append((1 << a) | (1 << b) | (1 << e))
            elem_to_aps4[e].append((1 << a) | (1 << b) | (1 << c))
    
    return aps3, aps4, elem_to_aps3, elem_to_aps4

def _initial_greedy_mask(elem_to_aps3, elem_to_aps4):
    """
    Greedy construction: add x if it doesn't complete a 4-AP.
    Prioritize elements with more 3-AP connections.
    """
    A_mask = 0
    # Score each element by number of 3-APs it participates in
    ap_count = [len(elem_to_aps3[i]) for i in range(N + 1)]
    
    # Process in order of decreasing 3-AP potential, but still check validity
    candidates = sorted(range(1, N + 1), key=lambda x: -ap_count[x])
    
    for x in candidates:
        valid = True
        for mask_other3 in elem_to_aps4[x]:
            if (A_mask & mask_other3) == mask_other3:
                valid = False
                break
        if valid:
            A_mask |= (1 << x)
    
    return A_mask

def construct():
    start_time = time.time()
    
    # Precompute
    aps3, aps4, elem_to_aps3, elem_to_aps4 = _precompute()
    
    def calc_score(mask):
        score = 0
        for a, b, c in aps3:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                score += 1
        return score

    def is_valid_add(y, mask, removed_x=0):
        """Check if adding y to mask (with x removed) is 4-AP free."""
        for mask_other3 in elem_to_aps4[y]:
            # Check if all other 3 elements are in mask minus x
            effective_mask = mask & ~(1 << removed_x) if removed_x else mask
            if (effective_mask & mask_other3) == mask_other3:
                return False
        return True

    def calc_delta_swap(x, y, mask):
        """Calculate score change for removing x and adding y."""
        loss = 0
        gain = 0
        
        # Loss: 3-APs containing x where other 2 are in mask
        for mask_other2 in elem_to_aps3[x]:
            if (mask & mask_other2) == mask_other2:
                loss += 1
                
        # Gain: 3-APs containing y where other 2 are in mask minus x
        for mask_other2 in elem_to_aps3[y]:
            if (mask & mask_other2) == mask_other2:
                if not (mask_other2 & (1 << x)):
                    gain += 1
                    
        return gain - loss

    # Initial greedy set
    A_mask = _initial_greedy_mask(elem_to_aps3, elem_to_aps4)
    current_score = calc_score(A_mask)
    best_score = current_score
    best_mask = A_mask
    
    # Simulated annealing
    def simulate_annealing(start_mask, start_score, time_budget):
        mask = start_mask
        score = start_score
        best_local_score = score
        best_local_mask = mask
        
        # Temperature schedule
        T0 = 10.0
        Tend = 0.1
        # We'll do iterations until time runs out
        # Estimate iterations per second first with a small batch
        
        # Precompute element lists for faster access
        def get_in_out(mask):
            in_set = [i for i in range(1, N + 1) if (mask >> i) & 1]
            out_set = [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
            return in_set, out_set
        
        in_set, out_set = get_in_out(mask)
        
        # Simulated annealing loop
        elapsed = time.time() - start_time
        remaining = time_budget - (elapsed - start_time)
        if remaining < 1.0:
            return best_local_mask, best_local_score
        
        # Determine number of iterations we can afford
        # Each iteration: pick random x from in_set, y from out_set, check validity, calc delta
        # Let's just run until time is up
        
        T = T0
        cooling_rate = 0.9999  # Will adjust based on time
        
        while (time.time() - start_time) < start_time + time_budget - 0.5:
            # Cool down
            T *= cooling_rate
            if T < Tend:
                T = Tend
            
            # Pick random swap
            if not in_set or not out_set:
                break
                
            x = random.choice(in_set)
            y = random.choice(out_set)
            
            # Check validity
            valid = True
            for mask_other3 in elem_to_aps4[y]:
                effective_mask = mask & ~(1 << x)
                if (effective_mask & mask_other3) == mask_other3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate delta
            delta = calc_delta_swap(x, y, mask)
            
            # Acceptance criterion
            accept = False
            if delta >= 0:
                accept = True
            else:
                # Accept with probability exp(delta/T)
                if T > 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Apply swap
                mask &= ~(1 << x)
                mask |= (1 << y)
                score += delta
                
                # Update in_set/out_set
                in_set.remove(x)
                in_set.append(y)
                out_set.remove(y)
                out_set.append(x)
                
                if score > best_local_score:
                    best_local_score = score
                    best_local_mask = mask
        
        return best_local_mask, best_local_score
    
    # Phase 1: Quick hill climbing to get a good starting point
    def hill_climb(mask, score, time_budget):
        local_mask = mask
        local_score = score
        best_local_score = score
        best_local_mask = mask
        
        while (time.time() - start_time) < start_time + time_budget - 0.5:
            improved = False
            in_set = [i for i in range(1, N + 1) if (local_mask >> i) & 1]
            out_set = [i for i in range(1, N + 1) if not ((local_mask >> i) & 1)]
            random.shuffle(in_set)
            random.shuffle(out_set)
            
            for x in in_set:
                if (time.time() - start_time) > start_time + time_budget - 0.5:
                    break
                for y in out_set:
                    # Check validity
                    valid = True
                    for mask_other3 in elem_to_aps4[y]:
                        effective_mask = local_mask & ~(1 << x)
                        if (effective_mask & mask_other3) == mask_other3:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    delta = calc_delta_swap(x, y, local_mask)
                    if delta > 0:
                        local_mask &= ~(1 << x)
                        local_mask |= (1 << y)
                        local_score += delta
                        improved = True
                        if local_score > best_local_score:
                            best_local_score = local_score
                            best_local_mask = local_mask
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_local_mask, best_local_score
    
    # Run hill climbing first (quick improvement)
    hill_time = min(5.0, (DEADLINE - (time.time() - start_time)) * 0.2)
    A_mask, current_score = hill_climb(A_mask, current_score, hill_time)
    if current_score > best_score:
        best_score = current_score
        best_mask = A_mask
    
    # Phase 2: Simulated annealing for more thorough search
    remaining_time = DEADLINE - (time.time() - start_time) - 1.0
    if remaining_time > 2.0:
        sa_mask, sa_score = simulate_annealing(A_mask, current_score, remaining_time)
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
    
    # Phase 3: Final hill climb from best found
    final_time = DEADLINE - (time.time() - start_time) - 0.5
    if final_time > 1.0:
        final_mask, final_score = hill_climb(best_mask, best_score, final_time)
        if final_score > best_score:
            best_score = final_score
            best_mask = final_mask
    
    return sorted([i for i in range(1, N + 1) if (best_mask >> i) & 1])

# EVOLVE-BLOCK-START
# Entry point is construct()
# EVOLVE-BLOCK-END
