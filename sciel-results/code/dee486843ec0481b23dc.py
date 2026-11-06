import time
import random
import math

def construct():
    deadline = time.time() + 50  # Safety margin before 90s kill
    
    N = 256
    M = N + 1  # use 1-indexed: bit i represents element i
    
    # Precompute 3-APs and 4-APs as lists of (a, b, c) / (a, b, c, d)
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each element x, store indices of 3-APs and 4-APs containing x
    elem_to_3aps = [[] for _ in range(M)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
    
    elem_to_4aps = [[] for _ in range(M)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # Bitmask helpers
    def bit(x):
        return 1 << x
    
    def count_3aps_mask(mask):
        count = 0
        for a, b, c in three_aps:
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                count += 1
        return count
    
    def delta_add_mask(mask, x):
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if ((mask >> b) & 1) and ((mask >> c) & 1):
                    delta += 1
            elif x == b:
                if ((mask >> a) & 1) and ((mask >> c) & 1):
                    delta += 1
            else:
                if ((mask >> a) & 1) and ((mask >> b) & 1):
                    delta += 1
        return delta
    
    def is_valid_add_mask(mask, x):
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                    return False
            elif x == b:
                if ((mask >> a) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                    return False
            elif x == c:
                if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> d) & 1):
                    return False
            else:
                if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1):
                    return False
        return True
    
    def is_valid_swap_mask(mask, x_out, x_in):
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            others = []
            if x_in == a:
                others = (b, c, d)
            elif x_in == b:
                others = (a, c, d)
            elif x_in == c:
                others = (a, b, d)
            else:
                others = (a, b, c)
            
            for y in others:
                if y == x_out:
                    break
                if not ((mask >> y) & 1):
                    break
            else:
                return False
        return True
    
    def swap_delta_mask(mask, x_out, x_in):
        # Loss from removing x_out
        loss = 0
        for ap_idx in elem_to_3aps[x_out]:
            a, b, c = three_aps[ap_idx]
            if x_out == a:
                others = (b, c)
            elif x_out == b:
                others = (a, c)
            else:
                others = (a, b)
            if all((mask >> y) & 1 for y in others):
                loss += 1
        
        # Gain from adding x_in (in mask without x_out)
        gain = 0
        for ap_idx in elem_to_3aps[x_in]:
            a, b, c = three_aps[ap_idx]
            if x_in == a:
                others = (b, c)
            elif x_in == b:
                others = (a, c)
            else:
                others = (a, b)
            ok = True
            for y in others:
                if y == x_out:
                    ok = False
                    break
                if not ((mask >> y) & 1):
                    ok = False
                    break
            if ok:
                gain += 1
        
        return gain - loss
    
    # Greedy initialization
    def greedy_init():
        mask = 0
        for x in range(1, N + 1):
            if is_valid_add_mask(mask, x):
                mask |= bit(x)
        return mask
    
    # Simulated annealing on a single mask
    def anneal(start_mask, T, cooling, min_T, steps_budget):
        mask = start_mask
        score = count_3aps_mask(mask)
        best_mask = mask
        best_score = score
        rng = random.Random(42)
        
        for _ in range(steps_budget):
            if T < min_T:
                T = min_T
            
            move_type = rng.random()
            accepted = False
            
            if move_type < 0.35:  # Add
                # Pick a random element not in mask
                x = rng.randint(1, N)
                if not (mask >> x) & 1:
                    if is_valid_add_mask(mask, x):
                        delta = delta_add_mask(mask, x)
                        if delta > 0 or (delta < 0 and rng.random() < math.exp(delta / T)) or (delta == 0 and rng.random() < 0.5):
                            mask |= bit(x)
                            score += delta
                            accepted = True
                            if score > best_score:
                                best_score = score
                                best_mask = mask
            
            elif move_type < 0.55:  # Remove
                if mask:
                    # Pick random element in mask
                    x = rng.randint(1, N)
                    if (mask >> x) & 1:
                        delta = -delta_add_mask(mask, x)
                        if rng.random() < math.exp(delta / T):
                            mask &= ~bit(x)
                            score += delta
                            accepted = True
            
            else:  # Swap
                if mask:
                    x_out = rng.randint(1, N)
                    if (mask >> x_out) & 1:
                        x_in = rng.randint(1, N)
                        if not (mask >> x_in) & 1:
                            if is_valid_swap_mask(mask, x_out, x_in):
                                delta = swap_delta_mask(mask, x_out, x_in)
                                if delta > 0 or rng.random() < math.exp(delta / T):
                                    mask &= ~bit(x_out)
                                    mask |= bit(x_in)
                                    score += delta
                                    accepted = True
                                    if score > best_score:
                                        best_score = score
                                        best_mask = mask
            
            if not accepted:
                T *= cooling
            else:
                T *= cooling
        
        return best_mask, best_score
    
    # Multi-restart simulated annealing
    best_mask = 0
    best_score = -1
    
    # First, greedy init
    gmask = greedy_init()
    gscore = count_3aps_mask(gmask)
    best_mask = gmask
    best_score = gscore
    
    # Run several annealing rounds with different starting points and parameters
    random.seed(42)
    start_time = time.time()
    
    # Phase 1: Anneal from greedy
    remaining = deadline - time.time()
    if remaining > 5:
        steps = max(1000, int(remaining * 500))
        m, s = anneal(gmask, 5.0, 0.999, 0.01, steps)
        if s > best_score:
            best_score = s
            best_mask = m
    
    # Phase 2: Multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Random starting mask: random subset, then fix by removing 4-AP violations
        rmask = 0
        for x in range(1, N + 1):
            if random.random() < 0.5:
                rmask |= bit(x)
        # Remove elements that create 4-APs (greedy removal)
        for x in range(1, N + 1):
            if (rmask >> x) & 1:
                # Check if x is part of a 4-AP in rmask
                temp = rmask & ~bit(x)
                if not is_valid_add_mask(temp, x):
                    rmask &= ~bit(x)
        
        rscore = count_3aps_mask(rmask)
        if rscore > best_score:
            best_score = rscore
            best_mask = rmask
        
        # Anneal from this random start
        steps = max(500, int(remaining * 200))
        T0 = 3.0 + random.random() * 4.0
        m, s = anneal(rmask, T0, 0.998, 0.005, steps)
        if s > best_score:
            best_score = s
            best_mask = m
    
    # Convert mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    if not result:
        result = [1]
    
    return sorted(result)
