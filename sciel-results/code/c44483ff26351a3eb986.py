import time
import random

N = 256
DEADLINE = 80.0  # safety margin before 90s kill

def _precompute():
    """Precompute 3-AP and 4-AP partner lists."""
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

def _count_3aps(in_set, ap3_partners):
    """Count 3-APs in the set."""
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    def try_update(in_set, score):
        nonlocal best_score, best_set
        if score > best_score:
            best_score = score
            best_set = in_set[:]
    
    def sa_run(in_set, score, time_limit):
        """Run simulated annealing on a given state."""
        nonlocal best_score, best_set, start_time
        
        T = 5.0
        T_min = 0.05
        cooling = 0.9998
        
        present = [i for i in range(1, N + 1) if in_set[i]]
        absent = [i for i in range(1, N + 1) if not in_set[i]]
        
        while time.time() - start_time < time_limit:
            # Choose move type: 60% swap, 25% add, 15% remove
            r = random.random()
            
            if r < 0.6:
                # Swap
                if not present or not absent:
                    T = max(T * cooling, T_min)
                    continue
                x = random.choice(present)
                y = random.choice(absent)
                
                # Check 4-AP constraint for adding y to S \ {x}
                valid = True
                for j, k, l in ap4_partners[y]:
                    if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
                        valid = False
                        break
                
                if not valid:
                    T = max(T * cooling, T_min)
                    continue
                
                # Calculate delta
                delta_rem = 0
                for j, k in ap3_partners[x]:
                    if in_set[j] and in_set[k]:
                        delta_rem += 1
                        
                delta_add = 0
                for j, k in ap3_partners[y]:
                    if (in_set[j] and j != x) and (in_set[k] and k != x):
                        delta_add += 1
                        
                delta = delta_add - delta_rem
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    in_set[x] = False
                    in_set[y] = True
                    score += delta
                    present.remove(x)
                    present.append(y)
                    absent.remove(y)
                    absent.append(x)
                    try_update(in_set, score)
                    
            elif r < 0.85:
                # Add
                if not absent:
                    T = max(T * cooling, T_min)
                    continue
                y = random.choice(absent)
                
                valid = True
                for j, k, l in ap4_partners[y]:
                    if in_set[j] and in_set[k] and in_set[l]:
                        valid = False
                        break
                if not valid:
                    T = max(T * cooling, T_min)
                    continue
                    
                delta = 0
                for j, k in ap3_partners[y]:
                    if in_set[j] and in_set[k]:
                        delta += 1
                        
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    in_set[y] = True
                    score += delta
                    present.append(y)
                    absent.remove(y)
                    try_update(in_set, score)
                    
            else:
                # Remove
                if not present:
                    T = max(T * cooling, T_min)
                    continue
                x = random.choice(present)
                
                delta = 0
                for j, k in ap3_partners[x]:
                    if in_set[j] and in_set[k]:
                        delta += 1
                delta = -delta
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    in_set[x] = False
                    score += delta
                    present.remove(x)
                    absent.append(x)
                    try_update(in_set, score)
                    
            T = max(T * cooling, T_min)
    
    # Multiple initialization strategies
    init_strategies = []
    
    # Strategy 1: Greedy random (shuffle and add if valid)
    for trial in range(3):
        in_set = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            valid = True
            for j, k, l in ap4_partners[x]:
                if in_set[j] and in_set[k] and in_set[l]:
                    valid = False
                    break
            if valid:
                in_set[x] = True
        init_strategies.append(in_set)
    
    # Strategy 2: Density-based (start with odd numbers, try to add)
    for density in [0.3, 0.4, 0.5]:
        in_set = [False] * (N + 1)
        for x in range(1, N + 1):
            if random.random() < density:
                valid = True
                for j, k, l in ap4_partners[x]:
                    if in_set[j] and in_set[k] and in_set[l]:
                        valid = False
                        break
                if valid:
                    in_set[x] = True
        init_strategies.append(in_set)
    
    # Strategy 3: Start from best found, perturb
    # (will be added after first SA run)
    
    # Run SA on each initialization
    for i, init in enumerate(init_strategies):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        in_set = init[:]
        score = _count_3aps(in_set, ap3_partners)
        try_update(in_set, score)
        
        # Allocate time: more time for later strategies
        time_left = DEADLINE - (time.time() - start_time)
        time_per_run = min(time_left / (len(init_strategies) - i + 1), 15.0)
        sa_run(in_set, score, start_time + time.time() - start_time + time_per_run)
        
        # Add perturbed version of best as new strategy
        if i < len(init_strategies) - 1:
            perturbed = best_set[:]
            # Randomly flip a few elements
            for _ in range(5):
                x = random.randint(1, N)
                if perturbed[x]:
                    # Try to remove
                    perturbed[x] = False
                else:
                    # Try to add if valid
                    valid = True
                    for j, k, l in ap4_partners[x]:
                        if perturbed[j] and perturbed[k] and perturbed[l]:
                            valid = False
                            break
                    if valid:
                        perturbed[x] = True
            init_strategies.append(perturbed)
    
    # Final intensive SA on best
    if time.time() - start_time < DEADLINE - 2.0:
        in_set = best_set[:]
        score = _count_3aps(in_set, ap3_partners)
        time_left = DEADLINE - (time.time() - start_time)
        sa_run(in_set, score, time.time() + time_left - 1.0)
    
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
