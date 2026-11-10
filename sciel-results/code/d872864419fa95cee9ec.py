import time
import random

N = 256
DEADLINE = 55.0

def _precompute():
    """
    Precompute data structures for fast incremental updates.
    Returns:
    ap3_partners: list of lists of tuples (j, k) for each i in 1..N
    ap4_partners: list of lists of tuples (j, k, l) for each i in 1..N
    """
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Update partners for a, b, c
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Update partners for a, b, c, e
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

def _greedy_init(ap3_partners, ap4_partners, randomize=False):
    """
    Greedy initialization: repeatedly add the element that maximizes
    3-AP gain without creating a 4-AP.
    """
    in_set = [False] * (N + 1)
    remaining = list(range(1, N + 1))
    
    if randomize:
        random.shuffle(remaining)
    
    # Precompute initial 3-AP degrees
    deg = [0] * (N + 1)
    for v in range(1, N + 1):
        deg[v] = len(ap3_partners[v])
    
    while remaining:
        # Find best candidate
        best_v = -1
        best_gain = -1
        for v in remaining:
            # Check 4-AP validity
            valid = True
            for j, k, l in ap4_partners[v]:
                if in_set[j] and in_set[k] and in_set[l]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Calculate gain
            gain = 0
            for j, k in ap3_partners[v]:
                if in_set[j] and in_set[k]:
                    gain += 1
            
            if gain > best_gain or (gain == best_gain and randomize and random.random() < 0.3):
                best_gain = gain
                best_v = v
        
        if best_v < 0:
            break
        
        in_set[best_v] = True
        remaining.remove(best_v)
        
        # Update degrees
        for j, k in ap3_partners[best_v]:
            if in_set[j]:
                deg[j] -= 1
            if in_set[k]:
                deg[k] -= 1
    
    return in_set

def _parallel_tempering(ap3_partners, ap4_partners, deadline, num_replicas=8):
    """
    Parallel tempering with multiple replicas at different temperatures.
    Hot replicas explore, cold replicas exploit, periodic swaps transfer solutions.
    """
    # Initialize replicas with diverse starting points
    replicas = []
    for i in range(num_replicas):
        if i == 0:
            # First replica: greedy
            s = _greedy_init(ap3_partners, ap4_partners, randomize=False)
        elif i == 1:
            # Second replica: randomized greedy
            s = _greedy_init(ap3_partners, ap4_partners, randomize=True)
        else:
            # Other replicas: random subsets
            s = [False] * (N + 1)
            for v in range(1, N + 1):
                if random.random() < 0.5:
                    s[v] = True
            # Ensure no 4-APs
            for v in range(1, N + 1):
                if s[v]:
                    for j, k, l in ap4_partners[v]:
                        if s[j] and s[k] and s[l]:
                            s[v] = False
                            break
        
        score = _count_3aps(s, ap3_partners)
        replicas.append({
            'set': s,
            'score': score,
            'T': 1.0 * (1.5 ** i)  # Geometric temperature spacing
        })
    
    best_score = max(r['score'] for r in replicas)
    best_set = max(replicas, key=lambda r: r['score'])['set'][:]
    
    swap_interval = 50  # Swap every 50 iterations
    iteration = 0
    
    while time.time() < deadline:
        for r in replicas:
            if time.time() >= deadline:
                break
            
            s = r['set']
            T = r['T']
            
            # Random move: swap, add, or remove
            move_type = random.random()
            present = [i for i in range(1, N + 1) if s[i]]
            absent = [i for i in range(1, N + 1) if not s[i]]
            
            if move_type < 0.5 and present and absent:
                # Swap
                x = random.choice(present)
                y = random.choice(absent)
                
                # Check 4-AP validity for adding y to S\{x}
                valid = True
                for j, k, l in ap4_partners[y]:
                    if (s[j] and j != x) and (s[k] and k != x) and (s[l] and l != x):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate delta
                delta_rem = 0
                for j, k in ap3_partners[x]:
                    if s[j] and s[k]:
                        delta_rem += 1
                        
                delta_add = 0
                for j, k in ap3_partners[y]:
                    if (s[j] and j != x) and (s[k] and k != x):
                        delta_add += 1
                        
                delta = delta_add - delta_rem
                
                # Accept with probability min(1, exp(delta/T))
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    s[x] = False
                    s[y] = True
                    r['score'] += delta
                    if r['score'] > best_score:
                        best_score = r['score']
                        best_set = s[:]
                        
            elif move_type < 0.8 and absent:
                # Add
                y = random.choice(absent)
                
                valid = True
                for j, k, l in ap4_partners[y]:
                    if s[j] and s[k] and s[l]:
                        valid = False
                        break
                if not valid:
                    continue
                    
                delta = 0
                for j, k in ap3_partners[y]:
                    if s[j] and s[k]:
                        delta += 1
                        
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    s[y] = True
                    r['score'] += delta
                    if r['score'] > best_score:
                        best_score = r['score']
                        best_set = s[:]
                        
            elif present:
                # Remove
                x = random.choice(present)
                
                delta = 0
                for j, k in ap3_partners[x]:
                    if s[j] and s[k]:
                        delta += 1
                delta = -delta
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    s[x] = False
                    r['score'] += delta
                    if r['score'] > best_score:
                        best_score = r['score']
                        best_set = s[:]
            
            iteration += 1
        
        # Periodic swap: exchange solutions between adjacent temperature levels
        if iteration % swap_interval == 0:
            for i in range(num_replicas - 1):
                r1 = replicas[i]
                r2 = replicas[i + 1]
                
                # Metropolis acceptance for swap
                delta_E = r1['score'] - r2['score']
                delta_inv_T = 1.0 / r1['T'] - 1.0 / r2['T']
                
                if delta_E * delta_inv_T > 0 or random.random() < pow(2.718281828, delta_E * delta_inv_T):
                    # Swap
                    r1['set'], r2['set'] = r2['set'], r1['set']
                    r1['score'], r2['score'] = r2['score'], r1['score']
    
    return best_score, best_set

def _local_search_applied(initial_set, ap3_partners, ap4_partners, deadline):
    """
    Apply local search (swap, add, remove) to improve the initial set.
    Returns (best_score, best_set).
    """
    in_set = initial_set[:]
    present = [i for i in range(1, N + 1) if in_set[i]]
    absent = [i for i in range(1, N + 1) if not in_set[i]]
    
    score = _count_3aps(in_set, ap3_partners)
    best_score = score
    best_set = in_set[:]
    
    T = 10.0
    T_min = 0.05
    cooling = 0.9998
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.6 and present and absent:
            # Swap
            x_idx = random.randrange(len(present))
            y_idx = random.randrange(len(absent))
            x = present[x_idx]
            y = absent[y_idx]
            
            # Check 4-AP validity for adding y to S\{x}
            valid = True
            for j, k, l in ap4_partners[y]:
                if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
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
                present[x_idx] = y
                absent[y_idx] = x
                score += delta
                if score > best_score:
                    best_score = score
                    best_set = in_set[:]
                    
        elif r < 0.85 and absent:
            # Add
            y_idx = random.randrange(len(absent))
            y = absent[y_idx]
            
            valid = True
            for j, k, l in ap4_partners[y]:
                if in_set[j] and in_set[k] and in_set[l]:
                    valid = False
                    break
            if not valid:
                T *= cooling
                continue
                
            delta = 0
            for j, k in ap3_partners[y]:
                if in_set[j] and in_set[k]:
                    delta += 1
                    
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                in_set[y] = True
                present.append(y)
                absent.pop(y_idx)
                score += delta
                if score > best_score:
                    best_score = score
                    best_set = in_set[:]
                    
        elif present:
            # Remove
            x_idx = random.randrange(len(present))
            x = present[x_idx]
            
            delta = 0
            for j, k in ap3_partners[x]:
                if in_set[j] and in_set[k]:
                    delta += 1
            delta = -delta
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                in_set[x] = False
                present.pop(x_idx)
                absent.append(x)
                score += delta
                if score > best_score:
                    best_score = score
                    best_set = in_set[:]
                    
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_score, best_set

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    # Phase 1: Parallel tempering for main exploration
    pt_deadline = start_time + 40.0
    pt_score, pt_set = _parallel_tempering(ap3_partners, ap4_partners, pt_deadline, num_replicas=8)
    
    if pt_score > best_score:
        best_score = pt_score
        best_set = pt_set[:]
    
    # Phase 2: Local search refinement from best PT result
    remaining_time = DEADLINE - (time.time() - start_time)
    if best_score > -1 and remaining_time > 5:
        ls_deadline = start_time + remaining_time - 2
        ls_score, ls_set = _local_search_applied(best_set, ap3_partners, ap4_partners, ls_deadline)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Phase 3: Additional local search from greedy initializations if time permits
    remaining_time = DEADLINE - (time.time() - start_time)
    num_greedy = min(2, int(remaining_time // 5)) if remaining_time > 5 else 0
    
    for _ in range(num_greedy):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        # Greedy initialization
        greedy_set = _greedy_init(ap3_partners, ap4_partners, randomize=True)
        
        # Local search from greedy
        ls_deadline = start_time + (DEADLINE - (time.time() - start_time) - 2)
        ls_score, ls_set = _local_search_applied(greedy_set, ap3_partners, ap4_partners, ls_deadline)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
