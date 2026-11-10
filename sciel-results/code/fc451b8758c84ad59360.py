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

def _count_3aps_slow(in_set, ap3_partners):
    """Slow count for verification or initial setup."""
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def _count_3aps_bitmask(in_set, ap3_partners):
    """Count 3-APs using bitmask representation."""
    mask = 0
    for i in range(1, N + 1):
        if in_set[i]:
            mask |= (1 << i)
    
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def _greedy_upper_bound(available, in_set, ap3_partners, ap4_partners):
    """
    Greedy upper bound: repeatedly pick the vertex with highest 3-AP degree
    that doesn't create a 4-AP, and sum its contributions.
    This gives an upper bound on the max 3-AP count achievable from 'available'.
    """
    if not available:
        return 0
    
    # Work with a copy to avoid mutation
    remaining = list(available)
    temp_set = in_set[:]
    
    bound = 0
    while remaining:
        # Find vertex with max 3-AP degree among remaining
        best_idx = -1
        best_deg = -1
        for idx, v in enumerate(remaining):
            deg = 0
            for j, k in ap3_partners[v]:
                if in_set[j] or j in remaining:
                    if in_set[k] or k in remaining:
                        deg += 1
            if deg > best_deg:
                best_deg = deg
                best_idx = idx
        
        if best_idx < 0:
            break
        
        v = remaining.pop(best_idx)
        bound += best_deg
        
        # Check if adding v creates any 4-AP
        valid = True
        for j, k, l in ap4_partners[v]:
            if (in_set[j] or j in remaining) and (in_set[k] or k in remaining) and (in_set[l] or l in remaining):
                # This would create a 4-AP, so we can't add v
                valid = False
                break
        
        if not valid:
            # Still add to bound as an upper bound (optimistic)
            pass
    
    return bound

def _branch_and_bound(ap3_partners, ap4_partners, deadline, best_score_ref, best_set_ref):
    """
    Branch-and-bound: build set by deciding include/exclude for each vertex.
    Upper bound: current_score + sum of 3-AP degrees of remaining vertices.
    """
    # Initial state: no vertices decided
    in_set = [False] * (N + 1)
    current_score = 0
    remaining = list(range(1, N + 1))
    
    # Precompute initial 3-AP degrees (how many 3-APs each vertex participates in)
    # When we add a vertex, its contribution is the number of 3-APs where the other two are already in_set
    # For upper bound, we use the total degree as an optimistic estimate
    
    def _current_3ap_contribution(v):
        """Number of 3-APs completed by adding v to current in_set."""
        count = 0
        for j, k in ap3_partners[v]:
            if in_set[j] and in_set[k]:
                count += 1
        return count
    
    def _upper_bound(rem):
        """
        Upper bound on additional 3-APs from remaining vertices.
        For each remaining vertex, count how many 3-APs it could complete
        with other remaining vertices (optimistic: assume all others are added).
        Each 3-AP is counted once per endpoint, so divide by 3.
        """
        if not rem:
            return 0
        
        # Build set of remaining for fast lookup
        rem_set = set(rem)
        
        total = 0
        for v in rem:
            for j, k in ap3_partners[v]:
                # Count if both j and k are in in_set OR in rem
                # (optimistic: assume all rem vertices will be added)
                j_in = in_set[j] or j in rem_set
                k_in = in_set[k] or k in rem_set
                if j_in and k_in:
                    total += 1
        
        return total // 3
    
    def _is_valid_addition(v):
        """Check if adding v to current in_set creates any 4-AP."""
        for j, k, l in ap4_partners[v]:
            if in_set[j] and in_set[k] and in_set[l]:
                return False
        return True
    
    def _bbsolve(rem, cur_score, depth):
        nonlocal best_score_ref, best_set_ref
        
        if time.time() > deadline:
            return
        
        if not rem:
            if cur_score > best_score_ref[0]:
                best_score_ref[0] = cur_score
                best_set_ref[0] = in_set[:]
            return
        
        # Upper bound pruning
        ub = cur_score + _upper_bound(rem)
        if ub <= best_score_ref[0]:
            return
        
        # Choose next vertex to branch on
        # Heuristic: pick vertex with highest 3-AP degree in current context
        best_v = -1
        best_deg = -1
        for v in rem:
            deg = _current_3ap_contribution(v)
            # Also consider potential with remaining
            rem_set_local = set(rem)
            for j, k in ap3_partners[v]:
                if (in_set[j] or j in rem_set_local) and (in_set[k] or k in rem_set_local):
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v < 0:
            return
        
        v = best_v
        rem_idx = rem.index(v)
        new_rem = rem[:rem_idx] + rem[rem_idx+1:]
        
        # Branch 1: Include v
        if _is_valid_addition(v):
            contrib = _current_3ap_contribution(v)
            in_set[v] = True
            _bbsolve(new_rem, cur_score + contrib, depth + 1)
            in_set[v] = False
        
        # Branch 2: Exclude v
        _bbsolve(new_rem, cur_score, depth + 1)
    
    _bbsolve(remaining, 0, 0)

def _local_search_applied(initial_set, ap3_partners, ap4_partners, deadline):
    """
    Apply local search (swap, add, remove) to improve the initial set.
    Returns (best_score, best_set).
    """
    in_set = initial_set[:]
    present = [i for i in range(1, N + 1) if in_set[i]]
    absent = [i for i in range(1, N + 1) if not in_set[i]]
    
    score = _count_3aps_slow(in_set, ap3_partners)
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
    
    # Phase 1: Branch-and-bound to get a good initial solution
    bb_deadline = start_time + 10.0
    best_score_ref = [-1]
    best_set_ref = [[False] * (N + 1)]
    
    if time.time() < bb_deadline:
        _branch_and_bound(ap3_partners, ap4_partners, bb_deadline, best_score_ref, best_set_ref)
    
    if best_score_ref[0] > best_score:
        best_score = best_score_ref[0]
        best_set = best_set_ref[0]
    
    # Phase 2: Local search from the B&B result and from greedy initializations
    remaining_time = DEADLINE - (time.time() - start_time)
    
    # Local search from B&B result
    if best_score > -1 and remaining_time > 5:
        ls_deadline = start_time + remaining_time - 2
        ls_score, ls_set = _local_search_applied(best_set, ap3_partners, ap4_partners, ls_deadline)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Additional local search from greedy initializations
    remaining_time = DEADLINE - (time.time() - start_time)
    num_greedy = min(3, int(remaining_time // 5)) if remaining_time > 5 else 0
    
    for _ in range(num_greedy):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        # Greedy initialization
        greedy_set = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            for j, k, l in ap4_partners[x]:
                if greedy_set[j] and greedy_set[k] and greedy_set[l]:
                    valid = False
                    break
            if valid:
                greedy_set[x] = True
        
        # Local search from greedy
        ls_deadline = start_time + (DEADLINE - (time.time() - start_time) - 2)
        ls_score, ls_set = _local_search_applied(greedy_set, ap3_partners, ap4_partners, ls_deadline)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
