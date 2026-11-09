import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    N = 256

    # Precompute 4-AP conflict graph: for each element, list of other elements that form a 4-AP with it
    # Actually, for MIS we need edges between pairs that cannot both be selected.
    # Two elements x, y conflict if there exist z, w such that {x,y,z,w} is a 4-AP.
    # More efficiently: for each element x, precompute the set of elements that conflict with x.
    
    # For each element x, find all 4-APs containing x. For each such 4-AP, x conflicts with the other 3 elements.
    conflict = [set() for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            d0 = a + 3 * d - 1
            # a0, b0, c0, d0 form a 4-AP
            conflict[a0].update([b0, c0, d0])
            conflict[b0].update([a0, c0, d0])
            conflict[c0].update([a0, b0, d0])
            conflict[d0].update([a0, b0, c0])
    
    # Precompute 3-AP completion potential for each element
    # three_ap_count[x] = number of pairs (u,v) in [0,N) such that (x,u,v) forms a 3-AP
    # Actually, we want: if we select x, how many 3-APs can it complete with already-selected elements?
    # For branch-and-bound, we need dynamic weights based on current selection.
    
    # Precompute for each element x, the list of (u,v) pairs such that x completes a 3-AP with u and v
    # i.e., (u, x, v) or (v, x, u) or (x, u, v) where x is one of the three positions
    three_ap_pairs = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            # (a0, b0, c0) is a 3-AP
            # a0 completes with b0, c0
            three_ap_pairs[a0].append((b0, c0))
            # b0 completes with a0, c0
            three_ap_pairs[b0].append((a0, c0))
            # c0 completes with a0, b0
            three_ap_pairs[c0].append((a0, b0))
    
    # Density-greedy initialization
    def greedy_init():
        state = 0
        # Weight = number of 3-AP pairs it can form, minus a penalty for conflict degree
        # Use a simple heuristic: prioritize elements with high 3-AP potential and low conflict
        order = list(range(N))
        random.shuffle(order)
        
        # Sort by initial "value" = len(three_ap_pairs[x]) - 0.1 * len(conflict[x])
        order.sort(key=lambda x: len(three_ap_pairs[x]) - 0.1 * len(conflict[x]), reverse=True)
        
        for x in order:
            # Check if x can be added (no conflict with already selected)
            can_add = True
            for y in conflict[x]:
                if (state >> y) & 1:
                    can_add = False
                    break
            if can_add:
                state |= (1 << x)
        
        return state
    
    # Count 3-APs in a state
    def count_3aps(state):
        count = 0
        for x in range(N):
            if (state >> x) & 1:
                for u, v in three_ap_pairs[x]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        count += 1
        return count // 3  # Each 3-AP is counted 3 times (once per element)
    
    # Branch and bound for maximum weight independent set
    # Weight of selecting x given current state S: number of 3-APs completed by adding x to S
    def weight_of(x, state):
        w = 0
        for u, v in three_ap_pairs[x]:
            if ((state >> u) & 1) and ((state >> v) & 1):
                w += 1
        return w
    
    # Upper bound: sum of weights of all remaining non-conflicting vertices
    def upper_bound(remaining, state, current_weight):
        bound = current_weight
        for x in remaining:
            bound += weight_of(x, state)
        return bound
    
    # Branch and bound solver
    def branch_and_bound(initial_state, time_limit):
        best_state = initial_state
        best_weight = count_3aps(initial_state)
        
        # Get list of elements in initial state
        selected = [i for i in range(N) if (initial_state >> i) & 1]
        remaining = [i for i in range(N) if not (initial_state >> i) & 1]
        
        # Sort remaining by potential weight (descending) for better pruning
        remaining.sort(key=lambda x: len(three_ap_pairs[x]), reverse=True)
        
        def bb(remaining, state, current_weight):
            nonlocal best_state, best_weight
            
            if time.time() > time_limit:
                return
            
            if not remaining:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_state = state
                return
            
            # Upper bound
            ub = current_weight
            for x in remaining:
                ub += weight_of(x, state)
            
            if ub <= best_weight:
                return
            
            # Try each remaining element
            for i, x in enumerate(remaining):
                if time.time() > time_limit:
                    return
                
                # Check if x conflicts with any selected element
                can_add = True
                for y in conflict[x]:
                    if (state >> y) & 1:
                        can_add = False
                        break
                
                if not can_add:
                    continue
                
                # Branch: include x
                new_state = state | (1 << x)
                new_weight = current_weight + weight_of(x, state)
                new_remaining = [r for r in remaining[i+1:] if not (conflict[x] & set([r]))]
                
                bb(new_remaining, new_state, new_weight)
                
                # Branch: exclude x
                # (implicitly handled by continuing the loop)
        
        bb(remaining, initial_state, best_weight)
        return best_state, best_weight
    
    # Generate multiple greedy initializations
    best_state = 0
    best_score = 0
    
    num_inits = 10
    for r in range(num_inits):
        if time.time() > deadline - 20.0:
            break
        
        state = greedy_init()
        score = count_3aps(state)
        
        if score > best_score:
            best_score = score
            best_state = state
    
    # Run branch and bound on the best initialization
    if time.time() < deadline - 5.0:
        bb_state, bb_score = branch_and_bound(best_state, deadline - 2.0)
        if bb_score > best_score:
            best_score = bb_score
            best_state = bb_state
    
    # Final polish: try to add any element that increases 3-AP count without creating 4-AP
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for x in range(N):
            if (best_state >> x) & 1:
                continue
            # Check 4-AP conflict
            can_add = True
            for y in conflict[x]:
                if (best_state >> y) & 1:
                    can_add = False
                    break
            if not can_add:
                continue
            delta = weight_of(x, best_state)
            if delta > 0:
                best_state |= (1 << x)
                best_score += delta
                changed = True
    
    # Convert to list of 1-based integers
    result = []
    for i in range(N):
        if (best_state >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)
