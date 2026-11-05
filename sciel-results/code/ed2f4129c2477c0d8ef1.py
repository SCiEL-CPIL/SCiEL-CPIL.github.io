import time
import random

def construct():
    N = 128
    deadline = time.time() + 50.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # For each position, precompute which APs it's in and its role (0,1,2,3)
    ap_indices_for_pos = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j, pos in enumerate(ap):
            ap_indices_for_pos[pos].append((idx, j))
    
    # Precompute violation masks:
    # For an AP with colors (c0,c1,c2,c3), it's a violation if < 3 distinct colors
    # We want to know: given 3 of the 4 colors, which 4th colors cause a violation?
    # 
    # Let's think about when {c0,c1,c2,c3} has < 3 distinct elements:
    # Case 1: All 4 same: c0=c1=c2=c3
    # Case 2: Exactly 2 distinct:
    #   - 3 of one, 1 of another (e.g., c0=c1=c2 != c3)
    #   - 2 of one, 2 of another (e.g., c0=c1 != c2=c3)
    #
    # For bitmask approach: for each AP index and position j in that AP,
    # we want a 128-bit mask where bit k is set if assigning color k to position j
    # would make this AP a violation, GIVEN the other 3 positions' colors.
    #
    # But this depends on the other 3 colors, so we can't precompute a static mask.
    # Instead, let's compute violations on the fly but efficiently.
    
    # Alternative: Precompute for each AP, for each of the 4 positions, 
    # a function that given the other 3 colors returns the forbidden colors.
    # 
    # Actually, let's just use a different approach:
    # For each position, maintain a count of violations it's involved in.
    # When we change a position, we update the violation counts for all APs it's in.
    
    def count_violations_fast(colors):
        """Count total violations using precomputed AP list."""
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if < 3 distinct colors
            # Fast check without set creation:
            if c0 == c1:
                if c2 == c3:
                    # Two pairs or all same
                    if c0 == c2:
                        count += 1  # All same
                    # else: c0=c1, c2=c3, c0!=c2 -> 2 distinct, violation
                    else:
                        count += 1
                elif c2 == c0:
                    # c0=c1=c2, c3 different -> 2 distinct
                    count += 1
                elif c3 == c0:
                    # c0=c1=c3, c2 different -> 2 distinct
                    count += 1
                elif c2 == c3:
                    # c0=c1, c2=c3, c0!=c2 -> 2 distinct
                    count += 1
                # else: c0=c1, all others different from each other and from c0 -> 3 distinct, OK
            elif c0 == c2:
                if c1 == c3:
                    # c0=c2, c1=c3, c0!=c1 -> 2 distinct
                    count += 1
                elif c1 == c0:
                    # c0=c1=c2, c3 different -> 2 distinct
                    count += 1
                elif c3 == c0:
                    # c0=c2=c3, c1 different -> 2 distinct
                    count += 1
                # else: OK
            elif c0 == c3:
                if c1 == c2:
                    # c0=c3, c1=c2, c0!=c1 -> 2 distinct
                    count += 1
                elif c1 == c0:
                    # c0=c1=c3, c2 different -> 2 distinct
                    count += 1
                elif c2 == c0:
                    # c0=c2=c3, c1 different -> 2 distinct
                    count += 1
                # else: OK
            elif c1 == c2:
                if c0 == c3:
                    # c1=c2, c0=c3, c0!=c1 -> 2 distinct
                    count += 1
                elif c0 == c1:
                    # c0=c1=c2, c3 different -> 2 distinct
                    count += 1
                elif c3 == c1:
                    # c1=c2=c3, c0 different -> 2 distinct
                    count += 1
                # else: OK
            elif c1 == c3:
                if c0 == c2:
                    # c1=c3, c0=c2, c0!=c1 -> 2 distinct
                    count += 1
                elif c0 == c1:
                    # c0=c1=c3, c2 different -> 2 distinct
                    count += 1
                elif c2 == c1:
                    # c1=c2=c3, c0 different -> 2 distinct
                    count += 1
                # else: OK
            elif c2 == c3:
                if c0 == c1:
                    # c2=c3, c0=c1, c0!=c2 -> 2 distinct
                    count += 1
                elif c0 == c2:
                    # c0=c2=c3, c1 different -> 2 distinct
                    count += 1
                elif c1 == c2:
                    # c1=c2=c3, c0 different -> 2 distinct
                    count += 1
                # else: OK
            # else: all distinct or 3 distinct -> OK
        return count
    
    def greedy_construct(k, seed):
        """Try to build a valid coloring with k colors using greedy + lookahead."""
        random.seed(seed)
        colors = [None] * N
        
        # Try multiple orderings
        for attempt in range(3):
            colors = [None] * N
            order = list(range(N))
            if attempt == 1:
                random.shuffle(order)
            elif attempt == 2:
                # Reverse order
                order = order[::-1]
            
            for pos in order:
                if time.time() > deadline:
                    return None, None
                
                # Determine which colors are forbidden for this position
                # A color c is forbidden if assigning it creates a violation in any AP containing pos
                forbidden = [False] * k
                
                for ap_idx, j in ap_indices_for_pos[pos]:
                    ap = aps[ap_idx]
                    # Get the other 3 positions
                    other_positions = [p for p in range(4) if p != j]
                    other_colors = [colors[ap[p]] for p in other_positions]
                    
                    # If any other position is unassigned, we can't determine constraints yet
                    # For greedy, we only consider APs where all other 3 are assigned
                    if any(c is None for c in other_colors):
                        continue
                    
                    oc0, oc1, oc2 = other_colors
                    
                    # For each candidate color c, check if {c, oc0, oc1, oc2} has < 3 distinct
                    for c in range(k):
                        if forbidden[c]:
                            continue
                        # Check distinct count of {c, oc0, oc1, oc2}
                        # Fast check:
                        vals = (c, oc0, oc1, oc2)
                        if vals[0] == vals[1]:
                            if vals[2] == vals[3]:
                                if vals[0] == vals[2]:
                                    forbidden[c] = True  # All same
                                else:
                                    forbidden[c] = True  # Two pairs
                            elif vals[2] == vals[0]:
                                forbidden[c] = True  # 3 same
                            elif vals[3] == vals[0]:
                                forbidden[c] = True  # 3 same
                            elif vals[2] == vals[3]:
                                forbidden[c] = True  # Two pairs
                        elif vals[0] == vals[2]:
                            if vals[1] == vals[3]:
                                forbidden[c] = True
                            elif vals[1] == vals[0]:
                                forbidden[c] = True
                            elif vals[3] == vals[0]:
                                forbidden[c] = True
                        elif vals[0] == vals[3]:
                            if vals[1] == vals[2]:
                                forbidden[c] = True
                            elif vals[1] == vals[0]:
                                forbidden[c] = True
                            elif vals[2] == vals[0]:
                                forbidden[c] = True
                        elif vals[1] == vals[2]:
                            if vals[0] == vals[3]:
                                forbidden[c] = True
                            elif vals[0] == vals[1]:
                                forbidden[c] = True
                            elif vals[3] == vals[1]:
                                forbidden[c] = True
                        elif vals[1] == vals[3]:
                            if vals[0] == vals[2]:
                                forbidden[c] = True
                            elif vals[0] == vals[1]:
                                forbidden[c] = True
                            elif vals[2] == vals[1]:
                                forbidden[c] = True
                        elif vals[2] == vals[3]:
                            if vals[0] == vals[1]:
                                forbidden[c] = True
                            elif vals[0] == vals[2]:
                                forbidden[c] = True
                            elif vals[1] == vals[2]:
                                forbidden[c] = True
                        # else: OK, not forbidden by this AP
                    # Early exit if all forbidden
                    if all(forbidden):
                        break
                
                if all(forbidden):
                    break  # Dead end, try different order/seed
                
                # Pick the least forbidden color (or lowest index if tie)
                # Actually, pick a color that minimizes future constraints
                # Simple: pick lowest index not forbidden
                chosen = None
                for c in range(k):
                    if not forbidden[c]:
                        chosen = c
                        break
                
                if chosen is None:
                    break  # Dead end
                
                colors[pos] = chosen
            
            # Check if we got a complete assignment
            if all(c is not None for c in colors):
                # Verify it's valid
                if count_violations_fast(colors) == 0:
                    return colors, attempt
            # If incomplete, try next attempt
        
        return None, None
    
    def local_search(colors, k, max_steps=5000):
        """Local search to fix violations."""
        if colors is None:
            return None
        
        current_violations = count_violations_fast(colors)
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        # Track violation count per AP for efficient updates
        ap_violation = [False] * M
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            vals = (c0, c1, c2, c3)
            is_violation = False
            if vals[0] == vals[1]:
                if vals[2] == vals[3]:
                    is_violation = True
                elif vals[2] == vals[0] or vals[3] == vals[0] or vals[2] == vals[3]:
                    is_violation = True
            elif vals[0] == vals[2]:
                if vals[1] == vals[3] or vals[1] == vals[0] or vals[3] == vals[0]:
                    is_violation = True
            elif vals[0] == vals[3]:
                if vals[1] == vals[2] or vals[1] == vals[0] or vals[2] == vals[0]:
                    is_violation = True
            elif vals[1] == vals[2]:
                if vals[0] == vals[3] or vals[0] == vals[1] or vals[3] == vals[1]:
                    is_violation = True
            elif vals[1] == vals[3]:
                if vals[0] == vals[2] or vals[0] == vals[1] or vals[2] == vals[1]:
                    is_violation = True
            elif vals[2] == vals[3]:
                if vals[0] == vals[1] or vals[0] == vals[2] or vals[1] == vals[2]:
                    is_violation = True
            ap_violation[idx] = is_violation
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random position involved in a violation
            violating_aps = [idx for idx, v in enumerate(ap_violation) if v]
            if not violating_aps:
                break
            
            ap_idx = random.choice(violating_aps)
            ap = aps[ap_idx]
            pos = random.choice(ap)
            
            current_val = colors[pos]
            
            # Try all k colors, find one that reduces violations
            best_new_val = current_val
            best_delta = 0
            
            for new_val in range(k):
                if new_val == current_val:
                    continue
                
                # Compute delta in violations if we change pos to new_val
                delta = 0
                for ap_i, j in ap_indices_for_pos[pos]:
                    other_ap = aps[ap_i]
                    # Get current violation status
                    was_violation = ap_violation[ap_i]
                    
                    # Compute new colors for this AP
                    new_colors = [colors[other_ap[p]] for p in range(4)]
                    new_colors[j] = new_val
                    
                    nc0, nc1, nc2, nc3 = new_colors
                    vals = (nc0, nc1, nc2, nc3)
                    is_violation = False
                    if vals[0] == vals[1]:
                        if vals[2] == vals[3]:
                            is_violation = True
                        elif vals[2] == vals[0] or vals[3] == vals[0] or vals[2] == vals[3]:
                            is_violation = True
                    elif vals[0] == vals[2]:
                        if vals[1] == vals[3] or vals[1] == vals[0] or vals[3] == vals[0]:
                            is_violation = True
                    elif vals[0] == vals[3]:
                        if vals[1] == vals[2] or vals[1] == vals[0] or vals[2] == vals[0]:
                            is_violation = True
                    elif vals[1] == vals[2]:
                        if vals[0] == vals[3] or vals[0] == vals[1] or vals[3] == vals[1]:
                            is_violation = True
                    elif vals[1] == vals[3]:
                        if vals[0] == vals[2] or vals[0] == vals[1] or vals[2] == vals[1]:
                            is_violation = True
                    elif vals[2] == vals[3]:
                        if vals[0] == vals[1] or vals[0] == vals[2] or vals[1] == vals[2]:
                            is_violation = True
                    
                    if is_violation and not was_violation:
                        delta += 1
                    elif not is_violation and was_violation:
                        delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_val = new_val
            
            # Apply the best move
            if best_new_val != current_val:
                colors[pos] = best_new_val
                current_violations += best_delta
                
                # Update ap_violation for affected APs
                for ap_i, j in ap_indices_for_pos[pos]:
                    other_ap = aps[ap_i]
                    nc0, nc1, nc2, nc3 = colors[other_ap[0]], colors[other_ap[1]], colors[other_ap[2]], colors[other_ap[3]]
                    vals = (nc0, nc1, nc2, nc3)
                    is_violation = False
                    if vals[0] == vals[1]:
                        if vals[2] == vals[3]:
                            is_violation = True
                        elif vals[2] == vals[0] or vals[3] == vals[0] or vals[2] == vals[3]:
                            is_violation = True
                    elif vals[0] == vals[2]:
                        if vals[1] == vals[3] or vals[1] == vals[0] or vals[3] == vals[0]:
                            is_violation = True
                    elif vals[0] == vals[3]:
                        if vals[1] == vals[2] or vals[1] == vals[0] or vals[2] == vals[0]:
                            is_violation = True
                    elif vals[1] == vals[2]:
                        if vals[0] == vals[3] or vals[0] == vals[1] or vals[3] == vals[1]:
                            is_violation = True
                    elif vals[1] == vals[3]:
                        if vals[0] == vals[2] or vals[0] == vals[1] or vals[2] == vals[1]:
                            is_violation = True
                    elif vals[2] == vals[3]:
                        if vals[0] == vals[1] or vals[0] == vals[2] or vals[1] == vals[2]:
                            is_violation = True
                    ap_violation[ap_i] = is_violation
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    break
            
            # If stuck, try a random move (simulated annealing style)
            elif step % 100 == 0:
                pos = random.randint(0, N - 1)
                new_val = random.randint(0, k - 1)
                if new_val != colors[pos]:
                    colors[pos] = new_val
                    # Recompute violations for affected APs
                    for ap_i, j in ap_indices_for_pos[pos]:
                        other_ap = aps[ap_i]
                        nc0, nc1, nc2, nc3 = colors[other_ap[0]], colors[other_ap[1]], colors[other_ap[2]], colors[other_ap[3]]
                        vals = (nc0, nc1, nc2, nc3)
                        is_violation = False
                        if vals[0] == vals[1]:
                            if vals[2] == vals[3]:
                                is_violation = True
                            elif vals[2] == vals[0] or vals[3] == vals[0] or vals[2] == vals[3]:
                                is_violation = True
                        elif vals[0] == vals[2]:
                            if vals[1] == vals[3] or vals[1] == vals[0] or vals[3] == vals[0]:
                                is_violation = True
                        elif vals[0] == vals[3]:
                            if vals[1] == vals[2] or vals[1] == vals[0] or vals[2] == vals[0]:
                                is_violation = True
                        elif vals[1] == vals[2]:
                            if vals[0] == vals[3] or vals[0] == vals[1] or vals[3] == vals[1]:
                                is_violation = True
                        elif vals[1] == vals[3]:
                            if vals[0] == vals[2] or vals[0] == vals[1] or vals[2] == vals[1]:
                                is_violation = True
                        elif vals[2] == vals[3]:
                            if vals[0] == vals[1] or vals[0] == vals[2] or vals[1] == vals[2]:
                                is_violation = True
                        ap_violation[ap_i] = is_violation
                    current_violations = sum(1 for v in ap_violation if v)
        
        return best_colors if best_violations == 0 else None
    
    # Search for minimum k
    best_solution = None
    
    for k in range(3, 30):
        if time.time() > deadline:
            break
        
        # Try greedy construction with multiple seeds
        found = False
        for seed in range(50):
            if time.time() > deadline:
                break
            
            colors, attempt = greedy_construct(k, seed)
            if colors is not None:
                # Verify
                if count_violations_fast(colors) == 0:
                    best_solution = colors
                    found = True
                    break
        
        if found:
            break
        
        # If greedy didn't work, try random init + local search
        if time.time() > deadline:
            break
        
        for seed in range(20):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            colors = [random.randint(0, k - 1) for _ in range(N)]
            result = local_search(colors, k, max_steps=3000)
            if result is not None:
                best_solution = result
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: use more colors
        for k in range(30, 128):
            if time.time() > deadline:
                break
            colors = list(range(k)) + [0] * (N - k)
            if count_violations_fast(colors) == 0:
                best_solution = colors
                break
            # Try local search
            result = local_search(colors, k, max_steps=1000)
            if result is not None:
                best_solution = result
                break
    
    if best_solution is None:
        # Ultimate fallback: all distinct colors
        best_solution = list(range(N))
    
    return best_solution
