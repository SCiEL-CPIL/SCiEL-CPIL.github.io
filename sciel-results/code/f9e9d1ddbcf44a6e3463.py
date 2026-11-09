import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-APs
    four_ap_by_elem = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            d0 = a + 3 * d - 1
            mask = (1 << a0) | (1 << b0) | (1 << c0) | (1 << d0)
            four_ap_by_elem[a0].append(mask)
            four_ap_by_elem[b0].append(mask)
            four_ap_by_elem[c0].append(mask)
            four_ap_by_elem[d0].append(mask)
            
    # Precompute 3-AP neighbors
    three_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            three_ap_by_elem[a0].append((b0, c0))
            three_ap_by_elem[b0].append((a0, c0))
            three_ap_by_elem[c0].append((a0, b0))

    def count_3aps(state):
        count = 0
        for x in range(N):
            if (state >> x) & 1:
                for u, v in three_ap_by_elem[x]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        count += 1
        return count // 3

    def is_valid_4ap(state):
        for x in range(N):
            if (state >> x) & 1:
                for mask in four_ap_by_elem[x]:
                    if (state & mask) == mask:
                        return False
        return True

    # --- Phase 1: Fast Greedy Initialization ---
    # Create a good starting point to reduce the search space for SA
    
    def greedy_init():
        state = 0
        available = list(range(N))
        random.shuffle(available)
        
        # Simple greedy: add elements that don't create 4-APs, prioritizing high connectivity?
        # Actually, let's just add all possible elements in a random order that doesn't break validity.
        # This ensures a valid starting set.
        
        for x in available:
            # Check if adding x is valid
            valid = True
            for mask in four_ap_by_elem[x]:
                if (state & mask) == (mask ^ (1 << x)):
                    valid = False
                    break
            if valid:
                state |= (1 << x)
        
        return state

    # --- Phase 2: Simulated Annealing ---
    
    current_state = greedy_init()
    current_score = count_3aps(current_state)
    current_valid = is_valid_4ap(current_state)
    
    best_state = current_state
    best_score = current_score
    
    T = 1.0
    T_min = 0.001
    alpha = 0.9995
    
    # Penalty for invalid states
    penalty = 100.0
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a random move: flip a random bit
        x = random.randint(0, N - 1)
        
        new_state = current_state ^ (1 << x)
        
        # Calculate score change
        # We need to efficiently calculate the change in 3-AP count and validity.
        # For simplicity in this implementation, we recalculate if the state is small, 
        # but N=256 is too large for full recalculation every step.
        # We use the delta approach.
        
        # Delta in 3-APs
        delta_3ap = 0
        if (current_state >> x) & 1:
            # Removing x
            for u, v in three_ap_by_elem[x]:
                if ((current_state >> u) & 1) and ((current_state >> v) & 1):
                    delta_3ap -= 1
        else:
            # Adding x
            for u, v in three_ap_by_elem[x]:
                if ((current_state >> u) & 1) and ((current_state >> v) & 1):
                    delta_3ap += 1
        
        # Delta in validity (4-AP violations)
        # This is harder to do incrementally without precomputed counters.
        # Let's just check validity of the new state? 
        # Checking all 4-APs is O(N * avg_aps_per_elem). 
        # But we can just check the 4-APs involving x.
        
        old_validity_penalty = 0 if current_valid else penalty
        new_valid = True
        if (new_state >> x) & 1:
            # x was added. Check if it completes any 4-AP.
            for mask in four_ap_by_elem[x]:
                if (new_state & mask) == mask:
                    new_valid = False
                    break
        else:
            # x was removed. It might have been part of a 4-AP. 
            # If it was, removing it fixes that violation.
            # But there might be other violations? No, only x changed.
            # So if current_valid was False, it was due to some 4-AP. 
            # If that 4-AP contained x, it's fixed. If not, it's still there.
            # However, we don't track *which* 4-APs are violated.
            # To be safe and correct, we must re-evaluate validity if the state was invalid.
            # Or, we can just check validity of new_state fully if it's potentially invalid.
            
            # Optimization: If current_valid is True, removing x keeps it valid.
            # If current_valid is False, we need to check if new_state is valid.
            # Since we only flip one bit, we can check if the bit flip fixed the violation.
            # But we don't know which violation. 
            # Let's just do a full validity check on new_state if current_valid is False.
            # If current_valid is True, new_state is definitely valid (removing an element can't create a 4-AP).
            
            if current_valid:
                new_valid = True
            else:
                # Full check
                for i in range(N):
                    if (new_state >> i) & 1:
                        for mask in four_ap_by_elem[i]:
                            if (new_state & mask) == mask:
                                new_valid = False
                                break
                        if not new_valid:
                            break
        
        new_validity_penalty = 0 if new_valid else penalty
        
        # Total score change
        # We want to maximize 3-APs and minimize violations.
        # Objective: score - penalty * violations
        # Since we only track binary validity, let's use:
        # Objective = 3AP_count if valid, else -infinity (or large negative)
        # Actually, let's use a penalty term.
        
        current_obj = current_score - old_validity_penalty
        new_obj = (current_score + delta_3ap) - new_validity_penalty
        
        delta_obj = new_obj - current_obj
        
        # Acceptance probability
        if delta_obj > 0:
            accept = True
        else:
            # Avoid math domain error
            if T == 0:
                accept = False
            else:
                accept = random.random() < math.exp(delta_obj / T)
                
        if accept:
            current_state = new_state
            current_score += delta_3ap
            current_valid = new_valid
            
            if current_valid and current_score > best_score:
                best_score = current_score
                best_state = current_state
                
        T *= alpha

    # Final polish: Greedy add any beneficial elements
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for x in range(N):
            if (best_state >> x) & 1:
                continue
            # Check if adding x is valid
            valid = True
            for mask in four_ap_by_elem[x]:
                if (best_state & mask) == (mask ^ (1 << x)):
                    valid = False
                    break
            if not valid:
                continue
            
            # Check if adding x increases 3-AP count
            gain = 0
            for u, v in three_ap_by_elem[x]:
                if ((best_state >> u) & 1) and ((best_state >> v) & 1):
                    gain += 1
            
            if gain > 0:
                best_state |= (1 << x)
                best_score += gain
                changed = True

    # Convert to list of 1-based integers
    result = []
    for i in range(N):
        if (best_state >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)
