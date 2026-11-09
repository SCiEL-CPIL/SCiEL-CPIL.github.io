import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit (2s buffer)
    
    N = 256
    
    # Precompute 4-APs
    # For each 4-AP (a, a+d, a+2d, a+3d), store the bitmask of the 4 elements.
    # Also, for each element x, store a list of 4-AP masks that contain x.
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
            
    # Precompute 3-AP neighbors for fast delta calculation
    # three_ap_by_elem[x] contains list of (u, v) such that (x, u, v) form a 3-AP
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
        # Check if state contains any 4-AP
        # We can iterate through all 4-APs and check if mask is subset of state
        # But we don't have a list of ALL masks, only by element.
        # Alternative: Iterate through elements in state, check their 4-APs.
        # To avoid double counting, we can just check if ANY 4-AP is fully contained.
        # For speed in SA, we want a fast check.
        # Let's precompute a list of all 4-AP masks.
        pass # We will use a different validity check strategy inside SA loop for speed, 
             # or precompute all masks.
        
    # Precompute list of all 4-AP masks for global validity check if needed, 
    # but for SA moves (single bit flip), we can update violation count incrementally.
    
    # Let's stick to the incremental update strategy for speed.
    # State: bitmask
    # We maintain a count of "active" 4-APs? No, we want 0.
    # If we flip a bit x, we only need to check 4-APs containing x.
    
    def get_delta_3aps(state, x, is_add):
        """
        Calculate change in 3-AP count if we flip bit x.
        is_add: True if adding x, False if removing x.
        """
        delta = 0
        for u, v in three_ap_by_elem[x]:
            if is_add:
                # Adding x: creates 3-AP if u and v are already in state
                if ((state >> u) & 1) and ((state >> v) & 1):
                    delta += 1
            else:
                # Removing x: destroys 3-AP if u and v are already in state
                if ((state >> u) & 1) and ((state >> v) & 1):
                    delta -= 1
        return delta

    def check_4ap_violation(state, x, is_add):
        """
        Check if flipping bit x (add or remove) creates or resolves a 4-AP violation.
        Returns True if the move is VALID (no new 4-AP created).
        If is_add: Check if adding x completes any 4-AP.
        If is_remove: Removing x cannot create a 4-AP, so always valid.
        """
        if not is_add:
            return True
        
        # Check if adding x creates a 4-AP
        for mask in four_ap_by_elem[x]:
            # If all OTHER bits in mask are already in state, then adding x completes the 4-AP.
            if (state & mask) == (mask ^ (1 << x)):
                return False
        return True

    # --- Phase 1: Greedy Initialization ---
    # Start with a good initial state using the greedy method from the seed
    state = 0
    current_score = 0
    available = [i for i in range(N)]
    
    # Simple greedy: pick element with max gain, tie-break random
    while available and time.time() < deadline - 30.0:
        best_x = -1
        best_gain = -1
        best_rand = -1.0
        
        for x in available:
            if not check_4ap_violation(state, x, True):
                continue
            gain = get_delta_3aps(state, x, True)
            noise = random.random() * 0.5
            if gain > best_gain or (gain == best_gain and noise > best_rand):
                best_gain = gain
                best_rand = noise
                best_x = x
        
        if best_x == -1:
            break
            
        state |= (1 << best_x)
        current_score += best_gain
        available.remove(best_x)

    # Ensure initial state is valid (greedy should guarantee this, but just in case)
    # And count score
    current_score = count_3aps(state)
    
    best_state = state
    best_score = current_score
    
    # --- Phase 2: Simulated Annealing ---
    # Start temperature high, cool exponentially
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.005  # Adjust based on time budget
    
    # Estimate number of steps we can afford
    # Each step is O(1) or O(deg) for delta calc, and O(deg) for 4-AP check.
    # deg is small (~20-30). So ~50-100 ops per step.
    # 30 seconds left. Python can do ~10M-50M simple ops/sec.
    # Let's say 500k-1M steps.
    
    iter_count = 0
    max_iters = 1000000
    
    while time.time() < deadline - 2.0 and iter_count < max_iters:
        iter_count += 1
        
        # Pick a random element to flip
        x = random.randint(0, N - 1)
        is_add = not ((state >> x) & 1)
        
        # Check validity
        if not check_4ap_violation(state, x, is_add):
            continue
        
        # Calculate score change
        delta = get_delta_3aps(state, x, is_add)
        
        # Acceptance probability
        if delta >= 0:
            accept = True
        else:
            # Boltzmann factor
            # Use log to avoid underflow? Or just exp.
            # T is small later, so exp(delta/T) will be very small.
            try:
                prob = math.exp(delta / T_start)
            except OverflowError:
                prob = 0.0
            accept = random.random() < prob
        
        if accept:
            if is_add:
                state |= (1 << x)
            else:
                state &= ~(1 << x)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_state = state
        
        # Cool down
        # Exponential cooling: T = T_start * exp(-cooling_rate * iter)
        # Or linear? Exponential is often better for SA.
        # Let's use exponential cooling towards T_end.
        # T = T_end + (T_start - T_end) * exp(-cooling_rate * iter)
        T_current = T_end + (T_start - T_end) * math.exp(-cooling_rate * iter_count)
        
        # If we want to restart or reset, we could, but simple SA is fine.
        
        # Occasionally, if stuck, do a "kick" by resetting to best_state and heating up?
        # For now, keep it simple.
        
        # Update T for next iteration
        T_start = T_current # Reuse variable for current T in next loop? 
        # No, better to calculate T directly in acceptance step.
        # Let's refactor: Calculate T at start of loop.
        
    # Final polish: Try to add any beneficial elements
    # This is a quick greedy pass on the best state found
    final_state = best_state
    final_score = best_score
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for x in range(N):
            if (final_state >> x) & 1:
                continue
            if not check_4ap_violation(final_state, x, True):
                continue
            delta = get_delta_3aps(final_state, x, True)
            if delta > 0:
                final_state |= (1 << x)
                final_score += delta
                changed = True

    # Convert to list of 1-based integers
    result = []
    for i in range(N):
        if (final_state >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)
