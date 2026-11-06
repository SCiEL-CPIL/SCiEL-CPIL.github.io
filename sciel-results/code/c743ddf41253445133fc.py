import time
import random

def construct():
    # Deadline guard: 90s limit, keep 10s margin -> 80s
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 128
    
    # Precompute all 4-term arithmetic progressions
    # AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 128
    # Note: problem says positions 1..128? 
    # "n=128 integer color labels in 0..127"
    # "positions" usually implies 1-based in math, but Python is 0-based.
    # The seed code uses `range(1, 129)` and `colors[x-1...]`, implying 1-based indexing in logic, 0-based in list.
    # Let's stick to 0-based indices 0..127 for the list, corresponding to positions 1..128.
    # AP on positions 1..128: a, a+d, a+2d, a+3d.
    # In 0-based indices: i, i+d, i+2d, i+3d where 0 <= i and i+3d <= 127.
    
    aps = []
    aps_by_pos = [[] for _ in range(n)]
    
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for pos in ap:
                aps_by_pos[pos].append(ap_idx)
                
    num_aps = len(aps)
    
    def is_violated(colors, ap_idx):
        a, b, c, d = aps[ap_idx]
        # Count distinct colors
        # We need at least 3 distinct.
        # Violation if distinct < 3.
        c0, c1, c2, c3 = colors[a], colors[b], colors[c], colors[d]
        # Check if distinct count <= 2
        # This happens if:
        # 1. All same
        # 2. 3 same, 1 diff
        # 3. 2 same, 2 same (pairs)
        # Easiest: len(set) < 3
        return len({c0, c1, c2, c3}) < 3

    def count_violations(colors):
        v = 0
        for i in range(num_aps):
            if is_violated(colors, i):
                v += 1
        return v

    def local_search(k, time_limit):
        # Initialize random coloring
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        # Precompute initial violation status
        viol_status = [False] * num_aps
        total_violations = 0
        for i in range(num_aps):
            if is_violated(colors, i):
                viol_status[i] = True
                total_violations += 1
        
        if total_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = total_violations
        
        # Simulated Annealing parameters
        temperature = 1.0
        cooling_rate = 0.999
        min_temperature = 0.01
        
        while time.time() < time_limit:
            if temperature < min_temperature:
                temperature = min_temperature
                
            # Pick random position
            i = random.randint(0, n - 1)
            old_color = colors[i]
            
            # Pick new color different from old
            # To ensure we explore, pick random from 0..k-1
            if k == 1:
                new_color = 0
            else:
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    new_color = (new_color + 1) % k
            
            # Calculate delta in violations
            # Only APs containing i can change
            delta_v = 0
            affected_aps = aps_by_pos[i]
            
            # We need to revert colors[i] to check old status, then set to new
            # But we have viol_status array.
            # Let's just re-evaluate the affected APs.
            
            # Save old colors state for affected APs? No, just use current colors array.
            # 1. Remove old violations for affected APs
            # 2. Set new color
            # 3. Add new violations for affected APs
            
            # Actually, simpler:
            # Calculate current violations for affected APs based on current colors (with old_color at i)
            # Then calculate with new_color at i.
            
            # Since k is small, and affected APs ~ 100-200, this is fast.
            
            # Optimization: Just flip the color, check, flip back if needed?
            # No, we want to accept/reject.
            
            # Let's compute the change.
            # Current state: colors[i] = old_color
            # New state: colors[i] = new_color
            
            # We can't easily "subtract" violations without knowing the exact status.
            # We maintain viol_status.
            
            # Step 1: Determine which affected APs are currently violated.
            # Step 2: Temporarily set colors[i] = new_color.
            # Step 3: Determine which affected APs are now violated.
            # Step 4: Calculate delta = new_count - old_count.
            
            # To avoid modifying global state too much during evaluation, we can do this:
            # For each ap in affected_aps:
            #   old_viol = viol_status[ap]
            #   new_viol = is_violated_with_color(colors, ap, i, new_color)
            #   if old_viol and not new_viol: delta -= 1
            #   if not old_viol and new_viol: delta += 1
            
            # Helper to check violation assuming colors[i] is val
            def is_violated_assuming(colors, ap_idx, pos_idx, val):
                a, b, c, d = aps[ap_idx]
                c0 = val if a == pos_idx else colors[a]
                c1 = val if b == pos_idx else colors[b]
                c2 = val if c == pos_idx else colors[c]
                c3 = val if d == pos_idx else colors[d]
                return len({c0, c1, c2, c3}) < 3

            delta_v = 0
            for ap_idx in affected_aps:
                old_v = viol_status[ap_idx]
                new_v = is_violated_assuming(colors, ap_idx, i, new_color)
                if old_v and not new_v:
                    delta_v -= 1
                elif not old_v and new_v:
                    delta_v += 1
            
            new_total = total_violations + delta_v
            
            # Acceptance criteria
            accept = False
            if new_total < total_violations:
                accept = True
            elif new_total == total_violations:
                accept = True
            else:
                # Metropolis criterion
                prob = pow(2.718281828, -delta_v / temperature)
                if random.random() < prob:
                    accept = True
            
            if accept:
                # Update state
                colors[i] = new_color
                for ap_idx in affected_aps:
                    old_v = viol_status[ap_idx]
                    new_v = is_violated_assuming(colors, ap_idx, i, new_color) # Re-calc to be safe, or use logic from above
                    # Actually, we already computed new_v in the loop above. 
                    # But we didn't store it. Let's just re-calc or store it.
                    # For simplicity and correctness, re-calc is safer if logic is complex.
                    # But 100 APs * 4 lookups is fine.
                    viol_status[ap_idx] = is_violated(colors, ap_idx)
                total_violations = new_total
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    
                if total_violations == 0:
                    return colors
                
                temperature *= cooling_rate
            else:
                # Reject
                temperature *= cooling_rate
            
        return best_colors

    # Try to find a solution with increasing number of colors
    # Start with 3 colors, as it's the minimum possible for "at least 3 distinct"
    # If 3 is impossible, try 4, etc.
    
    best_result = None
    best_k = 128 # Fallback
    
    # We will allocate time for each k
    # If k=3 fails quickly, we can try more.
    # Let's try k=3, 4, 5, 6.
    
    for k in range(3, 7):
        time_limit = time.time() + (deadline - time.time()) / (7 - k)
        # Allocate proportional time
        if time.time() > deadline - 5:
            break
            
        res = local_search(k, time_limit)
        if res is not None:
            # Verify
            if count_violations(res) == 0:
                best_result = res
                best_k = k
                break # Found minimal k, stop
        
        if best_result is None and res is not None:
            # Keep best partial result if we run out of time
            if count_violations(res) < count_violations(best_result if best_result else [0]*n):
                 best_result = res
                 best_k = k # This is not the true k, but just a container

    if best_result is not None:
        return best_result
    
    # Fallback: Greedy construction like seed, but maybe slightly better?
    # Or just return a random valid one if we found one in best_result?
    # If no valid one found, return the one with least violations?
    # The task says "Minimize the number of labels actually used".
    # If no valid coloring is found, the score is invalid? 
    # "An invalid result ... scores 0".
    # So we MUST return a valid coloring.
    # If local search fails to find 0 violations, we need a fallback that guarantees validity.
    # A simple valid coloring: use distinct colors for all? No, that uses 128 colors.
    # A periodic coloring?
    # Let's use a greedy approach that ensures validity, even if it uses many colors.
    
    # Greedy:
    # For each position x, assign the smallest color that doesn't violate any AP ending at x or involving x?
    # The seed program does this. Let's just use the seed logic as a guaranteed valid fallback.
    
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check APs ending at x-1 (0-based)
            # AP: a, a+d, a+2d, a+3d = x-1
            # So a+3d = x-1 => a = x-1-3d
            # Need a >= 0 => 3d <= x-1 => d <= (x-1)//3
            for d in range(1, (x-1)//3 + 1):
                a = x - 1 - 3 * d
                if a < 0: continue
                # AP indices: a, a+d, a+2d, a+3d
                # Colors: colors[a], colors[a+d], colors[a+2d], color (new)
                c1 = colors[a]
                c2 = colors[a + d]
                c3 = colors[a + 2 * d]
                # Need len({c1, c2, c3, color}) >= 3
                if len({c1, c2, c3, color}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # Should not happen if we have 128 colors
            colors.append(127)
            
    return colors

# Run construct
# construct()
