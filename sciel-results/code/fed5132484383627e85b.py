import time
import random
import sys

def construct():
    n = 128
    deadline = time.time() + 85.0
    
    # Precompute all 4-term APs (0-indexed)
    # AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    # For each position, list indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v

    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3

    def local_search(k, time_limit):
        # Initialize random
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        # Simulated Annealing
        temp = 1.0
        cooling = 0.9995
        min_temp = 0.01
        
        while time.time() < time_limit:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Calculate delta for changing pos to new_color
            # Only affected APs are in pos_to_aps[pos]
            affected_aps = pos_to_aps[pos]
            
            # Count current invalid APs in affected set
            # We can just evaluate the change directly
            
            best_new_color = old_color
            best_delta = 0
            
            # Try all k colors
            # Optimization: Only try colors that are different
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                # Temporarily set
                colors[pos] = new_color
                
                # Recount violations for affected APs
                new_invalid_count = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid_count += 1
                
                # We need old_invalid_count for these APs
                # To avoid recalculating old count every time, we can precompute it?
                # Or just accept the cost of recalculating.
                # Since k is small (3-10) and degree is small, this is fast.
                
                # Wait, we need the OLD invalid count for these specific APs to get delta.
                # Let's compute old_invalid_count once outside the new_color loop.
                # But it depends on old_color.
                # So compute old_invalid_count once.
                
                # Let's restructure:
                # 1. Compute old_invalid_count for affected_aps with old_color.
                # 2. For each new_color, compute new_invalid_count.
                # 3. delta = new - old.
                
                # This is inefficient if done inside the loop.
                pass
            
            # Restructured loop for efficiency
            
            # Compute old invalid count for affected APs
            colors[pos] = old_color
            old_invalid_count = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    old_invalid_count += 1
            
            best_new_color = old_color
            best_delta = 0
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_invalid_count = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid_count += 1
                
                delta = new_invalid_count - old_invalid_count
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                elif delta == best_delta and random.random() < 0.5:
                    best_new_color = new_color
            
            colors[pos] = old_color # Revert
            
            # Acceptance
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = True
            else:
                if temp > 0:
                    prob = pow(2.718281828, -best_delta / temp)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                colors[pos] = best_new_color
                current_violations += best_delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    return colors
            
            temp *= cooling
            if temp < min_temp:
                break
        
        return best_colors if best_violations == 0 else None

    # Try increasing k
    # Prioritize small k
    k_values = [3, 4, 5, 6, 8, 10, 16, 32, 64, 128]
    
    for k in k_values:
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time based on k
        # Smaller k is harder, but more rewarding.
        # Give 15s to k=3, 15s to k=4, 10s to k=5, 5s to others.
        if k == 3:
            time_slice = 15.0
        elif k == 4:
            time_slice = 15.0
        elif k == 5:
            time_slice = 10.0
        else:
            time_slice = 5.0
            
        if time.time() + time_slice > deadline - 2.0:
            time_slice = max(1.0, deadline - 2.0 - time.time())
        
        result = local_search(k, time.time() + time_slice)
        if result is not None:
            return result
    
    # Fallback
    return list(range(n))
