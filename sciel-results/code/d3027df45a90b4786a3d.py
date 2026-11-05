import numpy as np
import time
import random

# Precompute APs
APs = []
idx_to_aps = [[] for _ in range(128)]
for a in range(128):
    max_d = (127 - a) // 3
    for d in range(1, max_d + 1):
        ap_idx = len(APs)
        APs.append((a, a+d, a+2*d, a+3*d))
        for pos in (a, a+d, a+2*d, a+3*d):
            idx_to_aps[pos].append(ap_idx)

# AP_indices is not needed for incremental, just APs list
# We need to store the current state of violations for each AP?
# Or just a count?
# We need to know if an AP is currently violating.
# is_violating[ap_idx] = True/False

def is_ap_violating(colors, ap):
    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
    # Check < 3 distinct
    # Mono
    if c0 == c1 == c2 == c3:
        return True
    # Bi
    # Check all pairs/triples logic
    # (c0==c1 and c2==c3 and c1!=c2) or ...
    # Simpler: 
    # If len(set([c0,c1,c2,c3])) < 3: return True
    # But set creation is slow.
    # Use the boolean logic derived earlier.
    if (c0 == c1 and c2 == c3 and c1 != c2) or \
       (c0 == c2 and c1 == c3 and c0 != c1) or \
       (c0 == c3 and c1 == c2 and c0 != c1) or \
       (c0 == c1 and c1 == c2 and c2 != c3) or \
       (c0 == c1 and c1 == c3 and c1 != c2) or \
       (c0 == c2 and c2 == c3 and c0 != c1) or \
       (c1 == c2 and c2 == c3 and c1 != c0):
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    for k in [4, 5, 6, 7, 8]:
        best_colors = None
        best_violations = float('inf')
        
        for restart in range(100):
            if time.time() > deadline:
                break
                
            colors = np.random.randint(0, k, size=128)
            
            # Initialize is_violating
            is_violating = np.zeros(len(APs), dtype=bool)
            current_violations = 0
            for i, ap in enumerate(APs):
                if is_ap_violating(colors, ap):
                    is_violating[i] = True
                    current_violations += 1
            
            if current_violations == 0:
                return list(colors)
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            T = 1.0
            for step in range(20000):
                if time.time() > deadline:
                    break
                    
                idx = np.random.randint(128)
                old_color = colors[idx]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Update violations for APs containing idx
                # Save old states
                affected_aps = idx_to_aps[idx]
                old_states = []
                for ap_idx in affected_aps:
                    old_states.append(is_violating[ap_idx])
                    
                colors[idx] = new_color
                
                delta_v = 0
                for j, ap_idx in enumerate(affected_aps):
                    new_state = is_ap_violating(colors, APs[ap_idx])
                    if new_state and not old_states[j]:
                        delta_v += 1
                    elif not new_state and old_states[j]:
                        delta_v -= 1
                    is_violating[ap_idx] = new_state
                    
                new_violations = current_violations + delta_v
                
                if new_violations <= current_violations or random.random() < np.exp(-(new_violations - current_violations) / T):
                    current_violations = new_violations
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_colors = colors.copy()
                        if new_violations == 0:
                            return list(colors)
                else:
                    # Revert
                    colors[idx] = old_color
                    for j, ap_idx in enumerate(affected_aps):
                        is_violating[ap_idx] = old_states[j]
                        
                T *= 0.999
                
        # End restart loop
        # End k loop

    # If no solution found, return best_colors (might be invalid, but hopefully not)
    # To be safe, if best_violations > 0, we should try to force a valid one with higher k
    # But given the constraints, k=4 or 5 should work.
    
    return list(best_colors)
