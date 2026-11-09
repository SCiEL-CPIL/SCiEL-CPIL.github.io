"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0

    # Precompute APs
    # We need to map each variable to the APs it participates in.
    # AP is defined by (start, step, length).
    # To save memory and time, we only store the list of APs for each variable.
    # We also need the total length of each AP to track 'remaining'.
    
    ap_data = [] # List of lists of indices? No, just store the APs themselves for initial setup.
    var_aps = [[] for _ in range(n)] # var_aps[i] = list of ap_ids containing i
    
    # We will generate all APs
    ap_id = 0
    ap_lengths = []
    
    for d in range(1, n):
        max_len = (n - 1) // d + 1
        for L in range(1, max_len + 1):
            max_start = n - (L - 1) * d
            for a in range(max_start):
                indices = [a + k * d for k in range(L)]
                # Store AP length
                ap_lengths.append(L)
                # Map variables to this AP
                for idx in indices:
                    var_aps[idx].append(ap_id)
                ap_id += 1
                
    num_aps = ap_id

    # Sort variables by number of constraints (APs) descending?
    # Or keep natural order?
    # Let's try natural order first, it's simpler and often works with good pruning.
    # Actually, "Most Constrained" usually helps.
    # Let's create an ordering based on len(var_aps[i])
    order = list(range(n))
    # order.sort(key=lambda x: len(var_aps[x]), reverse=True)
    # But reversing the order might complicate the "remaining" logic if we don't map back.
    # Let's stick to natural order for simplicity and predictability, 
    # but optimize the inner loop heavily.
    
    best_s = None
    best_B = float('inf')
    
    # Try increasing B
    for B in range(1, 11):
        # Reset state
        s = [0] * n
        sums = [0] * num_aps
        remaining = ap_lengths[:] # Copy
        
        t_start_B = time.time()
        found = False
        
        def bt(k):
            nonlocal s, sums, remaining, found
            if found:
                return True
            if time.time() - t_start_B > 5.0:
                return False
            if k == n:
                found = True
                return True
                
            # Try 1 then -1
            # Heuristic: try the sign that reduces the magnitude of sums more?
            # For now, simple backtracking.
            
            for sigma in [1, -1]:
                # Check consistency for all APs containing k
                consistent = True
                ap_ids = var_aps[k]
                
                # Apply changes
                for ap_id in ap_ids:
                    new_sum = sums[ap_id] + sigma
                    rem = remaining[ap_id] - 1
                    # Pruning condition:
                    # The final sum will be in [new_sum - rem, new_sum + rem]
                    # We need this interval to intersect [-B, B]
                    # i.e., new_sum - rem <= B and new_sum + rem >= -B
                    if new_sum - rem > B or new_sum + rem < -B:
                        consistent = False
                        break
                    # Store temp changes? 
                    # To avoid overhead of copying, we update in place and revert.
                    # But if we break early, we only updated some.
                    # Better: check first, then update?
                    # Or update, check, revert if fail.
                    # Let's update, check, revert.
                    
                    sums[ap_id] = new_sum
                    remaining[ap_id] = rem

                if consistent:
                    s[k] = sigma
                    if bt(k + 1):
                        return True
                    # Revert
                    s[k] = 0
                    for ap_id in ap_ids:
                        sums[ap_id] -= sigma
                        remaining[ap_id] += 1
                else:
                    # Revert partial updates
                    for ap_id in ap_ids:
                        # Only revert if we updated it?
                        # We broke out of the loop, so some might be updated.
                        # We need to know which ones were updated.
                        # This is messy. 
                        # Better approach: 
                        # 1. Check all without updating.
                        # 2. If all consistent, update all.
                        # 3. Recurse.
                        # 4. Revert all.
                        pass

            return False

        # Refine bt to avoid partial update mess
        def bt_safe(k):
            nonlocal s, sums, remaining, found
            if found:
                return True
            if time.time() - t_start_B > 5.0:
                return False
            if k == n:
                found = True
                return True
                
            ap_ids = var_aps[k]
            
            for sigma in [1, -1]:
                # Check phase
                ok = True
                for ap_id in ap_ids:
                    new_sum = sums[ap_id] + sigma
                    rem = remaining[ap_id] - 1
                    if new_sum - rem > B or new_sum + rem < -B:
                        ok = False
                        break
                
                if ok:
                    # Update phase
                    for ap_id in ap_ids:
                        sums[ap_id] += sigma
                        remaining[ap_id] -= 1
                    s[k] = sigma
                    
                    if bt_safe(k + 1):
                        return True
                    
                    # Revert phase
                    for ap_id in ap_ids:
                        sums[ap_id] -= sigma
                        remaining[ap_id] += 1
                    s[k] = 0
            
            return False

        if bt_safe(0):
            best_s = s[:]
            best_B = B
            break
        else:
            # If time limit exceeded for this B, we might want to try a larger B?
            # Or just accept failure for this B and try next?
            # The time limit is per B.
            # If we don't find a solution for B=1 in 5s, we try B=2.
            pass

    if best_s is None:
        # Fallback: Random or Greedy
        # A simple greedy might give a decent value
        rng = random.Random(42)
        best_s = [rng.choice([-1, 1]) for _ in range(n)]
        
    return best_s

# EVOLVE-BLOCK-END
