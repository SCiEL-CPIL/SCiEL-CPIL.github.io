"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples for fast violation checking
    # A 4-AP is (a, a+d, a+2d, a+3d) with a>=1, a+3d<=N
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples for counting
    # A 3-AP is (a, a+d, a+2d) with a>=1, a+2d<=N
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    num_3aps = len(three_ap_tuples)
    num_4aps = len(four_ap_tuples)
    
    # Map element to list of 3-AP indices containing it
    ap3_by_elem = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        ap3_by_elem[a].append(idx)
        ap3_by_elem[b].append(idx)
        ap3_by_elem[c].append(idx)
        
    # Map element to list of 4-AP indices containing it
    ap4_by_elem = [[] for _ in range(N + 1)]
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        ap4_by_elem[a].append(idx)
        ap4_by_elem[b].append(idx)
        ap4_by_elem[c].append(idx)
        ap4_by_elem[d].append(idx)
        
    # State variables
    selected = [False] * (N + 1)
    ap3_counts = [0] * num_3aps
    ap4_counts = [0] * num_4aps
    total_3aps = 0
    max_ap4_count = 0 # If any ap4_count reaches 4, we have a violation
    
    def add_element(x):
        nonlocal total_3aps, max_ap4_count
        selected[x] = True
        for idx in ap3_by_elem[x]:
            ap3_counts[idx] += 1
            if ap3_counts[idx] == 3:
                total_3aps += 1
        for idx in ap4_by_elem[x]:
            ap4_counts[idx] += 1
            if ap4_counts[idx] > max_ap4_count:
                max_ap4_count = ap4_counts[idx]
                
    def remove_element(x):
        nonlocal total_3aps, max_ap4_count
        selected[x] = False
        for idx in ap3_by_elem[x]:
            if ap3_counts[idx] == 3:
                total_3aps -= 1
            ap3_counts[idx] -= 1
        # Recalculate max_ap4_count if necessary? 
        # For simplicity, we can just check if any ap4_count is 4 after removal?
        # Actually, if we remove, max_ap4_count can only decrease.
        # But we need to know if it drops below 4.
        # Let's just recalculate max if it was 4.
        if max_ap4_count == 4:
             max_ap4_count = max(ap4_counts)
        for idx in ap4_by_elem[x]:
            ap4_counts[idx] -= 1
            
    def is_valid():
        return max_ap4_count < 4
        
    # Greedy construction
    elements = list(range(1, N + 1))
    random.shuffle(elements)
    
    for x in elements:
        if time.time() > deadline - 10.0:
            break
        # Check if adding x is valid
        # We can simulate the addition to check validity and score
        # Since adding x only increases counts, we just need to check if any ap4_count becomes 4
        # And count how many ap3_counts become 3
        
        # Check 4-AP violation
        valid_add = True
        for idx in ap4_by_elem[x]:
            if ap4_counts[idx] == 3:
                valid_add = False
                break
        if not valid_add:
            continue
            
        # Count potential 3-AP increase
        gain = 0
        for idx in ap3_by_elem[x]:
            if ap3_counts[idx] == 2:
                gain += 1
                
        # We just add if it's valid? The original greedy tried to maximize score.
        # With incremental state, evaluating all candidates is faster, but still O(N * deg).
        # Let's stick to a simple greedy: add if valid. Or better, pick the one with highest gain among valid ones.
        
        # To keep it simple and fast, let's just add if valid.
        # But the original code tried to maximize. Let's do a limited greedy:
        # Find all valid x, pick one with max gain.
        
        # Actually, iterating all remaining elements every step is O(N^2 * deg).
        # N=256, so 256^2 * 192 ~ 12.5M ops. This is fast enough.
        
        best_x = -1
        best_gain = -1
        
        for y in elements:
             if selected[y]: continue
             
             valid_add = True
             for idx in ap4_by_elem[y]:
                 if ap4_counts[idx] == 3:
                     valid_add = False
                     break
             if not valid_add:
                 continue
                 
             gain = 0
             for idx in ap3_by_elem[y]:
                 if ap3_counts[idx] == 2:
                     gain += 1
                     
             if gain > best_gain:
                 best_gain = gain
                 best_x = y
                 
        if best_x == -1:
             break
             
        add_element(best_x)
        elements.remove(best_x)
        
    current_score = total_3aps
    best_score = current_score
    best_selected = selected[:]
    
    # Simulated Annealing
    if time.time() < deadline - 2.0:
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.999
        
        while time.time() < deadline - 2.0 and T > T_min:
            if random.random() < 0.5:
                # Remove
                if any(selected[1:]):
                    # Pick random selected element
                    sel_list = [i for i in range(1, N+1) if selected[i]]
                    if not sel_list: break
                    x = random.choice(sel_list)
                    
                    # Calculate delta
                    # Removing x decreases total_3aps if any ap3 containing x had count 3
                    delta = 0
                    for idx in ap3_by_elem[x]:
                        if ap3_counts[idx] == 3:
                            delta -= 1
                            
                    # Check if removing x is valid (it always is, as it reduces counts)
                    # But we need to update state
                    remove_element(x)
                    new_score = total_3aps
                    
                    if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = selected[:]
                    else:
                        # Revert
                        add_element(x)
                        current_score = total_3aps
            else:
                # Add
                unselected = [i for i in range(1, N+1) if not selected[i]]
                if unselected:
                    x = random.choice(unselected)
                    
                    # Check validity
                    valid_add = True
                    for idx in ap4_by_elem[x]:
                        if ap4_counts[idx] == 3:
                            valid_add = False
                            break
                            
                    if valid_add:
                        # Calculate delta
                        delta = 0
                        for idx in ap3_by_elem[x]:
                            if ap3_counts[idx] == 2:
                                delta += 1
                            
                        add_element(x)
                        new_score = total_3aps
                        
                        if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_selected = selected[:]
                        else:
                            # Revert
                            remove_element(x)
                            current_score = total_3aps
                            
            T *= cooling_rate
            
    result = [i for i in range(1, N+1) if best_selected[i]]
    return result
# EVOLVE-BLOCK-END
