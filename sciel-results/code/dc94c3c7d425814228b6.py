import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    # Precompute APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Precompute APs containing each position
    pos_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
            
    def is_bad_ap(ap, colors):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) <= 2

    def count_bad_aps(colors):
        count = 0
        for ap in aps:
            if is_bad_ap(ap, colors):
                count += 1
        return count

    best_colors = None
    best_bad_count = float('inf')
    
    # Try k from 4 to 12
    for k in range(4, 13):
        if time.time() > deadline - 10:
            break
            
        # Random restarts for this k
        restarts = 0
        max_restarts = 20
        
        while restarts < max_restarts and time.time() < deadline - 5:
            colors = [random.randint(0, k - 1) for _ in range(N)]
            current_bad = count_bad_aps(colors)
            
            if current_bad == 0:
                return colors
                
            temp = 1.0
            cooling = 0.9995
            no_improve = 0
            max_no_improve = 5000
            
            while no_improve < max_no_improve and time.time() < deadline - 5:
                if current_bad == 0:
                    return colors
                    
                # Pick a random bad AP
                # To pick a random bad AP efficiently, we can just pick a random AP and check if it's bad?
                # Or maintain a list? Maintaining a list is expensive to update.
                # Given N_aps is small, picking a random AP and checking is O(1) expected if bad density is high.
                # If bad density is low, we might loop.
                # Let's just pick a random position, then a random AP containing it, and check if it's bad.
                # If not bad, try another.
                
                # Better: Just pick a random AP index from 0 to num_aps-1.
                # If it's bad, proceed. If not, retry (up to a few times).
                
                found_bad_ap = False
                for _ in range(10):
                    ap_idx = random.randint(0, num_aps - 1)
                    if is_bad_ap(aps[ap_idx], colors):
                        found_bad_ap = True
                        break
                
                if not found_bad_ap:
                    # If we can't find a bad AP, it's likely 0 bads
                    # But we already checked count_bad.
                    # So this shouldn't happen unless count_bad is wrong.
                    break
                
                ap = aps[ap_idx]
                pos_idx = random.choice(ap)
                
                # Try all k colors for pos_idx
                best_new_color = colors[pos_idx]
                best_new_bad = current_bad
                
                affected_aps = pos_to_aps[pos_idx]
                
                # Calculate current bad status for affected APs
                # We need to know which of these are currently bad.
                current_bad_in_affected = 0
                for a_idx in affected_aps:
                    if is_bad_ap(aps[a_idx], colors):
                        current_bad_in_affected += 1
                        
                for new_color in range(k):
                    if new_color == colors[pos_idx]:
                        continue
                    
                    old_color = colors[pos_idx]
                    colors[pos_idx] = new_color
                    
                    new_bad_in_affected = 0
                    for a_idx in affected_aps:
                        if is_bad_ap(aps[a_idx], colors):
                            new_bad_in_affected += 1
                            
                    # Total bad = current_bad - current_bad_in_affected + new_bad_in_affected
                    total_new_bad = current_bad - current_bad_in_affected + new_bad_in_affected
                    
                    if total_new_bad < best_new_bad:
                        best_new_bad = total_new_bad
                        best_new_color = new_color
                        
                    colors[pos_idx] = old_color
                
                if best_new_bad < current_bad:
                    colors[pos_idx] = best_new_color
                    current_bad = best_new_bad
                    no_improve = 0
                    if current_bad == 0:
                        return colors
                else:
                    # Random move
                    if random.random() < temp:
                        new_color = random.randint(0, k - 1)
                        old_color = colors[pos_idx]
                        colors[pos_idx] = new_color
                        
                        new_bad_in_affected = 0
                        for a_idx in affected_aps:
                            if is_bad_ap(aps[a_idx], colors):
                                new_bad_in_affected += 1
                            
                        total_new_bad = current_bad - current_bad_in_affected + new_bad_in_affected
                        
                        if total_new_bad <= current_bad or random.random() < (2.71828 ** (-(total_new_bad - current_bad) / temp)):
                            current_bad = total_new_bad
                        else:
                            colors[pos_idx] = old_color
                    no_improve += 1
                    temp *= cooling
                    
            if current_bad < best_bad_count:
                best_bad_count = current_bad
                best_colors = colors.copy()
                
            restarts += 1
            
    if best_colors is not None:
        return best_colors
        
    return [random.randint(0, 3) for _ in range(N)]
