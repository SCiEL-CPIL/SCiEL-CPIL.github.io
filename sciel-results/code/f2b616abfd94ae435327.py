import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    
    # 1. Precompute 4-APs and 3-APs
    four_aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    three_aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 2
        for d in range(1, max_d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    num_4 = len(four_aps)
    num_3 = len(three_aps)
    
    # Index elements to APs
    four_ap_elem = [[] for _ in range(n + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            four_ap_elem[x].append(i)
            
    three_ap_elem = [[] for _ in range(n + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            three_ap_elem[x].append(i)

    # 2. Initial Set (Greedy Seed)
    selected = set()
    for x in range(1, n + 1):
        can_add = True
        for idx in four_ap_elem[x]:
            a, b, c, e = four_aps[idx]
            others = [y for y in (a, b, c, e) if y != x]
            if all(y in selected for y in others):
                can_add = False
                break
        if can_add:
            selected.add(x)
            
    sel_list = list(selected)

    # 3. State Initialization
    four_counts = [0] * num_4
    for x in selected:
        for idx in four_ap_elem[x]:
            four_counts[idx] += 1
            
    three_counts = [0] * num_3
    for x in selected:
        for idx in three_ap_elem[x]:
            three_counts[idx] += 1
            
    score = sum(1 for c in three_counts if c == 3)
    best_score = score
    best_set = set(selected)
    best_list = list(selected)

    # 4. Simulated Annealing
    T = 10.0
    cooling = 0.9995
    min_T = 0.1
    
    # Helper: Check if x can be added
    def can_add(x):
        for idx in four_ap_elem[x]:
            if four_counts[idx] == 3:
                return False
        return True

    # Helper: Score delta for adding x
    def delta_add(x):
        d = 0
        for idx in three_ap_elem[x]:
            if three_counts[idx] == 2:
                d += 1
        return d

    # Helper: Score delta for removing x
    def delta_remove(x):
        d = 0
        for idx in three_ap_elem[x]:
            if three_counts[idx] == 3:
                d -= 1
        return d

    # Main Loop
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.4 and len(selected) < n:
            # Move: Add
            x = random.randint(1, n)
            if x in selected:
                continue
            if not can_add(x):
                continue
                
            d_score = delta_add(x)
            
            # Apply
            selected.add(x)
            sel_list.append(x)
            for idx in four_ap_elem[x]:
                four_counts[idx] += 1
            for idx in three_ap_elem[x]:
                three_counts[idx] += 1
            score += d_score
            
            if score > best_score:
                best_score = score
                best_set = set(selected)
                best_list = list(sel_list)
                
            # Metropolis Criterion
            if d_score < 0:
                if random.random() > math.exp(d_score / T):
                    # Revert
                    selected.remove(x)
                    sel_list.remove(x) # O(N)
                    for idx in four_ap_elem[x]:
                        four_counts[idx] -= 1
                    for idx in three_ap_elem[x]:
                        three_counts[idx] -= 1
                    score -= d_score
        elif r < 0.7 and len(selected) > 0:
            # Move: Remove
            idx = random.randint(0, len(sel_list) - 1)
            x = sel_list[idx]
            
            d_score = delta_remove(x)
            
            # Apply
            selected.remove(x)
            # Swap and pop for O(1) removal
            sel_list[idx] = sel_list[-1]
            sel_list.pop()
            
            for idx_ap in four_ap_elem[x]:
                four_counts[idx_ap] -= 1
            for idx_ap in three_ap_elem[x]:
                three_counts[idx_ap] -= 1
            score += d_score
            
            if score > best_score:
                best_score = score
                best_set = set(selected)
                best_list = list(sel_list)
                
            # Metropolis Criterion
            if d_score < 0:
                if random.random() > math.exp(d_score / T):
                    # Revert
                    selected.add(x)
                    sel_list.append(x)
                    for idx_ap in four_ap_elem[x]:
                        four_counts[idx_ap] += 1
                    for idx_ap in three_ap_elem[x]:
                        three_counts[idx_ap] += 1
                    score -= d_score
        else:
            # Move: Swap
            if len(selected) == 0 or len(selected) == n:
                continue
                
            idx = random.randint(0, len(sel_list) - 1)
            x = sel_list[idx]
            y = random.randint(1, n)
            if y in selected:
                continue
                
            # 1. Remove x
            # We need to check if y can be added AFTER removing x.
            # To do this efficiently, we temporarily remove x, check y, then decide.
            
            # Temp Remove x
            selected.remove(x)
            sel_list[idx] = sel_list[-1]
            sel_list.pop()
            
            for idx_ap in four_ap_elem[x]:
                four_counts[idx_ap] -= 1
            for idx_ap in three_ap_elem[x]:
                three_counts[idx_ap] -= 1
            score += delta_remove(x) # This is the score change for removal
            
            # Check if y can be added
            if not can_add(y):
                # Revert Removal of x
                selected.add(x)
                sel_list.append(x)
                for idx_ap in four_ap_elem[x]:
                    four_counts[idx_ap] += 1
                for idx_ap in three_ap_elem[x]:
                    three_counts[idx_ap] += 1
                score -= delta_remove(x)
                continue
            
            # 2. Add y
            d_add = delta_add(y)
            
            selected.add(y)
            sel_list.append(y)
            
            for idx_ap in four_ap_elem[y]:
                four_counts[idx_ap] += 1
            for idx_ap in three_ap_elem[y]:
                three_counts[idx_ap] += 1
            score += d_add
            
            total_delta = delta_remove(x) + d_add
            
            if score > best_score:
                best_score = score
                best_set = set(selected)
                best_list = list(sel_list)
                
            # Metropolis Criterion for Swap
            if total_delta < 0:
                if random.random() > math.exp(total_delta / T):
                    # Revert Swap (Remove y, Add x)
                    selected.remove(y)
                    sel_list.remove(y)
                    for idx_ap in four_ap_elem[y]:
                        four_counts[idx_ap] -= 1
                    for idx_ap in three_ap_elem[y]:
                        three_counts[idx_ap] -= 1
                    score -= d_add
                    
                    selected.add(x)
                    sel_list.append(x)
                    for idx_ap in four_ap_elem[x]:
                        four_counts[idx_ap] += 1
                    for idx_ap in three_ap_elem[x]:
                        three_counts[idx_ap] += 1
                    score -= delta_remove(x)

        # Cool down
        if random.random() < 0.005:
            T *= cooling
            if T < min_T:
                T = min_T

    return sorted(best_set)
