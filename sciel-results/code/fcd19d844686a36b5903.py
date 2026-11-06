import random
import time
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit
    
    # Precompute all 4-APs (as tuples of 4 elements)
    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute all 3-APs (as tuples of 3 elements)
    three_aps = []
    for d in range(1, (n - 1) // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    # Build index: for each element, which 4-APs and 3-APs contain it
    elem_to_4aps = {x: [] for x in range(1, n + 1)}
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    elem_to_3aps = {x: [] for x in range(1, n + 1)}
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
    
    # Initial Greedy Construction
    current_set = set()
    for x in range(1, n + 1):
        # Check if adding x creates a 4-AP with existing elements
        is_valid = True
        for ap_idx in elem_to_4aps[x]:
            ap = four_aps[ap_idx]
            # Check if the other 3 elements are in current_set
            if all(elem in current_set for elem in ap if elem != x):
                is_valid = False
                break
        if is_valid:
            current_set.add(x)
    
    if not current_set:
        return [1]
    
    # Helper to count 3-APs in a set using precomputed index
    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if all(x in s for x in ap):
                count += 1
        return count
    
    current_score = count_3aps(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    # Simulated Annealing
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.1
    max_steps = 500000
    
    step_count = 0
    
    while time.time() - start_time < time_limit and step_count < max_steps:
        step_count += 1
        if temp < min_temp:
            temp = min_temp
        
        move_type = random.random()
        accepted = False
        delta_score = 0
        
        if len(current_set) == 0:
            break
            
        # --- MOVE: ADD ---
        if move_type < 0.35 and len(current_set) < n:
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if candidates:
                x = random.choice(candidates)
                # Check 4-AP validity for adding x
                valid = True
                for ap_idx in elem_to_4aps[x]:
                    ap = four_aps[ap_idx]
                    if all(elem in current_set for elem in ap if elem != x):
                        valid = False
                        break
                
                if valid:
                    # Calculate Delta 3-APs: new 3-APs involving x
                    delta = 0
                    for ap_idx in elem_to_3aps[x]:
                        ap = three_aps[ap_idx]
                        # Count if the other 2 elements are in current_set
                        if all(elem in current_set for elem in ap if elem != x):
                            delta += 1
                    
                    if delta >= 0 or random.random() < np.exp(delta / temp):
                        current_set.add(x)
                        current_score += delta
                        accepted = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
        
        # --- MOVE: REMOVE ---
        elif move_type < 0.65:
            if current_set:
                x = random.choice(list(current_set))
                # Calculate Delta 3-APs (Loss)
                delta = 0
                for ap_idx in elem_to_3aps[x]:
                    ap = three_aps[ap_idx]
                    if all(elem in current_set for elem in ap if elem != x):
                        delta -= 1

                if delta >= 0 or random.random() < np.exp(delta / temp):
                    current_set.remove(x)
                    current_score += delta
                    accepted = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
        
        # --- MOVE: SWAP ---
        else:
            if current_set and len(current_set) < n:
                x = random.choice(list(current_set))
                candidates = [y for y in range(1, n + 1) if y not in current_set]
                if candidates:
                    y = random.choice(candidates)
                    
                    # Check if S - {x} + {y} is valid (no 4-AP)
                    valid = True
                    for ap_idx in elem_to_4aps[y]:
                        ap = four_aps[ap_idx]
                        # Check if the other 3 elements are in S\{x}
                        other_elems = [elem for elem in ap if elem != y]
                        if all((elem in current_set and elem != x) for elem in other_elems):
                            valid = False
                            break
                    
                    if valid:
                        # Calculate Delta 3-APs
                        # Loss from x (in S)
                        loss_x = 0
                        for ap_idx in elem_to_3aps[x]:
                            ap = three_aps[ap_idx]
                            if all(elem in current_set for elem in ap if elem != x):
                                loss_x += 1
                                
                        # Gain from y (in S\{x})
                        gain_y = 0
                        for ap_idx in elem_to_3aps[y]:
                            ap = three_aps[ap_idx]
                            other_elems = [elem for elem in ap if elem != y]
                            if all((elem in current_set and elem != x) for elem in other_elems):
                                gain_y += 1
                                
                        delta = gain_y - loss_x
                        
                        if delta >= 0 or random.random() < np.exp(delta / temp):
                            current_set.remove(x)
                            current_set.add(y)
                            current_score += delta
                            accepted = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(current_set)
        
        if not accepted:
            temp *= cooling
        else:
            temp *= cooling
    
    return sorted(best_set)
