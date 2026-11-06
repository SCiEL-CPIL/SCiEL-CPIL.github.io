import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 55  # Safe margin
    N = 256

    # Precomputation
    three_aps = []
    four_aps = []

    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))

    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    elem_to_3aps = {x: [] for x in range(1, N + 1)}
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)

    elem_to_4aps = {x: [] for x in range(1, N + 1)}
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s:
                count += 1
        return count

    def delta_add(s, x):
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s: delta += 1
            elif x == b:
                if a in s and c in s: delta += 1
            else:
                if a in s and b in s: delta += 1
        return delta

    def delta_remove(s, x):
        return delta_add(s, x)

    def is_valid_add(s, x):
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if b in s and c in s and d in s: return False
            elif x == b:
                if a in s and c in s and d in s: return False
            elif x == c:
                if a in s and b in s and d in s: return False
            else:
                if a in s and b in s and c in s: return False
        return True

    def is_valid_swap(s, x_out, x_in):
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            others = []
            if x_in == a: others = [b, c, d]
            elif x_in == b: others = [a, c, d]
            elif x_in == c: others = [a, b, d]
            else: others = [a, b, c]
            
            valid = True
            for y in others:
                if y not in s:
                    valid = False
                    break
                if y == x_out:
                    valid = False
                    break
            if valid:
                return False
        return True

    # Greedy Initialization
    s = set()
    for x in range(1, N + 1):
        if is_valid_add(s, x):
            s.add(x)
            
    best_s = set(s)
    best_score = count_3aps(best_s)
    current_score = best_score

    # Simulated Annealing with better parameters
    random.seed(42)
    T = 5.0
    cooling = 0.998  # Slightly slower cooling
    min_T = 0.005
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        accepted = False
        delta = 0
        
        if move_type < 0.4: # Add
            candidates = [x for x in range(1, N + 1) if x not in s]
            if candidates:
                x = random.choice(candidates)
                if is_valid_add(s, x):
                    delta = delta_add(s, x)
                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)) or (delta == 0 and random.random() < 0.5):
                        s.add(x)
                        current_score += delta
                        accepted = True
                        if current_score > best_score:
                            best_score = current_score
                            best_s = set(s)
                            
        elif move_type < 0.6: # Remove
            if s:
                x = random.choice(list(s))
                delta = -delta_remove(s, x)
                if random.random() < math.exp(delta / T):
                    s.remove(x)
                    current_score += delta
                    accepted = True
                    
        else: # Swap
            if s:
                x_out = random.choice(list(s))
                candidates = [x for x in range(1, N + 1) if x not in s]
                if candidates:
                    x_in = random.choice(candidates)
                    if is_valid_swap(s, x_out, x_in):
                        loss = delta_remove(s, x_out)
                        gain = 0
                        for ap_idx in elem_to_3aps[x_in]:
                            a, b, c = three_aps[ap_idx]
                            others = []
                            if x_in == a: others = [b, c]
                            elif x_in == b: others = [a, c]
                            else: others = [a, b]
                            
                            if all(y in s for y in others) and all(y != x_out for y in others):
                                gain += 1
                                
                        delta = gain - loss
                        if delta > 0 or random.random() < math.exp(delta / T):
                            s.remove(x_out)
                            s.add(x_in)
                            current_score += delta
                            accepted = True
                            if current_score > best_score:
                                best_score = current_score
                                best_s = set(s)
                                
        T *= cooling
            
    return sorted(best_s)
