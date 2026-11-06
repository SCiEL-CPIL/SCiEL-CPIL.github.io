import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 55
    N = 256
    
    # Precompute
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a - 1, a + d - 1, a + 2 * d - 1))
    
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    n3 = len(three_aps)
    n4 = len(four_aps)
    
    three_aps_arr = np.array(three_aps)
    four_aps_arr = np.array(four_aps)
    
    elem_to_3aps = [[] for _ in range(N)]
    for i in range(n3):
        for x in three_aps_arr[i]:
            elem_to_3aps[x].append(i)
    
    elem_to_4aps = [[] for _ in range(N)]
    for i in range(n4):
        for x in four_aps_arr[i]:
            elem_to_4aps[x].append(i)
    
    def count_3aps(s):
        return int(np.sum(s[three_aps_arr[:, 0]] & s[three_aps_arr[:, 1]] & s[three_aps_arr[:, 2]]))
    
    def is_valid_add(s, x):
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps_arr[ap_idx]
            if x == a and s[b] and s[c] and s[d]: return False
            if x == b and s[a] and s[c] and s[d]: return False
            if x == c and s[a] and s[b] and s[d]: return False
            if x == d and s[a] and s[b] and s[c]: return False
        return True
    
    def delta_add(s, x):
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps_arr[ap_idx]
            if x == a and s[b] and s[c]: delta += 1
            elif x == b and s[a] and s[c]: delta += 1
            elif x == c and s[a] and s[b]: delta += 1
        return delta
    
    def delta_remove(s, x):
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps_arr[ap_idx]
            if x == a and s[b] and s[c]: delta += 1
            elif x == b and s[a] and s[c]: delta += 1
            elif x == c and s[a] and s[b]: delta += 1
        return delta
    
    def is_valid_swap(s, x_out, x_in):
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps_arr[ap_idx]
            if x_in == a: others = [b, c, d]
            elif x_in == b: others = [a, c, d]
            elif x_in == c: others = [a, b, d]
            else: others = [a, b, c]
            for y in others:
                if y == x_out or not s[y]:
                    break
            else:
                return False
        return True
    
    random.seed(42)
    best_s = None
    best_score = -1
    
    # Multiple restarts
    for restart in range(30):
        s = np.zeros(N, dtype=bool)
        
        # Different initializations
        if restart % 3 == 0:
            # Start dense, remove to break 4-APs
            s[:] = True
            for _ in range(300):
                # Find a 4-AP
                found = False
                for i in range(n4):
                    a, b, c, d = four_aps_arr[i]
                    if s[a] and s[b] and s[c] and s[d]:
                        # Remove one of them
                        x = random.choice([a, b, c, d])
                        s[x] = False
                        found = True
                        break
                if not found:
                    break
        elif restart % 3 == 1:
            # Start with random set
            k = random.randint(100, 180)
            s[random.sample(range(N), k)] = True
        else:
            # Start with structured set: numbers with even binary digit sum
            for x in range(N):
                if bin(x).count('1') % 2 == 0:
                    s[x] = True
        
        current_score = count_3aps(s)
        
        # Local search
        T = 4.0
        cooling = 0.9995
        min_T = 0.01
        no_improve = 0
        
        while time.time() < deadline - 3 and T >= min_T:
            move_type = random.random()
            
            if move_type < 0.45:  # Add
                candidates = [x for x in range(N) if not s[x]]
                if candidates:
                    x = random.choice(candidates)
                    if is_valid_add(s, x):
                        delta = delta_add(s, x)
                        if delta > 0 or random.random() < math.exp(delta / T):
                            s[x] = True
                            current_score += delta
                            no_improve = 0
                        else:
                            no_improve += 1
            elif move_type < 0.65:  # Remove
                if np.any(s):
                    x = random.choice(np.where(s)[0].tolist())
                    delta = -delta_remove(s, x)
                    if random.random() < math.exp(delta / T):
                        s[x] = False
                        current_score += delta
                        no_improve += 1
            else:  # Swap
                if np.any(s):
                    x_out = random.choice(np.where(s)[0].tolist())
                    candidates = [x for x in range(N) if not s[x]]
                    if candidates:
                        x_in = random.choice(candidates)
                        if is_valid_swap(s, x_out, x_in):
                            loss = delta_remove(s, x_out)
                            gain = 0
                            for ap_idx in elem_to_3aps[x_in]:
                                a, b, c = three_aps_arr[ap_idx]
                                if x_in == a:
                                    if s[b] and s[c] and b != x_out and c != x_out:
                                        gain += 1
                                elif x_in == b:
                                    if s[a] and s[c] and a != x_out and c != x_out:
                                        gain += 1
                                else:
                                    if s[a] and s[b] and a != x_out and b != x_out:
                                        gain += 1
                            delta = gain - loss
                            if delta > 0 or random.random() < math.exp(delta / T):
                                s[x_out] = False
                                s[x_in] = True
                                current_score += delta
                                no_improve = 0
                            else:
                                no_improve += 1
            
            T *= cooling
        
        if current_score > best_score:
            best_score = current_score
            best_s = s.copy()
    
    result = [x + 1 for x in range(N) if best_s[x]]
    return sorted(result)
