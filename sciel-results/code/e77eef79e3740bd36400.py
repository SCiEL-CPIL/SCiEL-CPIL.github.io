import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Greedy Initialization
    def greedy_construct():
        current_mask = 0
        current_set = set()
        for _ in range(N):
            best_val = -1
            best_score = -1
            for x in range(1, MAX_VAL + 1):
                if x in current_set:
                    continue
                new_mask = current_mask | (1 << x)
                s = score_mask(new_mask)
                if s > best_score:
                    best_score = s
                    best_val = x
            if best_val != -1:
                current_set.add(best_val)
                current_mask |= (1 << best_val)
        return list(current_set), current_mask, best_score

    # Generate initial candidates
    candidates = []
    
    # Greedy start
    g_list, g_mask, g_score = greedy_construct()
    candidates.append((g_list, g_mask, g_score))

    # Random starts
    for _ in range(20):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        m = list_to_mask(lst)
        s = score_mask(m)
        candidates.append((lst, m, s))
        
    # Structured starts
    lst_dense = list(range(1, 33))
    m_dense = list_to_mask(lst_dense)
    s_dense = score_mask(m_dense)
    candidates.append((lst_dense, m_dense, s_dense))
    
    lst_ap = list(range(1, 1 + 31 * 10, 10))
    m_ap = list_to_mask(lst_ap)
    s_ap = score_mask(m_ap)
    candidates.append((lst_ap, m_ap, s_ap))

    # Find best initial
    best_score = -1
    best_list = None
    best_mask = 0
    
    for lst, m, s in candidates:
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    # Simulated Annealing
    current_list = best_list[:]
    current_score = best_score
    current_mask = best_mask
    current_set = set(current_list)

    T = 10.0
    cooling_rate = 0.999
    no_improve_count = 0
    no_improve_threshold = 2000
    restart_temperature = 5.0

    while time.time() < deadline:
        move_type = random.random()
        if move_type < 0.5:
            # Single change
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]
            
            while True:
                new_val = random.randint(1, MAX_VAL)
                if new_val == old_val or new_val in current_set:
                    continue
                break
            
            new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
            new_score = score_mask(new_mask)
            delta = new_score - current_score
            
            if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
                current_list[idx] = new_val
                current_set.remove(old_val)
                current_set.add(new_val)
                current_score = new_score
                current_mask = new_mask
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list[:]
                    best_mask = current_mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Remove 2, Add 2
            i, j = random.sample(range(N), 2)
            old_val1 = current_list[i]
            old_val2 = current_list[j]
            
            new_val1 = -1
            new_val2 = -1
            attempts = 0
            while (new_val1 == -1 or new_val2 == -1) and attempts < 100:
                v1 = random.randint(1, MAX_VAL)
                v2 = random.randint(1, MAX_VAL)
                if v1 != old_val1 and v1 != old_val2 and v1 not in current_set:
                    new_val1 = v1
                if v2 != old_val1 and v2 != old_val2 and v2 not in current_set and v2 != new_val1:
                    new_val2 = v2
                attempts += 1
            
            if new_val1 != -1 and new_val2 != -1:
                new_mask = current_mask ^ (1 << old_val1) ^ (1 << old_val2) ^ (1 << new_val1) ^ (1 << new_val2)
                new_score = score_mask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
                    current_list[i] = new_val1
                    current_list[j] = new_val2
                    current_set.remove(old_val1)
                    current_set.remove(old_val2)
                    current_set.add(new_val1)
                    current_set.add(new_val2)
                    current_score = new_score
                    current_mask = new_mask
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list[:]
                        best_mask = current_mask
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        T *= cooling_rate
        if T < 0.1:
            T = 0.1
            
        if no_improve_count > no_improve_threshold:
            current_list = best_list[:]
            current_score = best_score
            current_mask = best_mask
            current_set = set(current_list)
            T = restart_temperature
            no_improve_count = 0
            restart_temperature *= 0.5
            if restart_temperature < 1.0:
                restart_temperature = 1.0

    return best_list
# EVOLVE-BLOCK-END
