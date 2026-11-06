import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    all_3aps = []
    all_4aps = []
    
    # 3-AP: (a, a+d, a+2d)
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                all_3aps.append((a, b, c))
                
    # 4-AP: (a, a+d, a+2d, a+3d)
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_4aps.append((a, b, c, e))
    
    # Index APs by element
    threes_by_elem = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(all_3aps):
        threes_by_elem[a].append(idx)
        threes_by_elem[b].append(idx)
        threes_by_elem[c].append(idx)
        
    fours_by_elem = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(all_4aps):
        fours_by_elem[a].append(idx)
        fours_by_elem[b].append(idx)
        fours_by_elem[c].append(idx)
        fours_by_elem[e].append(idx)
        
    # Helper to check if adding x creates a 4-AP given current S (bool array)
    def creates_4ap(S, x):
        for idx in fours_by_elem[x]:
            a, b, c, e = all_4aps[idx]
            if x == a:
                if S[b] and S[c] and S[e]:
                    return True
            elif x == b:
                if S[a] and S[c] and S[e]:
                    return True
            elif x == c:
                if S[a] and S[b] and S[e]:
                    return True
            elif x == e:
                if S[a] and S[b] and S[c]:
                    return True
        return False
        
    # Helper to count 3-APs gained/lost by adding/removing x
    def gain_3aps(S, x):
        count = 0
        for idx in threes_by_elem[x]:
            a, b, c = all_3aps[idx]
            if x == a:
                if S[b] and S[c]:
                    count += 1
            elif x == b:
                if S[a] and S[c]:
                    count += 1
            elif x == c:
                if S[a] and S[b]:
                    count += 1
        return count

    def count_3aps(S):
        count = 0
        for a, b, c in all_3aps:
            if S[a] and S[b] and S[c]:
                count += 1
        return count

    # Greedy construction: add elements that maximize 3-AP gain without creating 4-APs
    def greedy_construct():
        S = [False] * (N + 1)
        S_set = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_gain = -1
            for x in remaining:
                if creates_4ap(S, x):
                    continue
                g = gain_3aps(S, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x is None:
                break
                
            S[best_x] = True
            S_set.add(best_x)
            remaining.remove(best_x)
            
        return S, S_set

    # Simulated Annealing with swap bias
    def simulated_annealing(S_init, S_set_init, time_budget=15):
        S = S_init[:]
        S_set = set(S_set_init)
        
        current_3ap = count_3aps(S)
        
        best_S = S[:]
        best_S_set = set(S_set)
        best_3ap = current_3ap
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.005
        
        all_elements = list(range(1, N + 1))
        
        while time.time() < start_time + time_budget and T > min_T:
            # Choose move type with bias towards swaps
            r = random.random()
            if r < 0.5:
                move_type = 2  # Swap
            elif r < 0.8:
                move_type = 0  # Add
            else:
                move_type = 1  # Remove
            
            if move_type == 0:  # Add
                candidates = [x for x in all_elements if not S[x]]
                if not candidates:
                    T *= cooling_rate
                    continue
                x = random.choice(candidates)
                
                if creates_4ap(S, x):
                    T *= cooling_rate
                    continue
                    
                gain = gain_3aps(S, x)
                delta = gain
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[x] = True
                    S_set.add(x)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                        
            elif move_type == 1:  # Remove
                if not S_set:
                    T *= cooling_rate
                    continue
                x = random.choice(list(S_set))
                
                loss = gain_3aps(S, x)
                delta = -loss
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[x] = False
                    S_set.discard(x)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                        
            else:  # Swap
                if not S_set:
                    T *= cooling_rate
                    continue
                x = random.choice(list(S_set))
                candidates = [y for y in all_elements if not S[y]]
                if not candidates:
                    T *= cooling_rate
                    continue
                y = random.choice(candidates)
                
                # Temporarily remove x
                S[x] = False
                S_set.discard(x)
                
                if creates_4ap(S, y):
                    S[x] = True
                    S_set.add(x)
                    T *= cooling_rate
                    continue
                
                loss_x = gain_3aps(S, x)
                gain_y = gain_3aps(S, y)
                delta = gain_y - loss_x
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[y] = True
                    S_set.add(y)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                else:
                    S[x] = True
                    S_set.add(x)
                    
            T *= cooling_rate
            
        return best_S, best_S_set, best_3ap

    # Run multiple restarts with different strategies
    best_overall_3ap = 0
    best_overall_S = []
    
    # First: pure greedy
    if time.time() < deadline - 60:
        S_greedy, S_set_greedy = greedy_construct()
        best_S, best_S_set, best_3ap = simulated_annealing(S_greedy, S_set_greedy, time_budget=20)
        
        if best_3ap > best_overall_3ap:
            best_overall_3ap = best_3ap
            best_overall_S = best_S_set
    
    # Then: randomized greedy + SA
    num_restarts = 8
    time_per_restart = 8
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Randomized greedy: shuffle and pick best available
        S = [False] * (N + 1)
        S_set = set()
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        # First pass: add elements in shuffled order if no 4-AP
        for x in remaining[:]:
            if creates_4ap(S, x):
                remaining.remove(x)
                continue
            S[x] = True
            S_set.add(x)
        
        # Then do a few greedy improvements
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in list(S_set):
                # Try removing x and see if we can add something better
                pass  # Simplified: just do SA
        
        best_S, best_S_set, best_3ap = simulated_annealing(S, S_set, time_budget=time_per_restart)
        
        if best_3ap > best_overall_3ap:
            best_overall_3ap = best_3ap
            best_overall_S = best_S_set
    
    # Final polish on the best found
    if best_overall_S and time.time() < deadline - 3:
        S_final = [False] * (N + 1)
        for x in best_overall_S:
            S_final[x] = True
        best_S, best_S_set, best_3ap = simulated_annealing(S_final, set(best_overall_S), time_budget=3)
        
        if best_3ap > best_overall_3ap:
            best_overall_3ap = best_3ap
            best_overall_S = best_S_set
    
    if not best_overall_S:
        return [1]
        
    return sorted(best_overall_S)
