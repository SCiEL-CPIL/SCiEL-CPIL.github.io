import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    total_subsets = 1 << n
    
    # Precompute comparability sets
    # comp_sets[i] contains j if i < j or j < i
    comp_sets = [set() for _ in range(total_subsets)]
    for i in range(total_subsets):
        for j in range(i + 1, total_subsets):
            if (i & j) == i:
                comp_sets[i].add(j)
                comp_sets[j].add(i)
            elif (i & j) == j:
                comp_sets[i].add(j)
                comp_sets[j].add(i)
    
    # Greedy Initialization
    def greedy_init():
        current = []
        current_set = set()
        # Start with chain 0, 1, 3, 7, ...
        # Mask for {0..k-1} is (1<<k)-1
        for k in range(n + 1):
            mask = (1 << k) - 1
            current.append(mask)
            current_set.add(mask)
            
        # Fill rest greedily
        # To speed up, we can just add randoms if greedy is too slow, 
        # but greedy is better for start.
        for _ in range(m - len(current)):
            best_cand = -1
            best_score = -1
            # Sample a subset of candidates to speed up init?
            # No, 4096 is small.
            for cand in range(total_subsets):
                if cand in current_set:
                    continue
                score = 0
                for c in current:
                    if c in comp_sets[cand]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand == -1:
                break
            current.append(best_cand)
            current_set.add(best_cand)
            
        return current, current_set

    current, current_set = greedy_init()
    
    # Calculate initial score and degrees
    deg = [0] * m
    score = 0
    for i in range(m):
        for j in range(i + 1, m):
            if current[i] in comp_sets[current[j]]:
                score += 1
                deg[i] += 1
                deg[j] += 1
                
    best_score = score
    best_current = list(current)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    T_min = 0.01
    
    while time.time() < deadline:
        idx_a = random.randint(0, m - 1)
        A = current[idx_a]
        
        B = random.randint(0, total_subsets - 1)
        if B in current_set:
            continue
            
        # Calculate gain_B
        gain_B = 0
        for k in range(m):
            if k == idx_a:
                continue
            if current[k] in comp_sets[B]:
                gain_B += 1
                
        delta = gain_B - deg[idx_a]
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # Avoid division by zero or underflow
            if T < 1e-6:
                accept = False
            else:
                # exp(delta / T)
                # delta is negative
                ratio = delta / T
                if ratio < -100:
                    accept = False
                else:
                    accept = random.random() < (2.71828 ** ratio)
                    
        if accept:
            score += delta
            
            # Update degrees
            for k in range(m):
                if k == idx_a:
                    continue
                x = current[k]
                if x in comp_sets[A]:
                    deg[k] -= 1
                if x in comp_sets[B]:
                    deg[k] += 1
            
            deg[idx_a] = gain_B
            current[idx_a] = B
            current_set.remove(A)
            current_set.add(B)
            
            if score > best_score:
                best_score = score
                best_current = list(current)
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Convert best_current masks to lists of integers
    result = []
    for mask in best_current:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    return result
