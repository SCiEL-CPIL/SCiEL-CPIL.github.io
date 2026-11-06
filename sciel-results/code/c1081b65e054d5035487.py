import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 78
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = 129
    
    def evaluate(A):
        counts = [0] * num_targets
        A_sorted = sorted(A)
        k = len(A_sorted)
        for i in range(k):
            ai = A_sorted[i]
            for j in range(i, k):
                s = ai + A_sorted[j]
                if target_low <= s <= target_high:
                    counts[s - target_low] += 1
        if min(counts) == 0:
            return float('inf')
        return max(counts)
    
    def greedy_construct():
        A = []
        counts = [0] * num_targets
        A_set = set()
        for _ in range(50):
            best_elem = None
            best_score = None
            for x in range(n + 1):
                if x in A_set:
                    continue
                new_counts = counts[:]
                for y in A:
                    s = x + y
                    if target_low <= s <= target_high:
                        new_counts[s - target_low] += 1
                s = x + x
                if target_low <= s <= target_high:
                    new_counts[s - target_low] += 1
                
                if min(new_counts) == 0:
                    continue
                
                peak = max(new_counts)
                uncovered = sum(1 for c in new_counts if c == 0)
                score = (peak, uncovered)
                if best_score is None or score < best_score:
                    best_score = score
                    best_elem = x
            
            if best_elem is None:
                break
            A.append(best_elem)
            A_set.add(best_elem)
            for y in A[:-1]:
                s = best_elem + y
                if target_low <= s <= target_high:
                    counts[s - target_low] += 1
            s = best_elem + best_elem
            if target_low <= s <= target_high:
                counts[s - target_low] += 1
        
        return A
    
    best_A = None
    best_peak = float('inf')
    
    # Seed
    seed_A = list(range(64, 129))
    sp = evaluate(seed_A)
    if sp < best_peak:
        best_peak = sp
        best_A = seed_A[:]
    
    # Greedy
    if time.time() < deadline - 40:
        gA = greedy_construct()
        gp = evaluate(gA)
        if gp < best_peak:
            best_peak = gp
            best_A = gA[:]
    
    # SA with restarts
    for restart in range(25):
        if time.time() > deadline - 8:
            break
        
        if restart == 0:
            current_A = best_A[:]
        elif restart % 4 == 1:
            current_A = seed_A[:]
        elif restart % 4 == 2:
            size = random.randint(25, 38)
            current_A = random.sample(range(n + 1), size)
        else:
            size = random.randint(30, 45)
            current_A = random.sample(range(n + 1), size)
        
        current_peak = evaluate(current_A)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = current_A[:]
        
        T = 2.0
        T_min = 0.001
        cooling = 0.9997
        
        while time.time() < deadline - 4:
            if T < T_min:
                T = T_min
            
            r = random.random()
            
            if r < 0.25 and len(current_A) > 12:
                idx = random.randint(0, len(current_A) - 1)
                new_A = current_A[:idx] + current_A[idx+1:]
                new_peak = evaluate(new_A)
                delta = new_peak - current_peak
                if delta <= 0 or (delta != float('inf') and random.random() < np.exp(-delta / T)):
                    current_A = new_A
                    current_peak = new_peak
            elif r < 0.60:
                in_set = set(current_A)
                candidates = [x for x in range(n + 1) if x not in in_set]
                if candidates:
                    x = random.choice(candidates)
                    new_A = current_A + [x]
                    new_peak = evaluate(new_A)
                    delta = new_peak - current_peak
                    if delta <= 0 or (delta != float('inf') and random.random() < np.exp(-delta / T)):
                        current_A = new_A
                        current_peak = new_peak
            elif len(current_A) > 12:
                idx = random.randint(0, len(current_A) - 1)
                in_set = set(current_A)
                candidates = [x for x in range(n + 1) if x not in in_set]
                if candidates:
                    x = random.choice(candidates)
                    new_A = current_A[:]
                    new_A[idx] = x
                    new_peak = evaluate(new_A)
                    delta = new_peak - current_peak
                    if delta <= 0 or (delta != float('inf') and random.random() < np.exp(-delta / T)):
                        current_A = new_A
                        current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = current_A[:]
            
            T *= cooling
    
    # Verify
    if best_A is not None:
        final_peak = evaluate(best_A)
        if final_peak == float('inf'):
            return seed_A
    return best_A
