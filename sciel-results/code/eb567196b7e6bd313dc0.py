import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    S = set()
    sum_counts = [0] * 2001
    violation_count = 0
    
    def can_add(x):
        if sum_counts[2*x] >= 2:
            return False
        for y in S:
            if sum_counts[x+y] >= 2:
                return False
        return True

    def add(x):
        nonlocal violation_count
        S.add(x)
        # 2x
        old_c = sum_counts[2*x]
        sum_counts[2*x] += 1
        if old_c == 2:
            violation_count += 1
            
        for y in S:
            if y == x: continue
            s = x + y
            old_c = sum_counts[s]
            sum_counts[s] += 1
            if old_c == 2:
                violation_count += 1

    def remove(x):
        nonlocal violation_count
        S.remove(x)
        old_c = sum_counts[2*x]
        sum_counts[2*x] -= 1
        if old_c == 3:
            violation_count -= 1
            
        for y in S:
            s = x + y
            old_c = sum_counts[s]
            sum_counts[s] -= 1
            if old_c == 3:
                violation_count -= 1

    # Greedy
    for x in range(1, 1001):
        if can_add(x):
            add(x)
            
    best_S = S.copy()
    best_size = len(S)
    
    T = 50.0
    lambda_pen = 100.0
    
    while time.time() < deadline:
        x = random.randint(1, 1000)
        
        if x in S:
            # Remove
            dp = 0
            old_c = sum_counts[2*x]
            if old_c > 2:
                dp += (old_c - 3)**2 - (old_c - 2)**2
            
            for y in S:
                if y == x: continue
                s = x + y
                old_c = sum_counts[s]
                if old_c > 2:
                    dp += (old_c - 3)**2 - (old_c - 2)**2
            
            delta_E = 1 + lambda_pen * dp
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                remove(x)
                if violation_count == 0 and len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
        else:
            # Add
            dp = 0
            old_c = sum_counts[2*x]
            if old_c >= 2:
                dp += (old_c + 1 - 2)**2 - (old_c - 2)**2
            
            for y in S:
                s = x + y
                old_c = sum_counts[s]
                if old_c >= 2:
                    dp += (old_c + 1 - 2)**2 - (old_c - 2)**2
            
            delta_E = -1 + lambda_pen * dp
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                add(x)
                if violation_count == 0 and len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
        
        T *= 0.99
        if T < 1.0:
            T = 1.0
            
    return list(best_S)
