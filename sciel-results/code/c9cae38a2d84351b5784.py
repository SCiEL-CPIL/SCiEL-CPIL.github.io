import time
import random
import math

def _max_abs_ap_sum(a):
    n = len(a)
    max_abs = 1
    for d in range(1, n + 1):
        for r in range(d):
            sub = []
            pos = r
            while pos < n:
                sub.append(a[pos])
                pos += d
            if not sub:
                continue
            prefix = [0] * (len(sub) + 1)
            for i in range(len(sub)):
                prefix[i+1] = prefix[i] + sub[i]
            min_pref = 0
            max_sub = 0
            for i in range(1, len(prefix)):
                diff = prefix[i] - min_pref
                if diff > max_sub:
                    max_sub = diff
                if prefix[i] < min_pref:
                    min_pref = prefix[i]
            max_pref = 0
            min_sub = 0
            for i in range(1, len(prefix)):
                diff = prefix[i] - max_pref
                if diff < min_sub:
                    min_sub = diff
                if prefix[i] > max_pref:
                    max_pref = prefix[i]
            if max_sub > max_abs:
                max_abs = max_sub
            if -min_sub > max_abs:
                max_abs = -min_sub
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 256
    best_a = None
    best_score = float('inf')
    rng = random.Random(42)
    
    # Multiple initial candidates
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates = [tm]
    for seed in [1, 7, 13, 42, 99, 123, 256, 512, 1000, 2000, 4242, 12345]:
        r = random.Random(seed)
        candidates.append([r.choice((-1, 1)) for _ in range(n)])
    
    for a in candidates:
        score = _max_abs_ap_sum(a)
        if score < best_score:
            best_score = score
            best_a = a[:]
    
    current_a = best_a[:]
    current_score = best_score
    temperature = 5.0
    
    while time.time() < deadline:
        idx = rng.randint(0, n - 1)
        old_val = current_a[idx]
        current_a[idx] = -old_val
        new_score = _max_abs_ap_sum(current_a)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_a = current_a[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                if rng.random() < math.exp(-delta / max(temperature, 0.1)):
                    current_score = new_score
                else:
                    current_a[idx] = old_val
            else:
                current_score = new_score
        
        temperature *= 0.9995
        if temperature < 0.1:
            temperature = 0.1
    
    return best_a
