import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    lo, hi = 128, 256
    num_targets = hi - lo + 1
    
    def get_sums(A_list):
        """Return numpy array of counts for sums in [lo, hi]."""
        k = len(A_list)
        counts = np.zeros(hi + 1, dtype=np.int32)
        arr = np.array(A_list, dtype=np.int32)
        for i in range(k):
            s = arr[i] + arr[i:]
            mask = (s >= lo) & (s <= hi)
            if mask.any():
                idx = s[mask]
                np.add.at(counts, idx, 1)
        return counts
    
    def eval_counts(counts):
        tc = counts[lo:hi+1]
        missing = int(np.sum(tc == 0))
        if missing > 0:
            return None, missing
        return int(np.max(tc)), 0
    
    def eval_set(A_set):
        A_list = sorted(A_set)
        if len(A_list) < 2:
            return None, num_targets
        counts = get_sums(A_list)
        return eval_counts(counts)
    
    rng = random.Random(777)
    
    best_A = None
    best_peak = float('inf')
    
    # Seed solution
    seed_A = set(range(64, 129))
    p, m = eval_set(seed_A)
    if p is not None:
        best_A = set(seed_A)
        best_peak = p
    
    # Phase 1: Try to find small feasible sets via random init + repair
    for size in [18, 20, 22, 25, 28, 30]:
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            A = set(rng.sample(range(0, n + 1), size))
            # Repair: add elements to cover missing targets
            for repair_iter in range(50):
                p, missing = eval_set(A)
                if p is not None:
                    break
                if time.time() > deadline - 10:
                    break
                # Find a missing target
                A_list = sorted(A)
                counts = get_sums(A_list)
                missing_targets = [t for t in range(lo, hi+1) if counts[t] == 0]
                if not missing_targets:
                    break
                t = rng.choice(missing_targets)
                # Find a pair (a,b) in [0,128] with a+b=t, a≤b
                # Add one of them to A
                candidates = []
                for a in range(max(0, t - n), min(n, t // 2) + 1):
                    b = t - a
                    if 0 <= a <= n and 0 <= b <= n and a <= b:
                        candidates.append((a, b))
                if candidates:
                    a, b = rng.choice(candidates)
                    if rng.random() < 0.5:
                        A.add(a)
                    else:
                        A.add(b)
            
            if time.time() > deadline - 10:
                break
            p, m = eval_set(A)
            if p is not None and p < best_peak:
                best_A = set(A)
                best_peak = p
    
    if best_A is None:
        best_A = set(range(64, 129))
        best_peak = 33
    
    # Phase 2: Local search to minimize peak
    current_A = set(best_A)
    current_peak = best_peak
    temperature = 3.0
    cooling = 0.9995
    
    while time.time() < deadline - 1:
        mt = rng.random()
        if mt < 0.35:  # add
            if len(current_A) < n + 1:
                x = rng.randint(0, n)
                if x in current_A:
                    continue
                new_A = current_A | {x}
            else:
                continue
        elif mt < 0.65:  # remove
            if len(current_A) > 5:
                x = rng.choice(sorted(current_A))
                new_A = current_A - {x}
            else:
                continue
        else:  # swap
            if len(current_A) < 2:
                continue
            x = rng.choice(sorted(current_A))
            y = rng.randint(0, n)
            if y == x or y in current_A:
                continue
            new_A = (current_A - {x}) | {y}
        
        new_peak, new_missing = eval_set(new_A)
        
        if new_peak is not None:
            delta = new_peak - current_peak
            if delta <= 0:
                current_A = new_A
                current_peak = new_peak
                if current_peak < best_peak:
                    best_A = set(current_A)
                    best_peak = current_peak
            else:
                prob = np.exp(-delta / max(temperature, 0.001))
                if rng.random() < prob:
                    current_A = new_A
                    current_peak = new_peak
        else:
            if new_missing <= 1 and rng.random() < 0.003:
                current_A = new_A
                current_peak = current_peak + new_missing * 15
        
        temperature = max(temperature * cooling, 0.001)
    
    return list(best_A)
