import time
import random
import itertools

def is_sidon_fast(subset):
    """Check if subset has all pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    max_sum = 2 * s[-1]
    seen = bytearray(max_sum + 1)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if seen[sm]:
                return False
            seen[sm] = 1
    return True

def max_sidon_size(S):
    """Find largest Sidon subset of S."""
    S = list(S)
    n = len(S)
    for k in range(n, 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon_fast(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    candidates = []
    
    # APs with small differences - most sum collisions
    for d in range(1, 10):
        max_start = 128 - 11 * d
        if max_start < 0:
            continue
        for start in range(0, max_start + 1):
            cand = [start + i * d for i in range(12)]
            candidates.append(cand)
    
    # Random perturbations of small-d APs
    random.seed(42)
    for _ in range(500):
        d = random.randint(1, 4)
        max_start = 128 - 11 * d
        if max_start < 0:
            continue
        start = random.randint(0, max_start)
        base = [start + i * d for i in range(12)]
        cand = base[:]
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, 11)
            delta = random.choice([-2, -1, 1, 2, 3])
            new_val = cand[idx] + delta
            if 0 <= new_val <= 128 and new_val not in cand:
                cand[idx] = new_val
        if len(set(cand)) == 12:
            candidates.append(sorted(cand))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    current_best = None
    current_best_val = 13
    
    # Prioritize candidates with smallest gaps (more collisions)
    def priority(c):
        c_sorted = sorted(c)
        diffs = [c_sorted[i+1] - c_sorted[i] for i in range(11)]
        return min(diffs)
    
    unique_candidates.sort(key=priority)
    
    for c in unique_candidates:
        if time.time() > deadline:
            break
        if len(c) != 12 or len(set(c)) != 12:
            continue
        val = max_sidon_size(c)
        if val < current_best_val:
            current_best_val = val
            current_best = sorted(c)
            if current_best_val <= 5:
                break
    
    if current_best is None:
        current_best = list(range(12))
        current_best_val = max_sidon_size(current_best)
    
    return current_best
