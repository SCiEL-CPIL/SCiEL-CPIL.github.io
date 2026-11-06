import time
import itertools
import random

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    n = len(S)
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            vals = [S[i] for i in subset]
            if is_sidon(vals):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    best_set = None
    best_size = 999
    
    def evaluate_and_update(S):
        nonlocal best_set, best_size
        sz = max_sidon_size(S)
        if sz < best_size:
            best_size = sz
            best_set = S[:]
        return sz
    
    random.seed(999)
    
    # Strategy: create sets with many sum collisions
    # Use structure: pairs that sum to same value
    candidates = []
    
    # Type 1: Tight arithmetic-like structures
    for _ in range(300):
        base = random.randint(0, 110)
        step = random.randint(1, 5)
        s = []
        for i in range(12):
            val = base + i * step + random.randint(-2, 2)
            val = max(0, min(128, val))
            s.append(val)
        s = sorted(set(s))
        while len(s) < 12:
            s.append(random.randint(0, 128))
        s = sorted(list(set(s))[:12])
        if len(s) == 12:
            candidates.append(s)
    
    # Type 2: Random but clustered
    for _ in range(500):
        center = random.randint(10, 118)
        s = []
        for i in range(12):
            val = center + random.randint(-15, 15)
            val = max(0, min(128, val))
            s.append(val)
        s = sorted(set(s))
        while len(s) < 12:
            s.append(random.randint(0, 128))
        s = sorted(list(set(s))[:12])
        if len(s) == 12:
            candidates.append(s)
    
    # Type 3: Pure random
    for _ in range(200):
        s = sorted(random.sample(range(129), 12))
        candidates.append(s)
    
    # Evaluate
    for c in candidates:
        if time.time() > deadline:
            break
        evaluate_and_update(c)
    
    if best_set is None:
        best_set = list(range(12))
        best_size = max_sidon_size(best_set)
    
    # Local search with hill climbing
    current = best_set[:]
    current_size = best_size
    stall = 0
    
    while time.time() < deadline - 3:
        improved = False
        for i in range(12):
            if time.time() > deadline - 3:
                break
            old_val = current[i]
            # Try nearby values and random values
            candidates_val = set()
            for delta in range(-5, 6):
                nv = old_val + delta
                if 0 <= nv <= 128 and nv != old_val:
                    candidates_val.add(nv)
            candidates_val.update(random.sample(range(129), 5))
            
            for new_val in candidates_val:
                if new_val == old_val:
                    continue
                trial = current[:]
                trial[i] = new_val
                if len(set(trial)) != 12:
                    continue
                trial.sort()
                sz = max_sidon_size(trial)
                if sz < current_size:
                    current = trial
                    current_size = sz
                    if sz < best_size:
                        best_size = sz
                        best_set = trial[:]
                    improved = True
                    break
                elif sz == current_size and random.random() < 0.2:
                    current = trial
                    current_size = sz
                    improved = True
                    break
        
        if not improved:
            stall += 1
            if stall > 50:
                # Restart from best with perturbation
                current = best_set[:]
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, 11)
                    current[idx] = random.randint(0, 128)
                current = sorted(list(set(current)))
                while len(current) < 12:
                    current.append(random.randint(0, 128))
                current = sorted(list(set(current))[:12])
                current_size = max_sidon_size(current)
                stall = 0
    
    return best_set
