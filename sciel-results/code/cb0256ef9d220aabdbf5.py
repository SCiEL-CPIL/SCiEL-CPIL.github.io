import time
import random

def _greedy_build():
    S = []
    S_set = set()
    sums = set()
    for x in range(1, 4097):
        # Check if x can be added
        new_sums = set()
        
        s = 3 * x
        if s in sums:
            continue
        new_sums.add(s)
        
        valid = True
        for y in S:
            s = 2 * x + y
            if s in sums or s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if not valid:
            continue
            
        for y in S:
            s = x + 2 * y
            if s in sums or s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if not valid:
            continue
            
        n = len(S)
        for i in range(n):
            yi = S[i]
            for j in range(i, n):
                zj = S[j]
                s = x + yi + zj
                if s in sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                break
        if not valid:
            continue
            
        S.append(x)
        S_set.add(x)
        sums.update(new_sums)
        
    return S, sums

def _rebuild_sums(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Initial Greedy
    best_S, _ = _greedy_build()
    best_size = len(best_S)
    
    if best_size == 0:
        return best_S
        
    rng = random.Random(42)
    
    # 2. Local Search
    while time.time() < deadline:
        # Pick a random element to remove
        idx = rng.randint(0, len(best_S) - 1)
        removed_val = best_S[idx]
        
        # Create trial set
        trial_S = best_S[:idx] + best_S[idx+1:]
        trial_S_set = set(trial_S)
        
        # Rebuild sums for trial set
        trial_sums = _rebuild_sums(trial_S)
        
        # Try to add candidates in increasing order
        improved = False
        for x in range(1, 4097):
            if x in trial_S_set:
                continue
                
            # Check if x can be added to trial_S
            new_sums = set()
            
            s = 3 * x
            if s in trial_sums:
                continue
            new_sums.add(s)
            
            valid = True
            for y in trial_S:
                s = 2 * x + y
                if s in trial_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
                
            for y in trial_S:
                s = x + 2 * y
                if s in trial_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
                
            n = len(trial_S)
            for i in range(n):
                yi = trial_S[i]
                for j in range(i, n):
                    zj = trial_S[j]
                    s = x + yi + zj
                    if s in trial_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
                
            # Valid, add to trial_S
            trial_S.append(x)
            trial_S_set.add(x)
            trial_sums.update(new_sums)
            
            # Check if we improved
            if len(trial_S) > best_size:
                best_S = trial_S
                best_size = len(best_S)
                improved = True
                break # Restart search from new best
        
        # If not improved, we just discard the trial.
        
    return best_S
