"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _score_set(S_list):
    """Count distinct d > 0 such that a, a+d, a+2d all in S for some a."""
    S_set = set(S_list)
    distinct_d = set()
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            d = b - a
            if a + 2 * d in S_set:
                distinct_d.add(d)
    return len(distinct_d)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    random.seed(42)
    
    # Initial: range(1,33) gives 15
    best_set = list(range(1, 33))
    best_score = _score_set(best_set)
    
    # Try structured initial sets
    candidates = [
        list(range(1, 33)),
        list(range(1, 33, 1)),
        [1 + i * 16 for i in range(32)],
        [1 + i * 15 for i in range(32)],
        list(range(200, 232)),
        list(range(1, 17)) + list(range(250, 266)),
        list(range(1, 17)) + list(range(240, 256)),
        # Geometric-ish
        [int(1 * (1.5 ** i)) for i in range(32)] if int(1 * (1.5 ** 31)) <= 512 else None,
    ]
    
    for cand in candidates:
        if cand is None:
            continue
        if len(set(cand)) == 32 and all(1 <= x <= 512 for x in cand):
            s = _score_set(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
    
    # Simulated annealing
    current_set = list(best_set)
    current_score = best_score
    T = 3.0
    cooling = 0.9998
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        idx = random.randint(0, 31)
        old_val = current_set[idx]
        
        best_trial = None
        best_trial_score = current_score
        
        # Try several random replacements
        for _ in range(5):
            new_val = random.randint(1, 512)
            if new_val == old_val:
                continue
            trial = list(current_set)
            trial[idx] = new_val
            if len(set(trial)) == 32:
                ts = _score_set(trial)
                if ts > best_trial_score:
                    best_trial_score = ts
                    best_trial = trial
        
        # Try swap
        if random.random() < 0.5:
            idx2 = random.randint(0, 31)
            if idx2 != idx:
                trial = list(current_set)
                trial[idx], trial[idx2] = trial[idx2], trial[idx]
                ts = _score_set(trial)
                if ts > best_trial_score:
                    best_trial_score = ts
                    best_trial = trial
        
        # Try 2-swap (replace two elements)
        if random.random() < 0.3:
            idx2 = random.randint(0, 31)
            if idx2 != idx:
                new_val1 = random.randint(1, 512)
                new_val2 = random.randint(1, 512)
                trial = list(current_set)
                trial[idx] = new_val1
                trial[idx2] = new_val2
                if len(set(trial)) == 32:
                    ts = _score_set(trial)
                    if ts > best_trial_score:
                        best_trial_score = ts
                        best_trial = trial
        
        if best_trial is not None:
            delta = best_trial_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set = best_trial
                current_score = best_trial_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling
        
        if no_improve > max_no_improve:
            no_improve = 0
            T = max(T, 1.5)
    
    return best_set
# EVOLVE-BLOCK-END

