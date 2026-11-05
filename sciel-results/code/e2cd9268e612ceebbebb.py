import time
import random

def _score_set(sset):
    diffs = set()
    for a in sset:
        for b in sset:
            if b > a:
                d = b - a
                if a + 2*d in sset:
                    diffs.add(d)
    return len(diffs)

def construct():
    start = time.time()
    deadline = start + 140
    
    best = list(range(1, 33))
    best_set = set(best)
    best_score = _score_set(best_set)
    
    random.seed(42)
    
    while time.time() < deadline:
        idx = random.randint(0, 31)
        old_val = best[idx]
        new_val = random.randint(1, 512)
        
        if new_val == old_val or new_val in best_set:
            continue
        
        new_set = best_set - {old_val} | {new_val}
        new_score = _score_set(new_set)
        
        if new_score >= best_score:
            best_score = new_score
            best[idx] = new_val
            best_set = new_set
    
    return best
