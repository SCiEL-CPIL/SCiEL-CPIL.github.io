import time
import random

def _score_mask(mask):
    """Count distinct d>0 such that some a, a+d, a+2d are all in mask (bits 1..512)."""
    count = 0
    for d in range(1, 257):
        if mask & (mask >> d) & (mask >> (2 * d)):
            count += 1
    return count

def _make_mask(s):
    mask = 0
    for x in s:
        mask |= (1 << x)
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    rng = random.Random(42)
    all_elems = list(range(1, 513))
    
    best_overall_score = -1
    best_overall_s = None
    
    def evaluate_and_update(s):
        nonlocal best_overall_score, best_overall_s
        mask = _make_mask(s)
        sc = _score_mask(mask)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_s = list(s)
        return mask, sc
    
    # Initial candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    for _ in range(30):
        candidates.append(rng.sample(all_elems, 32))
    for _ in range(10):
        candidates.append(rng.sample(range(1, 128), 32))
    candidates.append(list(range(1, 513, 16))[:32])
    candidates.append(list(range(2, 513, 16))[:32])
    candidates.append(list(range(1, 513, 15))[:32])
    for _ in range(10):
        b1 = rng.sample(range(1, 160), 16)
        b2 = rng.sample(range(350, 513), 16)
        candidates.append(b1 + b2)
    for _ in range(10):
        b1 = rng.sample(range(1, 100), 10)
        b2 = rng.sample(range(200, 350), 12)
        b3 = rng.sample(range(400, 513), 10)
        candidates.append(b1 + b2 + b3)
    
    for s in candidates:
        evaluate_and_update(s)
    
    # Local search from best
    def local_search(s, deadline):
        nonlocal best_overall_score, best_overall_s
        s = list(s)
        mask = _make_mask(s)
        sc = _score_mask(mask)
        in_set = set(s)
        n = 32
        
        while time.time() < deadline:
            improved = False
            for _ in range(5000):
                if time.time() >= deadline:
                    break
                i = rng.randrange(n)
                old = s[i]
                new = rng.choice(all_elems)
                if new == old or new in in_set:
                    continue
                new_mask = (mask & ~(1 << old)) | (1 << new)
                new_sc = _score_mask(new_mask)
                if new_sc > sc:
                    s[i] = new
                    mask = new_mask
                    sc = new_sc
                    in_set.discard(old)
                    in_set.add(new)
                    improved = True
                    if sc > best_overall_score:
                        best_overall_score = sc
                        best_overall_s = list(s)
                    break
            if not improved:
                # Perturbation: random swap
                i = rng.randrange(n)
                old = s[i]
                new = rng.choice(all_elems)
                if new in in_set:
                    continue
                s[i] = new
                mask = (mask & ~(1 << old)) | (1 << new)
                in_set.discard(old)
                in_set.add(new)
                sc = _score_mask(mask)
    
    # Local search from best candidate
    local_search(best_overall_s, deadline - 5.0)
    
    # Multi-start from other candidates
    for s in candidates:
        if time.time() >= deadline - 5.0:
            break
        mask = _make_mask(s)
        sc = _score_mask(mask)
        if sc >= best_overall_score - 3:
            local_search(s, deadline - 5.0)
    
    # More random restarts
    while time.time() < deadline - 5.0:
        s = rng.sample(all_elems, 32)
        mask = _make_mask(s)
        sc = _score_mask(mask)
        if sc >= best_overall_score - 5:
            local_search(s, deadline - 5.0)
    
    return best_overall_s
