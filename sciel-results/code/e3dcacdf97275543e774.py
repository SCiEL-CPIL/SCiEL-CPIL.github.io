import time
import random

def _check_validity(values, x):
    """
    Check if adding x to the end of values results in all distinct subarray sums.
    values: list of ints
    x: int
    Returns True if valid, False otherwise.
    """
    # We need to check new subarray sums ending at x.
    # These are x, x+values[-1], x+values[-1]+values[-2], etc.
    # We need to ensure these are not in the set of existing sums.
    # And we need to ensure these new sums are distinct from each other.
    
    # Optimization: We can maintain the set of sums incrementally.
    # But for simplicity in a helper function, we compute the new tails.
    
    tails = []
    total = x
    tails.append(total)
    for a in reversed(values):
        total += a
        tails.append(total)
        
    # Check for duplicates within tails
    if len(set(tails)) != len(tails):
        return False
        
    # The caller should pass the existing sums set to avoid recomputation or global state issues
    # For this helper, we'll just return the tails and let the caller check against existing sums.
    # Actually, it's better to do the check inside the loop in the main strategy.
    pass

def _greedy_smallest():
    values = []
    sums = set()
    for x in range(1, 257):
        # Check if x can be added
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
            
        if not sums.intersection(tails) and len(set(tails)) == len(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_largest():
    values = []
    sums = set()
    next_min = 1
    while next_min <= 256:
        found = False
        # Try largest first
        for x in range(256, next_min - 1, -1):
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
                
            if not sums.intersection(tails) and len(set(tails)) == len(tails):
                values.append(x)
                sums.update(tails)
                next_min = x + 1
                found = True
                break
        if not found:
            break
    return values

def _greedy_random(retries=50):
    best = []
    start_time = time.time()
    for _ in range(retries):
        if time.time() - start_time > 50: # Leave time for other strategies
            break
        values = []
        sums = set()
        next_min = 1
        # Shuffle candidates? Or pick randomly from valid range?
        # Let's pick randomly from the valid range [next_min, 256]
        # But we must ensure strictly increasing.
        while next_min <= 256:
            candidates = list(range(next_min, 257))
            random.shuffle(candidates)
            found = False
            for x in candidates:
                tails = []
                total = x
                tails.append(total)
                for a in reversed(values):
                    total += a
                    tails.append(total)
                    
                if not sums.intersection(tails) and len(set(tails)) == len(tails):
                    values.append(x)
                    sums.update(tails)
                    next_min = x + 1
                    found = True
                    break
            if not found:
                break
        if len(values) > len(best):
            best = values[:]
    return best

def construct():
    start_time = time.time()
    deadline = 80 # 80s limit to be safe
    
    candidates = []
    
    # Strategy 1: Smallest Greedy
    v1 = _greedy_smallest()
    candidates.append(v1)
    
    if time.time() - start_time > deadline:
        return max(candidates, key=len)

    # Strategy 2: Largest Greedy
    v2 = _greedy_largest()
    candidates.append(v2)
    
    if time.time() - start_time > deadline:
        return max(candidates, key=len)

    # Strategy 3: Randomized Greedy
    v3 = _greedy_random()
    candidates.append(v3)
    
    best = max(candidates, key=len)
    return best
