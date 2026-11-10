"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _get_consecutive_sums(values, x):
    """Get all new consecutive block sums ending at x appended to values."""
    tails = []
    total = x
    tails.append(total)
    for a in reversed(values):
        total += a
        tails.append(total)
    return tails

def _is_valid(values, sums, x):
    """Check if x can be appended to values without creating duplicate sums."""
    tails = _get_consecutive_sums(values, x)
    return len(sums.intersection(tails)) == 0

def _greedy_with_randomization(start_time, deadline, seed):
    """Greedy with randomization: at each step, try random candidates."""
    rng = random.Random(seed)
    values = []
    sums = set()
    used = set()
    
    while True:
        if time.time() > deadline:
            break
        # Generate candidates: unused values greater than last value
        last = values[-1] if values else 0
        candidates = [x for x in range(last + 1, 257) if x not in used]
        if not candidates:
            break
        
        # Shuffle candidates with some bias toward smaller values
        # Try a random subset first, then fall back to smallest
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        
        # Try candidates in shuffled order, but limit attempts per step
        found = False
        max_attempts = min(len(shuffled), 50)
        for i in range(max_attempts):
            x = shuffled[i]
            tails = _get_consecutive_sums(values, x)
            if not sums.intersection(tails):
                values.append(x)
                used.add(x)
                sums.update(tails)
                found = True
                break
        
        if not found:
            # Try the smallest unused value as fallback
            for x in candidates:
                tails = _get_consecutive_sums(values, x)
                if not sums.intersection(tails):
                    values.append(x)
                    used.add(x)
                    sums.update(tails)
                    found = True
                    break
            if not found:
                break
    
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 20  # 20s deadline, safe margin for 60s limit
    
    best = []
    
    # First, run the original deterministic greedy to get a baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    best = values
    
    # Now try randomized greedy with different seeds
    seed = 42
    max_restarts = 1000
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        result = _greedy_with_randomization(start_time, deadline, seed)
        if len(result) > len(best):
            best = result
        seed += 1
    
    return best
# EVOLVE-BLOCK-END
