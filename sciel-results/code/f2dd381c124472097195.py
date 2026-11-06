"""Difference-space optimization for 3-term progression coverage."""
import time
import random

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _score_bitmask(mask):
    """Count distinct d>0 such that some a has a, a+d, a+2d all in mask (1-indexed bits)."""
    count = 0
    # d ranges from 1 to 255 (since max value is 512, a+2d <= 512 => d <= 256, but a>=1 so d<=255.5)
    for d in range(1, 256):
        # shift mask by d bits; if mask & (mask >> d) & (mask >> 2*d) is nonzero, d is covered
        if mask & (mask >> d) & (mask >> (2 * d)):
            count += 1
    return count

def _score_set(s):
    """Score a set of integers (1-indexed)."""
    mask = 0
    for x in s:
        mask |= (1 << x)
    return _score_bitmask(mask)

def _random_set(n=32, lo=1, hi=512):
    return set(random.sample(range(lo, hi + 1), n))

def _greedy_build():
    """Greedy: pick elements one by one to maximize marginal score."""
    current = set()
    mask = 0
    remaining = set(range(1, 513))
    for _ in range(32):
        if time.time() > _DEADLINE:
            break
        best_x = None
        best_score = -1
        # Sample a subset of candidates to keep it fast
        candidates = list(remaining)
        random.shuffle(candidates)
        for x in candidates:
            trial_mask = mask | (1 << x)
            s = _score_bitmask(trial_mask)
            if s > best_score:
                best_score = s
                best_x = x
        if best_x is None:
            break
        current.add(best_x)
        remaining.discard(best_x)
        mask |= (1 << best_x)
    return current

def _simulated_annealing(initial, max_iters=200000):
    """SA with single-element moves (add/remove/swap)."""
    current = set(initial)
    if len(current) != 32:
        # Pad or trim to 32
        if len(current) > 32:
            current = set(list(current)[:32])
        else:
            extras = random.sample([x for x in range(1, 513) if x not in current], 32 - len(current))
            current.update(extras)
    
    current_mask = 0
    for x in current:
        current_mask |= (1 << x)
    current_score = _score_bitmask(current_mask)
    best_set = set(current)
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    cooling = (T_min / T) ** (1.0 / max(1, max_iters))
    
    it = 0
    while it < max_iters:
        if time.time() > _DEADLINE:
            break
        # Every ~1000 iters, re-cool if stuck
        it += 1
        
        x = random.randint(1, 512)
        if x in current:
            # Remove
            trial_mask = current_mask & ~(1 << x)
            if trial_mask.bit_count() != 32:
                continue
            trial_score = _score_bitmask(trial_mask)
            delta = trial_score - current_score
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                current.discard(x)
                current_mask = trial_mask
                current_score = trial_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current)
        else:
            # Add: need to remove one
            to_remove = random.choice(list(current))
            trial_mask = (current_mask | (1 << x)) & ~(1 << to_remove)
            if trial_mask.bit_count() != 32:
                continue
            trial_score = _score_bitmask(trial_mask)
            delta = trial_score - current_score
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                current.discard(to_remove)
                current.add(x)
                current_mask = trial_mask
                current_score = trial_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current)
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat occasionally
            if it % 50000 == 0:
                T = 30.0
    return best_set

def construct():
    start = time.time()
    deadline = start + 145.0  # 5s safety margin
    
    random.seed(42)
    
    # Start with a greedy construction
    best_set = _greedy_build()
    best_score = _score_set(best_set)
    
    # Run SA from multiple starting points
    attempts = 0
    while time.time() < deadline - 5.0:
        if attempts == 0:
            init = best_set
        else:
            init = _random_set(32)
            # Or perturb best
            if random.random() < 0.5:
                init = set(best_set)
                for _ in range(random.randint(1, 5)):
                    if init:
                        to_remove = random.choice(list(init))
                        init.discard(to_remove)
                        extras = [x for x in range(1, 513) if x not in init]
                        if extras:
                            init.add(random.choice(extras))
        
        result = _simulated_annealing(init, max_iters=50000)
        s = _score_set(result)
        if s > best_score:
            best_score = s
            best_set = result
        attempts += 1
        
        # Check time
        if time.time() > deadline - 2.0:
            break
    
    # Final safety: ensure exactly 32 distinct ints in 1..512
    result = list(best_set)
    if len(result) != 32:
        # Fix
        if len(result) > 32:
            result = result[:32]
        else:
            existing = set(result)
            for x in range(1, 513):
                if x not in existing:
                    result.append(x)
                if len(result) == 32:
                    break
    
    # Validate
    assert len(result) == 32
    assert len(set(result)) == 32
    assert all(1 <= x <= 512 for x in result)
    
    return result

# Pre-compute a good solution at import time to ensure fast response
# Actually, let's not pre-compute to avoid import timeout; construct() handles it.
