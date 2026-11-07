"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _can_add(a, current_list):
    """Check if a can be added: for all distinct b,c in current_list, a does not divide b*c."""
    n = len(current_list)
    for i in range(n):
        b = current_list[i]
        for j in range(i + 1, n):
            c = current_list[j]
            if (b * c) % a == 0:
                return False
    return True

def _greedy_construct(order, deadline):
    current_list = []
    for a in order:
        if time.time() > deadline:
            break
        if _can_add(a, current_list):
            current_list.append(a)
    return current_list

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Precompute prime factors
    pf_cache = {}
    for a in range(2, 257):
        pf_cache[a] = _prime_factors(a)
    
    # Safety score: prefer numbers with large unique prime factors
    def safety_score(a):
        pf = pf_cache[a]
        max_pf = max(pf) if pf else 0
        return (-max_pf, len(pf), a)
    
    base_order = sorted(range(2, 257), key=safety_score)
    
    # Deterministic greedy first
    best = _greedy_construct(base_order, deadline)
    best_size = len(best)
    
    # Randomized restarts
    rng = random.Random(12345)
    while time.time() < deadline - 3:
        # Randomized order with bias toward high-safety numbers
        all_nums = list(range(2, 257))
        
        # Weighted shuffle: higher safety = more likely to come earlier
        # Use exponential weights based on rank
        ranked = sorted(range(2, 257), key=safety_score)
        weights = [1.0 / (i + 1) for i in range(len(ranked))]
        
        # Partial selection: pick first few from top, shuffle rest
        top_n = 20 + rng.randint(0, 20)
        top = ranked[:top_n]
        rest = ranked[top_n:]
        rng.shuffle(top)
        rng.shuffle(rest)
        order = top + rest
        
        result = _greedy_construct(order, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Validate
    best_set = set(best)
    assert len(best_set) == len(best)
    for a in best:
        assert isinstance(a, int) and not isinstance(a, bool)
        assert 2 <= a <= 256
    
    return list(best)
# EVOLVE-BLOCK-END
