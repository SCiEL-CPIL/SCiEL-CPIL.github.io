"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _score_set(s):
    """Count distinct d>0 such that a, a+d, a+2d all in s for some a.
    Uses a set for O(1) membership."""
    ss = set(s)
    max_val = max(ss)
    min_val = min(ss)
    diffs = set()
    # For each pair (a, b) with a < b, check if a + 2*(b-a) is in ss
    # Actually, for each d, we need to check if there exists a such that
    # a, a+d, a+2d are all in ss.
    # Equivalent: for each pair (x, y) in ss with x < y, let d = y - x.
    # Then check if x + 2*d = x + 2*(y-x) = 2y - x is in ss.
    # But this only checks one specific a for each pair. We need to check
    # all a for a given d.
    # Better approach: for each d from 1 to max_val, check if there exists
    # a in ss such that a+d in ss and a+2d in ss.
    # Even better: use the fact that if a, a+d, a+2d are in ss, then
    # for the pair (a, a+d), the third element is a+2d.
    # Let's iterate over all pairs and collect the d values where the
    # 3-term AP exists.
    
    # For each element a in ss, for each element b in ss with b > a:
    #   d = b - a
    #   if a + 2*d in ss: then d is a valid difference
    # This is O(n^2) which for n=32 is 1024 checks.
    
    elements = sorted(ss)
    n = len(elements)
    for i in range(n):
        for j in range(i + 1, n):
            d = elements[j] - elements[i]
            # Check if elements[i] + 2*d is in ss
            third = elements[i] + 2 * d
            if third in ss:
                diffs.add(d)
    return len(diffs)

def _greedy_polish(s, max_iters=1000):
    """Try to improve the set by swapping elements or moving elements."""
    current_score = _score_set(s)
    ss = set(s)
    n = len(s)
    max_val = 512
    
    for _ in range(max_iters):
        improved = False
        # Try moving each element to a new position
        for i in range(n):
            old_val = s[i]
            # Try all possible new values not in ss
            for new_val in range(1, max_val + 1):
                if new_val in ss:
                    continue
                ss.remove(old_val)
                ss.add(new_val)
                new_score = _score_set(ss)
                if new_score > current_score:
                    s[i] = new_val
                    current_score = new_score
                    improved = True
                    break
                ss.remove(new_val)
                ss.add(old_val)
            if improved:
                break
        if not improved:
            break
    return s, current_score

def _simulated_annealing(s, time_limit=60.0):
    """Simulated annealing to find a better set."""
    current = list(s)
    current_score = _score_set(current)
    best = list(current)
    best_score = current_score
    
    start_time = time.time()
    temp = 10.0
    cooling = 0.999
    ss = set(current)
    max_val = 512
    n = len(current)
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor: pick a random element and move it to a random position
        i = random.randint(0, n - 1)
        old_val = current[i]
        new_val = random.randint(1, max_val)
        
        if new_val in ss:
            temp *= cooling
            continue
        
        ss.remove(old_val)
        ss.add(new_val)
        new_score = _score_set(ss)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 0.001)):
            current[i] = new_val
            current_score = new_score
            if new_score > best_score:
                best = list(current)
                best_score = new_score
        else:
            ss.remove(new_val)
            ss.add(old_val)
        
        temp *= cooling
    
    return best, best_score

def construct():
    """Construct a set of 32 distinct integers in 1..512 maximizing the number
    of distinct d>0 such that a, a+d, a+2d all lie in the set for some a."""
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with a random initial set
    random.seed(42)
    current = random.sample(range(1, 513), 32)
    current_score = _score_set(current)
    
    best = list(current)
    best_score = current_score
    
    # Try several random restarts with simulated annealing
    num_restarts = 5
    time_per_restart = 20.0
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial set
        s = random.sample(range(1, 513), 32)
        s_score = _score_set(s)
        
        # Greedy polish
        s, s_score = _greedy_polish(s, max_iters=500)
        
        if s_score > best_score:
            best = list(s)
            best_score = s_score
        
        # Simulated annealing
        remaining = min(time_per_restart, deadline - time.time() - 5.0)
        if remaining > 1.0:
            s2, s2_score = _simulated_annealing(s, time_limit=remaining)
            if s2_score > best_score:
                best = list(s2)
                best_score = s2_score
        
        # Final greedy polish on the best
        if time.time() < deadline - 5.0:
            best, best_score = _greedy_polish(best, max_iters=200)
    
    # Final polish
    if time.time() < deadline - 2.0:
        best, best_score = _greedy_polish(best, max_iters=100)
    
    return best
# EVOLVE-BLOCK-END
