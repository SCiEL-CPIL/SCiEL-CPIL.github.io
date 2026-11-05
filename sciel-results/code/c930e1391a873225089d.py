"""Rewritten program: find a 32-element set in 1..512 maximizing distinct 3-AP differences."""
import time
import random
from itertools import combinations

def _score(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in set s."""
    sset = set(s)
    diffs = set()
    for a in s:
        for b in s:
            if b == a:
                continue
            d = b - a
            if d <= 0:
                continue
            if a + 2 * d in sset:
                diffs.add(d)
    return len(diffs)

def _local_search(s, deadline, max_iter=5000):
    """Greedy local search: try swapping one element with another to improve score."""
    best = list(s)
    best_score = _score(best)
    n = len(best)
    best_set = set(best)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing one element and adding a new one
        for i in range(n):
            removed = best[i]
            if removed in best_set:
                best_set.remove(removed)
            
            # Try adding each possible new element not already in set
            # To speed up, only try elements near the removed one or random samples
            candidates = []
            # Sample some candidates
            for _ in range(20):
                c = random.randint(1, 512)
                if c not in best_set:
                    candidates.append(c)
            # Also try neighbors
            for delta in range(-20, 21):
                c = removed + delta
                if 1 <= c <= 512 and c not in best_set:
                    candidates.append(c)
            
            for c in candidates:
                if c in best_set:
                    continue
                best_set.add(c)
                new_s = list(best)
                new_s[i] = c
                ns = _score(new_s)
                if ns > best_score:
                    best_score = ns
                    best = new_s
                    improved = True
                    break
                best_set.remove(c)
            
            if improved:
                break
            best_set.add(removed)
        
        if not improved:
            # Stuck in local optimum, try random restart within neighborhood
            # Or break if no improvement
            break
    
    return best, best_score

def _simulated_annealing(s, deadline, max_iter=10000):
    """Simple simulated annealing to escape local optima."""
    best = list(s)
    best_score = _score(best)
    current = list(best)
    current_score = best_score
    n = len(current)
    current_set = set(current)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        if T < T_min:
            break
        
        # Generate a neighbor: swap one element with a random new one
        i = random.randint(0, n - 1)
        removed = current[i]
        current_set.remove(removed)
        
        # Pick a random new element
        new_elem = random.randint(1, 512)
        attempts = 0
        while new_elem in current_set and attempts < 100:
            new_elem = random.randint(1, 512)
            attempts += 1
        
        if new_elem in current_set:
            current_set.add(removed)
            continue
        
        current_set.add(new_elem)
        new_s = list(current)
        new_s[i] = new_elem
        new_score = _score(new_s)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current = new_s
            current_score = new_score
            current_set = set(current)
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        else:
            current_set.remove(new_elem)
            current_set.add(removed)
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with a good initial set: smallest 32 numbers (dense)
    initial = list(range(1, 33))
    best = list(initial)
    best_score = _score(best)
    
    # Try several random initial sets
    random.seed(42)
    
    # Also try 5-smooth based initial set
    smooth5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90]
    if len(smooth5) >= 32:
        s = list(smooth5[:32])
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Run local search from the best initial
    if time.time() < deadline - 10:
        s, sc = _local_search(best, deadline - 10, max_iter=3000)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Run simulated annealing from the best
    if time.time() < deadline - 5:
        s, sc = _simulated_annealing(best, deadline - 5, max_iter=20000)
        if sc > best_score:
            best_score = sc
            best = s
    
    # Try a few more random restarts if time permits
    for attempt in range(10):
        if time.time() > deadline - 2:
            break
        
        # Random initial set
        random_set = random.sample(range(1, 513), 32)
        rs = _score(random_set)
        
        if rs > best_score or rs >= best_score - 5:
            # Run local search from this random start
            deadline_ls = deadline - 1
            s, sc = _local_search(random_set, deadline_ls, max_iter=2000)
            if sc > best_score:
                best_score = sc
                best = s
    
    # Final validation
    assert len(best) == 32
    assert len(set(best)) == 32
    assert all(1 <= x <= 512 for x in best)
    
    return best

# EVOLVE-BLOCK-START
# The actual construct() is defined above; keep the block marker for compatibility
# EVOLVE-BLOCK-END
