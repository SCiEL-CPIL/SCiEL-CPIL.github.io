"""B2[2] set construction via local search."""
import time
import random
import numpy as np

_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _deadline_ok(margin=10.0):
    return _elapsed() < 150.0 - margin

def _count_reps(chosen):
    """Count representations a+b with a<=b for each sum."""
    n = len(chosen)
    counts = {}
    for i in range(n):
        a = chosen[i]
        for j in range(i, n):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    return counts

def _excess(chosen):
    """Total number of representations exceeding 2."""
    counts = _count_reps(chosen)
    total = 0
    for s, c in counts.items():
        if c > 2:
            total += c - 2
    return total

def _fitness(chosen):
    """Higher is better: size minus penalty for excess reps."""
    n = len(chosen)
    exc = _excess(chosen)
    return n - 100.0 * exc

def _greedy_init():
    """Greedy construction as baseline."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        ok = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            counts = trial
    return chosen

def _local_search(initial, time_limit=120.0):
    """Local search with swaps and insertions/deletions."""
    chosen = sorted(initial)
    best = list(chosen)
    best_fit = _fitness(best)
    
    rng = random.Random(42)
    start_time = time.time()
    
    # Precompute for speed
    def try_swap(i, j):
        """Swap element i with element j (j is a candidate from 1..1000 not in set)."""
        pass
    
    # Use a different approach: try replacing one element with another
    def evaluate_set(s):
        return _fitness(s)
    
    current = list(best)
    current_fit = best_fit
    
    # Simulated annealing style
    temp = 10.0
    cooling = 0.999
    
    while _elapsed() < time_limit - 10:
        # Pick a random operation
        op = rng.random()
        
        if op < 0.5 and len(current) > 0:
            # Replace a random element with a random other element
            i = rng.randrange(len(current))
            old = current[i]
            # Try a few candidates
            for attempt in range(20):
                new_val = rng.randint(1, 1000)
                if new_val == old or new_val in current:
                    continue
                trial = list(current)
                trial[i] = new_val
                trial.sort()
                f = _fitness(trial)
                if f >= current_fit or rng.random() < np.exp((f - current_fit) / max(temp, 0.01)):
                    current = trial
                    current_fit = f
                    if f > best_fit:
                        best = list(current)
                        best_fit = f
                    break
        elif op < 0.8 and len(current) < 1000:
            # Try to insert a new element
            for attempt in range(20):
                new_val = rng.randint(1, 1000)
                if new_val in current:
                    continue
                trial = list(current)
                trial.append(new_val)
                trial.sort()
                f = _fitness(trial)
                if f >= current_fit or rng.random() < np.exp((f - current_fit) / max(temp, 0.01)):
                    current = trial
                    current_fit = f
                    if f > best_fit:
                        best = list(current)
                        best_fit = f
                    break
        else:
            # Try to delete an element (rarely helpful but helps escape)
            if len(current) > 1:
                i = rng.randrange(len(current))
                trial = list(current)
                del trial[i]
                f = _fitness(trial)
                if f >= current_fit or rng.random() < np.exp((f - current_fit) / max(temp, 0.01)):
                    current = trial
                    current_fit = f
                    if f > best_fit:
                        best = list(current)
                        best_fit = f
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best

def construct():
    global _START
    _START = time.time()
    
    # Start with greedy
    initial = _greedy_init()
    
    # Local search
    result = _local_search(initial, time_limit=130.0)
    
    # Ensure validity
    counts = _count_reps(result)
    valid = all(c <= 2 for c in counts.values())
    
    if not valid:
        # Fallback to greedy
        result = initial
    
    return result
