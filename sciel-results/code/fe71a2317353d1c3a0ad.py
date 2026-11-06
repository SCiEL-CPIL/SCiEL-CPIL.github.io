"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def _reset_length(a, b):
    """Compute shortest reset word length via BFS on subsets."""
    start = tuple(range(N))
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        # apply a
        next_a = tuple(sorted(set(a[i] for i in subset)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        # apply b
        next_b = tuple(sorted(set(b[i] for i in subset)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    return -1  # not synchronizing

def _is_synchronizing(a, b):
    return _reset_length(a, b) >= 0

def _try_improvement(a, b, current_score, rng, max_evals=500):
    """Try to find an improvement by modifying transitions that preserve sync."""
    best_a, best_b = a[:], b[:]
    best_score = current_score
    
    for _ in range(max_evals):
        # Choose which letter to modify
        letter_idx = rng.random() < 0.5
        if letter_idx:
            candidate = best_a[:]
        else:
            candidate = best_b[:]
        
        # Pick a state to modify
        state = rng.randint(0, N - 1)
        old_val = candidate[state]
        
        # Try a few random targets
        for _ in range(3):
            new_val = rng.randint(0, N - 1)
            if new_val == old_val:
                continue
            candidate[state] = new_val
            
            if letter_idx:
                score = _reset_length(candidate, best_b)
            else:
                score = _reset_length(best_a, candidate)
            
            if score > best_score:
                best_score = score
                if letter_idx:
                    best_a = candidate[:]
                else:
                    best_b = candidate[:]
                break
            elif score >= 0:
                # Keep this change if it maintains synchronization
                if letter_idx:
                    best_a = candidate[:]
                else:
                    best_b = candidate[:]
                break
    
    return best_a, best_b, best_score

def construct():
    # Start with the classic Černý automaton
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    best_score = _reset_length(a, b)
    best_a, best_b = a[:], b[:]
    
    start_time = time.time()
    time_limit = 55  # leave margin
    
    rng = random.Random(42)
    
    # Phase 1: Structured local search
    while time.time() - start_time < time_limit - 10:
        new_a, new_b, new_score = _try_improvement(best_a, best_b, best_score, rng, max_evals=200)
        if new_score > best_score:
            best_score = new_score
            best_a, best_b = new_a, new_b
    
    # Phase 2: More aggressive random search with occasional restarts
    while time.time() - start_time < time_limit:
        a = best_a[:]
        b = best_b[:]
        # Make 1-3 random changes
        num_changes = rng.randint(1, 3)
        for _ in range(num_changes):
            letter = rng.random() < 0.5
            state = rng.randint(0, N - 1)
            if letter:
                a[state] = rng.randint(0, N - 1)
            else:
                b[state] = rng.randint(0, N - 1)
        
        score = _reset_length(a, b)
        if score > best_score:
            best_score = score
            best_a, best_b = a[:], b[:]
    
    return best_a, best_b
# EVOLVE-BLOCK-END
