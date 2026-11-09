import time
from fractions import Fraction
from heapq import heappush, heappop
import itertools

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def check_time():
        return time.time() > deadline

    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248

    # Verify
    s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
    assert s == TARGET, f"Known solution invalid: {s}"

    # Dijkstra-like search in the graph of rational residues
    # State: (remaining_fraction, min_denom)
    # min_denom ensures we pick denominators in increasing order (distinctness)
    # Priority: maximum denominator used so far
    # We track: (max_denom_so_far, remaining, min_denom, path)
    
    # Priority queue: (max_denom_so_far, counter, remaining, min_denom, path)
    # We stop when we pop a state with remaining = 0
    
    counter = itertools.count()
    
    pq = []
    initial_remaining = TARGET
    initial_min_denom = 1
    heappush(pq, (0, next(counter), initial_remaining, initial_min_denom, []))
    
    # Track the best max_denom seen for each (remaining, min_denom) state
    best_seen = {}
    
    found = False
    best_path = None
    best_max_found = best_max
    
    # Limit the number of states to prevent memory explosion
    max_states = 500000
    states_explored = 0
    
    while pq and not check_time() and states_explored < max_states:
        max_denom_so_far, _, remaining, min_denom, path = heappop(pq)
        states_explored += 1
        
        # If we've found a better solution, we can stop
        if remaining == 0:
            if max_denom_so_far < best_max_found:
                best_max_found = max_denom_so_far
                best_path = path
                found = True
            break
        
        # If max_denom_so_far >= best_max_found, no need to explore further
        if max_denom_so_far >= best_max_found:
            continue
        
        # Check if we've seen a better or equal state
        state_key = (remaining, min_denom)
        if state_key in best_seen:
            if best_seen[state_key] <= max_denom_so_far:
                continue
        else:
            best_seen[state_key] = max_denom_so_far
        
        # Determine the range of denominators to try
        if remaining <= 0:
            continue
            
        # Lower bound for d: max(min_denom, ceil(1/remaining))
        lower_d = max(min_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Upper bound for d: we want max < best_max_found
        upper_d = best_max_found - 1
        
        if lower_d > upper_d:
            continue
        
        # Limit path length
        if len(path) >= MAX_TERMS:
            continue
        
        for d in range(lower_d, upper_d + 1):
            if check_time():
                break
                
            new_remaining = remaining - Fraction(1, d)
            
            if new_remaining < 0:
                break
            
            if new_remaining == 0:
                new_max = max(max_denom_so_far, d)
                if new_max < best_max_found:
                    best_max_found = new_max
                    best_path = path + [d]
                    found = True
                    break
            
            if new_remaining > 0:
                new_max = max(max_denom_so_far, d)
                if new_max >= best_max_found:
                    continue
                    
                new_state_key = (new_remaining, d + 1)
                if new_state_key in best_seen:
                    if best_seen[new_state_key] <= new_max:
                        continue
                
                heappush(pq, (new_max, next(counter), new_remaining, d + 1, path + [d]))
        
        if found:
            break
    
    if found and best_path is not None:
        best_solution = best_path
        best_max = best_max_found
    
    return best_solution
