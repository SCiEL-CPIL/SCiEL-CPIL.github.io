import time
import random

def construct():
    N = 128
    # Start with the known good solution
    S = set(range(N//2 + 1, N + 1))
    current_size = len(S)
    
    # Precompute sums? No, just use bitset or set lookup.
    # Set lookup is O(1).
    
    def is_valid(S):
        # Check for any a, b, c in S distinct such that a+b, a+c, b+c in S
        # Optimization: iterate over pairs a, b.
        # If a+b in S, then check if there exists c in S such that a+c in S and b+c in S.
        # We can iterate c.
        items = list(S)
        for i in range(len(items)):
            a = items[i]
            for j in range(i+1, len(items)):
                b = items[j]
                if a + b in S:
                    # Found a pair whose sum is in S.
                    # Now check if there is a third element c
                    # such that a+c in S and b+c in S.
                    for c in items:
                        if c == a or c == b:
                            continue
                        if a + c in S and b + c in S:
                            return False
        return True

    # Local Search
    start_time = time.time()
    time_limit = 50 # seconds
    best_S = S.copy()
    best_size = current_size
    
    # Randomize initial state slightly? Or just start from the good one.
    # Let's try to add small numbers.
    candidates = list(range(1, N//2 + 1))
    random.shuffle(candidates)
    
    current_S = S.copy()
    
    while time.time() - start_time < time_limit:
        # Try to add a random candidate
        if candidates:
            x = random.choice(candidates)
            if x not in current_S:
                new_S = current_S.copy()
                new_S.add(x)
                if is_valid(new_S):
                    current_S = new_S
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_S = current_S.copy()
                        # Remove from candidates if added? 
                        # We can re-add later if we remove it.
                        try:
                            candidates.remove(x)
                        except ValueError:
                            pass
        
        # Try to remove a random element from current_S to allow adding others?
        # Or just swap.
        # Let's try swapping: remove one, add another.
        if len(current_S) > 0 and candidates:
            x_to_remove = random.choice(list(current_S))
            x_to_add = random.choice(candidates)
            if x_to_add not in current_S:
                new_S = current_S.copy()
                new_S.remove(x_to_remove)
                new_S.add(x_to_add)
                if is_valid(new_S):
                    # Accept if size is same or larger? 
                    # If size is same, we are just moving around.
                    # If we remove from large and add to small, size is same.
                    # We want to maximize size.
                    if len(new_S) >= len(current_S):
                         # Only accept if it leads to a better state or is neutral.
                         # To escape local minima, accept neutral moves with prob 0.5?
                         # For now, accept if size is strictly larger, or with prob 0.1 if same.
                         if len(new_S) > len(current_S) or random.random() < 0.1:
                             current_S = new_S
                             if len(current_S) > best_size:
                                 best_size = len(current_S)
                                 best_S = current_S.copy()
        
        # Occasionally, try to add multiple small numbers at once?
        # Or just restart with a different random seed?
        
    # Final check
    if not is_valid(best_S):
        # Fallback to safe set
        return list(range(N//2 + 1, N + 1))
        
    return list(best_S)
