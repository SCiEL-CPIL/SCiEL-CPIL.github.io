"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin
    
    # Start with the safe tail [171, 256]
    start_threshold = 171
    current_set = set(range(start_threshold, n + 1))
    best_set = set(current_set)
    best_size = len(current_set)
    
    def is_valid_addition(S, x):
        """Check if adding x to S is valid."""
        # Case 1: x is a (smallest)
        greater_than_x = [y for y in S if y > x]
        if len(greater_than_x) >= 2:
            for i in range(len(greater_than_x)):
                for j in range(i, len(greater_than_x)):
                    b = greater_than_x[i]
                    c = greater_than_x[j]
                    if b <= c and (b + c) % x == 0:
                        return False
        
        # Case 2: x is b (middle)
        less_than_x = [y for y in S if y < x]
        greater_or_equal_x = [y for y in S if y >= x]
        
        for a in less_than_x:
            for c in greater_or_equal_x:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c (largest)
        less_or_equal_x = [y for y in S if y <= x]
        
        for i in range(len(less_or_equal_x)):
            for j in range(i + 1, len(less_or_equal_x)):
                a = less_or_equal_x[i]
                b = less_or_equal_x[j]
                if a < b and (b + x) % a == 0:
                    return False
        
        return True

    # Candidate pool: numbers not in current set
    candidates = list(range(1, start_threshold))
    random.shuffle(candidates)
    
    attempts = 0
    consecutive_failures = 0
    max_consecutive_failures = 100
    
    while time.time() < deadline:
        if not candidates:
            candidates = list(range(1, start_threshold))
            random.shuffle(candidates)
        
        x = candidates.pop()
        
        if is_valid_addition(current_set, x):
            current_set.add(x)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            consecutive_failures = 0
        else:
            # Try to swap x with some y in current_set
            swapped = False
            # Try a few random swaps
            for _ in range(5):
                if not current_set:
                    break
                y = random.choice(list(current_set))
                if x == y:
                    continue
                # Check if S \ {y} U {x} is valid
                # This is equivalent to is_valid_addition(S - {y}, x)
                # We can optimize by checking conflicts, but full check is safe
                temp_set = current_set - {y}
                if is_valid_addition(temp_set, x):
                    current_set = temp_set
                    current_set.add(x)
                    swapped = True
                    break
            
            if not swapped:
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    random.shuffle(candidates)
                    consecutive_failures = 0
        
        if time.time() > deadline - 5:
            break
    
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
