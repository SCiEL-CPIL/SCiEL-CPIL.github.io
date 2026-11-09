import random
import time
import itertools
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    L = reduce(lcm, range(1, n + 1))
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i
        
    best_abs_sum = float('inf')
    best_coeffs = [0] * n
    
    # 1. Exhaustive search for k <= 4
    for k in range(1, 5):
        if time.time() > deadline:
            break
        for indices in itertools.combinations(range(1, n + 1), k):
            for assignment in itertools.product([-1, 0, 1], repeat=k):
                if all(a == 0 for a in assignment):
                    continue
                s = 0
                for idx, a in zip(indices, assignment):
                    s += a * w[idx]
                if s != 0:
                    abs_s = abs(s)
                    if abs_s < best_abs_sum:
                        best_abs_sum = abs_s
                        new_coeffs = [0] * n
                        for idx, a in zip(indices, assignment):
                            new_coeffs[idx - 1] = a
                        best_coeffs = new_coeffs
                        if best_abs_sum == 1:
                            return best_coeffs

    # 2. Local Search
    # We will try to improve the best_coeffs found so far, or start from random
    
    def calc_sum(coeffs):
        s = 0
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * w[i + 1]
        return s
        
    # Initialize with best_coeffs or random
    current_coeffs = best_coeffs[:]
    current_sum = calc_sum(current_coeffs)
    if current_sum == 0:
        # Make it non-zero
        current_coeffs[0] = 1
        current_sum = w[1]
        
    current_abs = abs(current_sum)
    
    while time.time() < deadline:
        # Try to find a better state
        improved = False
        # Randomly shuffle the order of indices to try
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            old_c = current_coeffs[i]
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                # Calculate new sum
                # new_sum = current_sum - old_c * w[i+1] + new_c * w[i+1]
                delta = (new_c - old_c) * w[i + 1]
                new_sum = current_sum + delta
                if new_sum == 0:
                    continue
                new_abs = abs(new_sum)
                if new_abs < current_abs:
                    current_coeffs[i] = new_c
                    current_sum = new_sum
                    current_abs = new_abs
                    improved = True
                    break # Restart shuffle or continue? Continue is faster
            if improved:
                if current_abs == 1:
                    return current_coeffs
                # If improved, we might want to restart the shuffle to escape local minima
                # But for now, let's just continue the loop
                break # Break to restart the outer while loop iteration?
                # Actually, let's just continue the for loop to try more moves
                # But a greedy approach is fine.
        
        # If no improvement in a full pass, try a random perturbation
        if not improved:
            # Pick a random index and change it
            i = random.randint(0, n - 1)
            old_c = current_coeffs[i]
            choices = [c for c in [-1, 0, 1] if c != old_c]
            new_c = random.choice(choices)
            delta = (new_c - old_c) * w[i + 1]
            new_sum = current_sum + delta
            if new_sum != 0:
                current_coeffs[i] = new_c
                current_sum = new_sum
                current_abs = abs(new_sum)
            else:
                # If it became 0, revert or try another
                # Just revert for now
                pass
        
        # Update global best
        if current_abs < best_abs_sum:
            best_abs_sum = current_abs
            best_coeffs = current_coeffs[:]
            if best_abs_sum == 1:
                return best_coeffs

    return best_coeffs
