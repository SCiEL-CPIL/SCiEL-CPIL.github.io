import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    N = 256
    
    def count_3ap(A):
        """Count number of 3-term APs in set A."""
        A_set = set(A)
        count = 0
        for a in A:
            for c in A:
                if a < c:
                    b = (a + c) // 2
                    if a + c == 2 * b and b in A_set and a != c:
                        count += 1
        return count // 1  # Each AP counted once since a < c
    
    def count_3ap_fast(A):
        """Faster count using numpy or optimized loops."""
        A_list = sorted(A)
        A_set = set(A_list)
        count = 0
        for i, a in enumerate(A_list):
            for c in A_list[i+1:]:
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if b in A_set and b > a and b < c:
                        count += 1
        return count
    
    def has_4ap(A, x):
        """Check if adding x to A creates a 4-term AP."""
        A_set = set(A)
        # x is 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if x + d in A_set and x + 2*d in A_set and x + 3*d in A_set:
                return True
        # x is 2nd term: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in A_set and x + d in A_set and x + 2*d in A_set:
                return True
        # x is 3rd term: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2*d in A_set and x - d in A_set and x + d in A_set:
                return True
        # x is 4th term: x-3d, x-2d, x-d, x
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3*d in A_set and x - 2*d in A_set and x - d in A_set:
                return True
        return False
    
    def gain_3ap(A, x):
        """Count how many new 3-APs are created by adding x to A."""
        A_set = set(A)
        gain = 0
        # x is 1st term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A_set and x + 2*d in A_set:
                gain += 1
        # x is 2nd term: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A_set and x + d in A_set:
                gain += 1
        # x is 3rd term: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if x - 2*d in A_set and x - d in A_set:
                gain += 1
        return gain
    
    # Strategy: Use a combination of greedy construction with local search
    # and multiple restarts with different random seeds
    
    best_A = None
    best_score = -1
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # First restart: pure greedy
            A = set()
            remaining = set(range(1, N + 1))
            
            while remaining and time.time() < deadline - 10:
                best_x = -1
                best_gain = -1
                for x in remaining:
                    if not has_4ap(A, x):
                        g = gain_3ap(A, x)
                        if g > best_gain or (g == best_gain and random.random() < 0.1):
                            best_gain = g
                            best_x = x
                if best_x == -1:
                    break
                A.add(best_x)
                remaining.remove(best_x)
        else:
            # Subsequent restarts: start with random subset, then greedily add
            A = set()
            # Start with a random sparse set
            num_initial = random.randint(10, 30)
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates[:num_initial]:
                if not has_4ap(A, x):
                    A.add(x)
            
            remaining = set(range(1, N + 1)) - A
            
            # Now greedily add elements
            while remaining and time.time() < deadline - 10:
                best_x = -1
                best_gain = -1
                for x in remaining:
                    if not has_4ap(A, x):
                        g = gain_3ap(A, x)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                if best_x == -1:
                    break
                A.add(best_x)
                remaining.remove(best_x)
        
        # Local search: try swaps
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                current_score = count_3ap_fast(A)
                A_list = list(A)
                random.shuffle(A_list)
                
                for a in A_list:
                    if time.time() > deadline - 3:
                        break
                    # Try removing a and adding b
                    A_temp = A - {a}
                    remaining = set(range(1, N + 1)) - A_temp
                    
                    best_b = -1
                    best_new_score = current_score
                    
                    # Sample some candidates for b
                    remaining_list = list(remaining)
                    random.shuffle(remaining_list)
                    for b in remaining_list[:50]:
                        if not has_4ap(A_temp, b):
                            A_try = A_temp | {b}
                            new_score = count_3ap_fast(A_try)
                            if new_score > best_new_score:
                                best_new_score = new_score
                                best_b = b
                    
                    if best_b != -1 and best_new_score > current_score:
                        A = A_temp | {best_b}
                        improved = True
        
        # Evaluate
        score = count_3ap_fast(A)
        if score > best_score:
            best_score = score
            best_A = A.copy()
    
    if best_A is None:
        # Fallback: simple greedy
        A = set()
        for x in range(1, N + 1):
            if not has_4ap(A, x):
                A.add(x)
        best_A = A
    
    return sorted(best_A)
