import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 128
    
    def is_valid(s):
        """
        Check if set s violates the condition:
        No three distinct members a, b, c in s such that a+b, a+c, b+c are all in s.
        """
        lst = sorted(s)
        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                a = lst[i]
                b = lst[j]
                ab = a + b
                if ab not in s:
                    continue
                for k in range(j+1, n):
                    c = lst[k]
                    ac = a + c
                    bc = b + c
                    if ac in s and bc in s:
                        return False
        return True
    
    def greedy_add(s, x):
        """Try to add x to s. Return new set if valid, else None."""
        if x in s:
            return s
        s_new = s | {x}
        if is_valid(s_new):
            return s_new
        return None
    
    def greedy_construct(order):
        """Build set by greedily adding elements in given order."""
        s = set()
        for x in order:
            s_new = greedy_add(s, x)
            if s_new is not None:
                s = s_new
        return s
    
    def local_search(s, max_steps=5000):
        """Hill climbing: try adding/removing/swapping elements."""
        s = set(s)
        current_size = len(s)
        no_improve_count = 0
        steps = 0
        
        while steps < max_steps:
            steps += 1
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # Try adding a new element
            candidates_to_add = [x for x in range(1, N+1) if x not in s]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add:
                s_new = s | {x}
                if is_valid(s_new):
                    s = s_new
                    current_size += 1
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
            
            # Try removing an element (to allow adding others later)
            if len(s) > 10:
                candidates_to_remove = list(s)
                random.shuffle(candidates_to_remove)
                for x in candidates_to_remove:
                    s_new = s - {x}
                    # After removal, try to add a few new elements
                    s_temp = set(s_new)
                    for y in range(1, N+1):
                        if y not in s_temp:
                            s_temp_new = s_temp | {y}
                            if is_valid(s_temp_new) and len(s_temp_new) > current_size:
                                s = s_temp_new
                                current_size = len(s)
                                improved = True
                                break
                    if improved:
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            
            # Random swap: remove one, try to add another
            if not improved and len(s) > 5:
                candidates_to_swap = list(s)
                random.shuffle(candidates_to_swap)
                for x in candidates_to_swap:
                    s_new = s - {x}
                    for y in range(1, N+1):
                        if y not in s_new:
                            s_temp = s_new | {y}
                            if is_valid(s_temp) and len(s_temp) > current_size:
                                s = s_temp
                                current_size = len(s)
                                improved = True
                                break
                    if improved:
                        no_improve_count = 0
                        break
        
        return s, current_size
    
    # Strategy 1: Start with upper half as baseline
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    # Try to improve via local search
    if time.time() < deadline - 5.0:
        improved_set, improved_size = local_search(best_set, max_steps=3000)
        if improved_size > best_size:
            best_set = improved_set
            best_size = improved_size
    
    # Strategy 2: Randomized greedy restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        order = list(range(1, N+1))
        random.shuffle(order)
        
        # Try different biases
        bias = random.random()
        if bias < 0.33:
            # Favor larger numbers
            order.sort(key=lambda x: -x + random.randint(0, N))
        elif bias < 0.66:
            # Favor smaller numbers
            order.sort(key=lambda x: x + random.randint(0, N))
        # else: random
        
        s = greedy_construct(order)
        
        # Local search on this result
        s, size = local_search(s, max_steps=1000)
        
        if size > best_size:
            best_set = s
            best_size = size
    
    # Strategy 3: Greedy with specific patterns
    if time.time() < deadline - 3.0:
        # Try all odd numbers
        s_odd = set(range(1, N+1, 2))
        if is_valid(s_odd) and len(s_odd) > best_size:
            best_set = s_odd
            best_size = len(best_set)
        
        # Try numbers congruent to 1 or 2 mod 3
        s_mod3 = set(x for x in range(1, N+1) if x % 3 in (1, 2))
        if is_valid(s_mod3) and len(s_mod3) > best_size:
            best_set = s_mod3
            best_size = len(best_set)
        
        # Try powers of 2 neighborhood
        s_pow = set()
        for i in range(1, 9):
            base = 2 ** i
            for j in range(base-2, base+3):
                if 1 <= j <= N:
                    s_pow.add(j)
        s_pow = set(x for x in s_pow if x <= N)
        if is_valid(s_pow) and len(s_pow) > best_size:
            best_set = s_pow
            best_size = len(best_set)
    
    # Final local search on best set
    if time.time() < deadline - 2.0 and len(best_set) > 0:
        improved_set, improved_size = local_search(best_set, max_steps=2000)
        if improved_size > best_size:
            best_set = improved_set
            best_size = improved_size
    
    # Ensure we return Python ints in a list
    result = sorted(int(x) for x in best_set)
    return result
