import time
import random
import numpy as np

def construct():
    N = 1000
    max_val = 1000
    total_deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)  # in_S[i] is True if i is in S
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    # Max sum is 2000
    counts = [0] * (2 * max_val + 1)
    
    # Helper to add a number to the set and update counts
    def try_add(x):
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    def try_remove(x):
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    def is_valid_swap(a, c):
        deltas = {}
        deltas[2*a] = deltas.get(2*a, 0) - 1
        for y in S:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
        
        deltas[2*c] = deltas.get(2*c, 0) + 1
        for y in S:
            if y != a:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
        
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False
        return True

    def is_valid_2swap(a, b, c, d):
        if c == d or in_S[c] or in_S[d] or a == b:
            return False

        deltas = {}
        deltas[2*a] = deltas.get(2*a, 0) - 1
        deltas[2*b] = deltas.get(2*b, 0) - 1
        if a != b:
            s_ab = a + b
            deltas[s_ab] = deltas.get(s_ab, 0) - 1
            
        for y in S:
            if y != a and y != b:
                s_ay = a + y
                s_by = b + y
                deltas[s_ay] = deltas.get(s_ay, 0) - 1
                deltas[s_by] = deltas.get(s_by, 0) - 1
        
        deltas[2*c] = deltas.get(2*c, 0) + 1
        deltas[2*d] = deltas.get(2*d, 0) + 1
        if c != d:
            s_cd = c + d
            deltas[s_cd] = deltas.get(s_cd, 0) + 1
            
        for y in S:
            if y != a and y != b:
                s_cy = c + y
                s_dy = d + y
                deltas[s_cy] = deltas.get(s_cy, 0) + 1
                deltas[s_dy] = deltas.get(s_dy, 0) + 1
        
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False
        return True

    def do_2swap(a, b, c, d):
        S.remove(a)
        in_S[a] = False
        S.remove(b)
        in_S[b] = False
        
        counts[2*a] -= 1
        counts[2*b] -= 1
        if a != b:
            counts[a+b] -= 1
            
        for y in S:
            if y != a and y != b:
                counts[a+y] -= 1
                counts[b+y] -= 1
                
        S.append(c)
        in_S[c] = True
        S.append(d)
        in_S[d] = True
        
        counts[2*c] += 1
        counts[2*d] += 1
        if c != d:
            counts[c+d] += 1
            
        for y in S:
            if y != c and y != d:
                counts[c+y] += 1
                counts[d+y] += 1

    def reset_state():
        S.clear()
        for i in range(max_val + 1):
            in_S[i] = False
        for i in range(2 * max_val + 1):
            counts[i] = 0

    def greedy_init(mode='random'):
        reset_state()
        candidates = list(range(1, max_val + 1))
        if mode == 'random':
            random.shuffle(candidates)
        elif mode == 'small':
            pass
        elif mode == 'large':
            candidates.reverse()
        
        for i in candidates:
            try_add(i)
        return list(S)

    def targeted_add_pass():
        """Systematically try to add each candidate not in S. Returns number added."""
        added = 0
        for c in range(1, max_val + 1):
            if in_S[c]:
                continue
            if try_add(c):
                added += 1
        return added

    # Track best solution
    best_S = []
    best_size = 0
    
    # Multi-start Local Search
    all_numbers = list(range(1, max_val + 1))
    
    while time.time() < total_deadline:
        # 1. Initialization
        mode = random.choice(['random', 'small', 'random'])
        init_S = greedy_init(mode)
        
        if len(init_S) > best_size:
            best_size = len(init_S)
            best_S = list(init_S)
        
        if len(S) == 0:
            continue
            
        # 2. Local Search
        T = 1.0
        cooling_rate = 0.9995
        no_improve_count = 0
        local_best_size = len(S)
        local_best_S = list(S)
        
        # Local search loop with time limit per restart
        restart_deadline = time.time() + 5.0
        
        while time.time() < restart_deadline and time.time() < total_deadline:
            move_type = random.random()
            
            if move_type < 0.35:
                # Targeted add: systematically try all candidates
                added = targeted_add_pass()
                if added > 0:
                    if len(S) > local_best_size:
                        local_best_size = len(S)
                        local_best_S = list(S)
                        no_improve_count = 0
                    T = max(0.1, T * cooling_rate)
                else:
                    no_improve_count += 1
            elif move_type < 0.55:
                # Random Add
                c = random.choice(all_numbers)
                if in_S[c]:
                    continue
                if try_add(c):
                    if len(S) > local_best_size:
                        local_best_size = len(S)
                        local_best_S = list(S)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
                else:
                    no_improve_count += 1
            elif move_type < 0.75:
                # Swap
                if not S:
                    continue
                a = random.choice(S)
                c = random.choice(all_numbers)
                if in_S[c]:
                    continue
                if is_valid_swap(a, c):
                    try_remove(a)
                    try_add(c)
                    no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
                else:
                    no_improve_count += 1
            elif move_type < 0.95:
                # 2-Swap
                if len(S) < 2:
                    continue
                a, b = random.sample(S, 2)
                c, d = random.sample(all_numbers, 2)
                if in_S[c] or in_S[d]:
                    continue
                if is_valid_2swap(a, b, c, d):
                    do_2swap(a, b, c, d)
                    no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
                else:
                    no_improve_count += 1
            else:
                # Remove
                if S:
                    a = random.choice(S)
                    try_remove(a)
                    no_improve_count += 1
                    T = max(0.1, T * cooling_rate)
            
            # Restart local search if stuck for too long
            if no_improve_count > 300:
                break
        
        # After local search, do a final targeted add pass
        added = targeted_add_pass()
        if len(S) > local_best_size:
            local_best_size = len(S)
            local_best_S = list(S)
        
        # Update global best
        if local_best_size > best_size:
            best_size = local_best_size
            best_S = local_best_S

    return best_S
