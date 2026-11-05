import time
import random

def construct():
    n = 256
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Start with the known valid large set
    k = 171
    S = set(range(k, n + 1))
    
    def is_valid_with(S, a):
        """Check if adding a to S maintains validity."""
        S_list = sorted(S)
        larger = [x for x in S_list if x > a]
        for i, b in enumerate(larger):
            for c in larger[i:]:
                if (b + c) % a == 0:
                    return False
        return True
    
    # Try to add numbers from 1 upward
    for a in range(1, k):
        if time.time() > deadline:
            break
        if is_valid_with(S, a):
            S.add(a)
    
    # Try descending order on a fresh copy
    S2 = set(range(k, n + 1))
    for a in range(k - 1, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_with(S2, a):
            S2.add(a)
    
    # Try ascending again on S2
    for a in range(1, k):
        if time.time() > deadline:
            break
        if is_valid_with(S2, a):
            S2.add(a)
    
    # Pick the larger set
    if len(S2) > len(S):
        S = S2
    
    # Simulated annealing local search to find larger valid sets
    def count_conflicts(S, a):
        """Count number of divisibility conflicts if a is in S."""
        S_list = sorted(S)
        larger = [x for x in S_list if x > a]
        count = 0
        for i, b in enumerate(larger):
            for c in larger[i:]:
                if (b + c) % a == 0:
                    count += 1
        return count
    
    def total_penalty(S):
        """Total penalty for all elements in S."""
        total = 0
        S_list = sorted(S)
        for a in S_list:
            total += count_conflicts(S, a)
        return total
    
    # Simulated annealing phase
    best_S = set(S)
    current_S = set(S)
    current_penalty = total_penalty(current_S)
    
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Random move: add, remove, or swap an element
        move_type = random.random()
        
        if move_type < 0.4:
            # Try adding a random element not in S
            candidates = [x for x in range(1, n + 1) if x not in current_S]
            if not candidates:
                T *= cooling
                continue
            a = random.choice(candidates)
            new_S = set(current_S)
            new_S.add(a)
            new_penalty = total_penalty(new_S)
            delta = new_penalty - current_penalty - len(new_S) + len(current_S)
            # Objective: maximize |S| - penalty, so delta_obj = (|newS| - newP) - (|curS| - curP)
            delta_obj = (len(new_S) - new_penalty) - (len(current_S) - current_penalty)
            if delta_obj >= 0 or random.random() < pow(2.718281828, delta_obj / T):
                current_S = new_S
                current_penalty = new_penalty
                if len(current_S) > len(best_S):
                    best_S = set(current_S)
        elif move_type < 0.7:
            # Try removing a random element from S
            if not current_S:
                T *= cooling
                continue
            a = random.choice(sorted(current_S))
            new_S = set(current_S)
            new_S.discard(a)
            new_penalty = total_penalty(new_S)
            delta_obj = (len(new_S) - new_penalty) - (len(current_S) - current_penalty)
            if delta_obj >= 0 or random.random() < pow(2.718281828, delta_obj / T):
                current_S = new_S
                current_penalty = new_penalty
                if len(current_S) > len(best_S):
                    best_S = set(current_S)
        else:
            # Swap: remove one, add another
            if not current_S:
                T *= cooling
                continue
            candidates = [x for x in range(1, n + 1) if x not in current_S]
            if not candidates:
                T *= cooling
                continue
            a_remove = random.choice(sorted(current_S))
            a_add = random.choice(candidates)
            new_S = set(current_S)
            new_S.discard(a_remove)
            new_S.add(a_add)
            new_penalty = total_penalty(new_S)
            delta_obj = (len(new_S) - new_penalty) - (len(current_S) - current_penalty)
            if delta_obj >= 0 or random.random() < pow(2.718281828, delta_obj / T):
                current_S = new_S
                current_penalty = new_penalty
                if len(current_S) > len(best_S):
                    best_S = set(current_S)
        
        T *= cooling
    
    # Use the best found
    if len(best_S) > len(S):
        S = best_S
    
    # Final greedy improvement pass
    for a in range(1, n + 1):
        if time.time() > deadline:
            break
        if a not in S and is_valid_with(S, a):
            S.add(a)
    
    # Final verification
    S_list = sorted(S)
    valid = True
    for i, a in enumerate(S_list):
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            for c in S_list[j:]:
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return list(range(171, 257))
    
    return S_list
