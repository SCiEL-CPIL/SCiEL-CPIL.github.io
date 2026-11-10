import numpy as np
import random
import math

N = 16

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(p, a, b):
    d1 = _dist2(p, a)
    d2 = _dist2(p, b)
    d3 = _dist2(a, b)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_no_isosceles(S):
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            for k in range(j+1, len(S_list)):
                if _is_isosceles(S_list[i], S_list[j], S_list[k]):
                    return False
    return True

def _is_maximal(S, all_points):
    S_set = set(S)
    for p in all_points:
        if p in S_set:
            continue
        # Check if S U {p} is no-isosceles
        # This is equivalent to checking if p forms an isosceles triangle with any pair in S
        # If p does NOT form an isosceles triangle with any pair in S, then p can be added.
        # If p can be added, S is NOT maximal.
        can_add = True
        S_list = list(S_set)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if _is_isosceles(p, S_list[i], S_list[j]):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def _fitness(S, all_points):
    # Returns (is_valid, size)
    # We want to minimize size, subject to being no-isosceles and maximal.
    if not _is_no_isosceles(S):
        return (False, float('inf'))
    if not _is_maximal(S, all_points):
        # Penalize non-maximal sets to keep search in valid region
        # But we can also just repair them.
        # Let's return a high cost.
        return (False, float('inf'))
    return (True, len(S))

def construct():
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Initial solution: Greedy maximal set
    random.seed(42)
    points_shuffled = all_points[:]
    random.shuffle(points_shuffled)
    S = set()
    for p in points_shuffled:
        # Check if p can be added to S
        can_add = True
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if _is_isosceles(p, S_list[i], S_list[j]):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.add(p)
    
    best_S = set(S)
    best_size = len(S)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    steps = 5000
    
    current_S = set(S)
    current_valid, current_size = _fitness(current_S, all_points)
    if not current_valid:
        # Should not happen with greedy init, but just in case
        # Repair: add points until maximal
        changed = True
        while changed:
            changed = False
            for p in all_points:
                if p in current_S:
                    continue
                # Check if p can be added
                can_add = True
                S_list = list(current_S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        if _is_isosceles(p, S_list[i], S_list[j]):
                            can_add = False
                            break
                if can_add:
                    current_S.add(p)
                    changed = True
                    break
        current_valid, current_size = _fitness(current_S, all_points)
        best_S = set(current_S)
        best_size = current_size

    for step in range(steps):
        if T < min_T:
            break
            
        # Generate neighbor
        new_S = set(current_S)
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a point
            if new_S:
                p_to_remove = random.choice(list(new_S))
                new_S.remove(p_to_remove)
        elif move_type < 0.8:
            # Add a point
            candidates = [p for p in all_points if p not in new_S]
            if candidates:
                p_to_add = random.choice(candidates)
                new_S.add(p_to_add)
        else:
            # Swap
            if new_S:
                p_to_remove = random.choice(list(new_S))
                candidates = [p for p in all_points if p not in new_S]
                if candidates:
                    p_to_add = random.choice(candidates)
                    new_S.remove(p_to_remove)
                    new_S.add(p_to_add)

        new_valid, new_size = _fitness(new_S, all_points)
        
        # If new state is invalid (not maximal or not no-isosceles), 
        # we might want to repair it to a valid maximal state to compare.
        # But repairing is expensive. 
        # Alternative: Only accept valid states, or penalize heavily.
        
        # Let's only accept valid states to keep the search simple and fast.
        # This might get stuck, but combined with swaps, it should explore.
        
        if not new_valid:
            # Reject invalid states
            T *= cooling
            continue
            
        if new_size < best_size:
            best_S = set(new_S)
            best_size = new_size
            
        # Accept/Reject
        if new_size <= current_size:
            current_S = new_S
            current_size = new_size
        else:
            delta = new_size - current_size
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_S = new_S
                current_size = new_size
                
        T *= cooling
        
    return list(best_S)
