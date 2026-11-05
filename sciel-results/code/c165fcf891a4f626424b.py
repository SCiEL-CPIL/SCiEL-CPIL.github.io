import random
import time
import math
import numpy as np

N = 64

def construct():
    # Precompute all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_valid_addition(p, current_set):
        # Same as before, O(K^2)
        d_to_s = {}
        for q in current_set:
            d = dist_sq(p, q)
            if d in d_to_s:
                return False
            d_to_s[d] = q
        for q in current_set:
            d_pq = dist_sq(p, q)
            for r in current_set:
                if r == q:
                    continue
                if dist_sq(q, r) == d_pq:
                    return False
        return True

    # 1. Greedy Initialization
    random.seed(42)
    shuffled = all_points[:]
    random.shuffle(shuffled)
    current_set = []
    for p in shuffled:
        if is_valid_addition(p, current_set):
            current_set.append(p)
            
    best_set = current_set[:]
    best_size = len(current_set)
    
    # 2. Simulated Annealing
    start_time = time.time()
    timeout = 290
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    # We need a set for O(1) lookup of points in current_set
    # But current_set is a list. We can maintain a set for membership.
    current_set_set = set(current_set)
    
    while time.time() - start_time < timeout and T > min_T:
        # Generate a move: Swap a point in set with a point outside
        # Or Add/Remove
        
        # To keep size similar, let's do swaps primarily.
        # Pick random index in current_set
        if current_set:
            idx = random.randint(0, len(current_set) - 1)
            q_out = current_set[idx]
            
            # Pick random point from all_points not in current_set
            # This is O(N^2) to find if we scan. 
            # Better: Keep a list of points not in set? 
            # Or just pick random from all_points and check if in set.
            # If in set, retry or pick another.
            
            p_in = None
            # Try a few times to find a point outside
            for _ in range(10):
                cand = random.choice(all_points)
                if cand not in current_set_set:
                    p_in = cand
                    break
            
            if p_in is None:
                continue
            
            # Perform swap
            # Remove q_out
            current_set.pop(idx)
            current_set_set.remove(q_out)
            
            # Check if p_in can be added
            if is_valid_addition(p_in, current_set):
                # Valid move
                current_set.append(p_in)
                current_set_set.add(p_in)
                new_size = len(current_set)
                
                if new_size > best_size:
                    best_size = new_size
                    best_set = current_set[:]
                
                # Acceptance: Always accept if size increases?
                # In SA, we usually minimize energy. Here we maximize size.
                # If size is same, we might accept to explore.
                # If size decreases, we accept with prob exp(-delta/T).
                
                # Since we only accept valid additions, size never decreases in this "swap" logic 
                # unless we explicitly allow removals.
                # Actually, if is_valid_addition fails, we revert.
                # So this is effectively Hill Climbing with random swaps.
                
                # To make it true SA, we should allow "bad" moves (smaller sets) to escape local minima.
                # But "smaller set" is not a valid state if we define state as "valid set".
                # Usually, in packing problems, we define the state as the set of points, and the "cost" is the number of violations.
                # Here, the constraint is hard. We cannot have violations.
                # So we are restricted to the space of valid sets.
                # In this space, we can only move to adjacent valid sets.
                # A move is valid if the new set is valid.
                # If we swap q with p, and the new set is valid, we accept it.
                # This is still Hill Climbing if we always accept improvements.
                # To do SA, we need to accept non-improvements (same size) with some probability, 
                # or allow "removal" moves that reduce size, and then "addition" moves that increase it.
                
                # Let's allow size changes.
                # If we remove a point, size decreases.
                # If we add a point, size increases.
                
                # Let's modify the move generation:
                # 1. Try to add a random point (if valid).
                # 2. Try to remove a random point (always valid).
                # 3. Try to swap.
                
                # If we remove a point, we go to a smaller set.
                # We should accept this with probability exp(-1/T) (since delta = -1).
                # Then from the smaller set, we might find a way to add 2 points later?
                
                # Let's implement full SA on the space of valid sets.
                # Energy E = -len(set).
                # Move:
                #   a) Add random p: if valid, new_E = E - 1. Delta = -1. Accept with prob 1 (improvement).
                #   b) Remove random q: new_E = E + 1. Delta = 1. Accept with prob exp(-1/T).
                #   c) Swap q, p: if valid, new_E = E. Delta = 0. Accept with prob 1 (or random).
                
                # This allows exploration.
            else:
                # Revert
                current_set.append(q_out)
                current_set_set.add(q_out)
        else:
            # Set is empty, try to add
            p_in = random.choice(all_points)
            if is_valid_addition(p_in, current_set):
                current_set.append(p_in)
                current_set_set.add(p_in)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]

        T *= cooling_rate

    return best_set
