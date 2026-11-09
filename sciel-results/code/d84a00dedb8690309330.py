import random
import math

def construct():
    best_set = []
    best_score = -1
    
    # Helper to calculate penalty and update counts
    # We will manage counts and penalty externally for speed
    
    for restart in range(10):
        # 1. Greedy Initialization
        current_set = []
        counts = [0] * 2001
        total_penalty = 0
        
        def get_penalty_change_add(x, current_set, counts):
            """
            Returns delta_penalty if x is added.
            Does NOT modify counts.
            """
            delta_pen = 0
            # 2*x
            s = 2 * x
            old_c = counts[s]
            new_c = old_c + 1
            delta_pen += max(0, new_c - 2) - max(0, old_c - 2)
            
            # x + y
            for y in current_set:
                s = x + y
                old_c = counts[s]
                new_c = old_c + 1
                delta_pen += max(0, new_c - 2) - max(0, old_c - 2)
            return delta_pen

        def get_penalty_change_remove(x, current_set, counts):
            """
            Returns delta_penalty if x is removed.
            Does NOT modify counts.
            Assumes x is in current_set.
            """
            delta_pen = 0
            # 2*x
            s = 2 * x
            old_c = counts[s]
            new_c = old_c - 1
            delta_pen += max(0, new_c - 2) - max(0, old_c - 2)
            
            # x + y
            for y in current_set:
                if y == x:
                    continue
                s = x + y
                old_c = counts[s]
                new_c = old_c - 1
                delta_pen += max(0, new_c - 2) - max(0, old_c - 2)
            return delta_pen

        def do_add(x, current_set, counts):
            current_set.append(x)
            s = 2 * x
            counts[s] += 1
            for y in current_set[:-1]:
                counts[x + y] += 1

        def do_remove(x, current_set, counts):
            current_set.remove(x)
            s = 2 * x
            counts[s] -= 1
            for y in current_set:
                counts[x + y] -= 1

        # Greedy fill
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        for x in candidates:
            dp = get_penalty_change_add(x, current_set, counts)
            if total_penalty + dp == 0: # Only add if it keeps valid
                do_add(x, current_set, counts)
                # total_penalty remains 0

        # 2. Simulated Annealing
        # Objective: Maximize size, minimize penalty.
        # Score = size - 1000 * penalty
        current_score = len(current_set) - 1000 * total_penalty
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]

        T = 10.0
        steps = 5000
        
        for step in range(steps):
            # Generate move
            r = random.random()
            if r < 0.5:
                # Add a random element not in set
                # To make it faster, pick from a small random subset of candidates?
                # Or just pick 1..1000 and check if in set.
                # Using a set for membership check is O(1).
                # But we have a list. Let's maintain a set for fast lookup?
                # For now, O(N) check is okay for 1000.
                x = random.randint(1, 1000)
                if x in current_set:
                    continue
                
                dp = get_penalty_change_add(x, current_set, counts)
                new_penalty = total_penalty + dp
                new_size = len(current_set) + 1
                new_score = new_size - 1000 * new_penalty
                delta_score = new_score - current_score
                
                if delta_score >= 0 or (T > 0.01 and random.random() < math.exp(delta_score / T)):
                    do_add(x, current_set, counts)
                    total_penalty = new_penalty
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
                        
            else:
                # Remove a random element from set
                if not current_set:
                    continue
                idx = random.randint(0, len(current_set) - 1)
                x = current_set[idx]
                
                dp = get_penalty_change_remove(x, current_set, counts)
                new_penalty = total_penalty + dp
                new_size = len(current_set) - 1
                new_score = new_size - 1000 * new_penalty
                delta_score = new_score - current_score
                
                if delta_score >= 0 or (T > 0.01 and random.random() < math.exp(delta_score / T)):
                    do_remove(x, current_set, counts)
                    total_penalty = new_penalty
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
            
            # Cool down
            T *= 0.999

    return best_set
