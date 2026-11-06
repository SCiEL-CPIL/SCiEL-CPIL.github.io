import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    # Prescribed elements
    base = [1, 2, 5, 11]
    base_set = frozenset(base)

    # Helper to check validity and get sums
    def get_validity_and_sums(candidates):
        n = len(candidates)
        sums = set()
        for i in range(n):
            a = candidates[i]
            for j in range(i, n):
                b = candidates[j]
                s = a + b
                if s in sums:
                    return False, None
                sums.add(s)
        return True, sums

    def is_valid_set(s):
        valid, _ = get_validity_and_sums(sorted(s))
        return valid

    # Fast check if a candidate can be added to a set
    def can_add(candidate, current_set, current_sums):
        # Check if candidate is already in the set
        if candidate in current_set:
            return False
        
        # Check new sums: candidate + x for all x in current_set, and 2*candidate
        for x in current_set:
            s = candidate + x
            if s in current_sums:
                return False
        if 2 * candidate in current_sums:
            return False
            
        # Also need to check that the new sums don't conflict with each other?
        # No, since candidate is distinct from all x in current_set, 
        # {candidate + x} are distinct.
        # And 2*candidate is distinct from candidate+x if x != candidate (which is true).
        # So we just need to check against existing sums.
        return True

    def get_new_sums(candidate, current_set):
        return {candidate + x for x in current_set} | {2 * candidate}

    # Greedy expansion
    def try_expand(current_set, current_sums, max_size=20):
        curr = list(current_set)
        curr.sort()
        sums = set(current_sums)
        
        remaining = [i for i in range(1, 285) if i not in current_set]
        random.shuffle(remaining)
        
        for x in remaining:
            if len(curr) >= max_size:
                break
            if can_add(x, current_set, sums):
                curr.append(x)
                curr.sort()
                sums |= get_new_sums(x, current_set)
                current_set.add(x)
        
        return set(curr), sums

    # Simulated Annealing
    def simulated_annealing(initial_set, initial_sums, steps_per_temp=100, start_temp=1.0, end_temp=0.01):
        curr_set = set(initial_set)
        curr_sums = set(initial_sums)
        curr_size = len(curr_set)
        
        best_set = set(curr_set)
        best_sums = set(curr_sums)
        best_size = curr_size
        
        non_base = [x for x in curr_set if x not in base_set]
        
        for step in range(steps_per_temp):
            # Choose a move
            # 1. Add a random element
            # 2. Remove a random non-base element
            # 3. Swap a non-base element with a random element
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                candidates = [i for i in range(1, 285) if i not in curr_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(x, curr_set, curr_sums):
                    new_set = curr_set | {x}
                    new_sums = curr_sums | get_new_sums(x, curr_set)
                    new_size = len(new_set)
                    
                    # Accept if better or with some probability
                    if new_size > curr_size:
                        curr_set = new_set
                        curr_sums = new_sums
                        curr_size = new_size
                        if curr_size > best_size:
                            best_set = set(curr_set)
                            best_sums = set(curr_sums)
                            best_size = curr_size
                # If invalid, we could still accept with low prob to escape local minima?
                # But invalid sets are not allowed. We must stay in the valid space.
                # So we can only accept valid moves.
                # To escape local minima, we need to allow "worse" valid moves?
                # Size is the objective. A "worse" move would be removing an element.
                
            elif move_type < 0.7:
                # Remove
                if non_base:
                    x = random.choice(non_base)
                    if x in curr_set:
                        # Remove x and its sums
                        new_set = curr_set - {x}
                        # Recalculate sums is expensive, but necessary for correctness
                        # Or we can track which sums involve x
                        # Let's just recalculate for simplicity in this step, or optimize later
                        valid, new_sums = get_validity_and_sums(sorted(new_set))
                        if valid:
                            new_size = len(new_set)
                            # Always accept removal? No, it reduces size.
                            # We accept removal to allow subsequent better additions.
                            # This is a "kick".
                            curr_set = new_set
                            curr_sums = new_sums
                            curr_size = new_size
                            # Update non_base list
                            non_base = [x for x in curr_set if x not in base_set]
                            
            else:
                # Swap
                if non_base:
                    x = random.choice(non_base)
                    candidates = [i for i in range(1, 285) if i not in curr_set]
                    if candidates:
                        y = random.choice(candidates)
                        new_set = (curr_set - {x}) | {y}
                        valid, new_sums = get_validity_and_sums(sorted(new_set))
                        if valid:
                            new_size = len(new_set)
                            # Accept swap if size is same or better?
                            # Swap keeps size same. It changes the "landscape" for future additions.
                            # We should accept swaps with high probability to explore.
                            # Or use temperature?
                            # Let's accept swaps always if valid, to mix the set.
                            curr_set = new_set
                            curr_sums = new_sums
                            curr_size = new_size
                            non_base = [x for x in curr_set if x not in base_set]
        
        return best_set, best_sums

    # Main search loop
    best_set = set(base)
    best_sums = set()
    valid, best_sums = get_validity_and_sums(base)
    best_size = len(base)

    # Initial greedy expansions
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        curr_set, curr_sums = try_expand(set(base), set(best_sums))
        if len(curr_set) > best_size:
            best_set = curr_set
            best_sums = curr_sums
            best_size = len(curr_set)
            if best_size == 20:
                return sorted(best_set)

    # Simulated Annealing on the best set found
    if best_size > len(base):
        if time.time() < deadline - 20:
            best_set, best_sums = simulated_annealing(best_set, best_sums, steps_per_temp=500)
            if len(best_set) > best_size:
                best_size = len(best_set)
                if best_size == 20:
                    return sorted(best_set)

    # Final check
    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
