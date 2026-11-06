import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    # Pre-calculate initial sums
    init_sums = set()
    fixed_list = list(fixed)
    for i in range(len(fixed_list)):
        for j in range(i, len(fixed_list)):
            init_sums.add(fixed_list[i] + fixed_list[j])
            
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums

    def is_valid_addition(x, current_set, current_sums):
        # Check if x can be added to current_set
        # Sums involving x: x + y for y in current_set, and x + x
        for y in current_set:
            if x + y in current_sums:
                return False
        if x + x in current_sums:
            return False
        return True

    def add_to_set(x, current_set, current_sums):
        for y in current_set:
            current_sums.add(x + y)
        current_sums.add(x + x)
        current_set.append(x)
        current_set.sort()

    def remove_from_set(x, current_set, current_sums):
        # Remove sums involving x
        for y in current_set:
            if y != x:
                current_sums.discard(x + y)
        current_sums.discard(x + x)
        current_set.remove(x)

    # Initial greedy fill
    current_set = list(fixed)
    current_sums = get_sums(current_set)
    
    # Greedy add random candidates
    candidates = [x for x in range(1, 285) if x not in fixed]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if is_valid_addition(x, current_set, current_sums):
            add_to_set(x, current_set, current_sums)
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)

    # Simulated Annealing
    # We will try to improve the size by allowing temporary removals.
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Helper to get random non-fixed element
    def get_random_non_fixed(s):
        removable = [x for x in s if x not in fixed]
        if not removable:
            return None
        return random.choice(removable)

    while time.time() < deadline:
        if T < min_T:
            # Restart with higher T if stuck? Or just break?
            # Let's try to restart with a perturbation
            T = 1.0
            # Perturb: remove 2-4 random elements
            removable = [x for x in current_set if x not in fixed]
            if not removable:
                break
            num_remove = min(len(removable), random.randint(2, 4))
            to_remove = random.sample(removable, num_remove)
            for x in to_remove:
                remove_from_set(x, current_set, current_sums)
            # Greedy repair
            all_candidates = [x for x in range(1, 285) if x not in current_set]
            random.shuffle(all_candidates)
            for x in all_candidates:
                if time.time() > deadline:
                    break
                if is_valid_addition(x, current_set, current_sums):
                    add_to_set(x, current_set, current_sums)
            continue

        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random element
            x = random.randint(1, 284)
            if x in current_set:
                continue
            if is_valid_addition(x, current_set, current_sums):
                old_size = len(current_set)
                add_to_set(x, current_set, current_sums)
                new_size = len(current_set)
                # Always accept size increase
                if new_size > best_size:
                    best_size = new_size
                    best_set = list(current_set)
            else:
                # Try to remove a random element to allow this one later?
                # Or just accept the failure and try another move.
                # Let's try a swap instead if add fails?
                pass

        elif move_type < 0.8:
            # Try to remove a random non-fixed element
            x = get_random_non_fixed(current_set)
            if x is None:
                continue
            old_size = len(current_set)
            remove_from_set(x, current_set, current_sums)
            new_size = len(current_set)
            
            # Acceptance criterion
            if new_size < old_size:
                # Probability of accepting size decrease
                delta = old_size - new_size
                prob = math.exp(-delta / T)
                if random.random() > prob:
                    # Reject: add it back
                    add_to_set(x, current_set, current_sums)
                    # Note: add_to_set sorts, which is fine
                # If accepted, we stay with smaller set
            else:
                # Size same or increased (impossible for remove)
                pass

        else:
            # Swap: Remove x, Add y
            x = get_random_non_fixed(current_set)
            if x is None:
                continue
            
            # Try a few random y's
            swapped = False
            for _ in range(10): # Try up to 10 random candidates for y
                y = random.randint(1, 284)
                if y in current_set or y == x:
                    continue
                
                # Check if y is valid with current_set minus x
                # To do this efficiently, we can check if y is valid with current_set,
                # BUT we are removing x.
                # Sums in current_sums include sums with x.
                # We need to check if y + z is in (current_sums - sums_with_x) for all z in current_set - {x}.
                
                # Simpler: Just remove x, check y, then add y.
                # If y fails, we put x back.
                
                remove_from_set(x, current_set, current_sums)
                
                if is_valid_addition(y, current_set, current_sums):
                    add_to_set(y, current_set, current_sums)
                    swapped = True
                    break
                else:
                    # y is not valid, put x back
                    add_to_set(x, current_set, current_sums)
                    # continue to next y candidate
            
            # If we swapped, size is same.
            # If we didn't swap, state is same (x was put back).

        T *= cooling_rate

    return best_set
