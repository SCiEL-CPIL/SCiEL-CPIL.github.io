import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit

    def can_add(x, selected, used_sums):
        if 2 * x in used_sums:
            return False
        for y in selected:
            if x + y in used_sums:
                return False
        return True

    def add_element(x, selected, used_sums):
        selected.append(x)
        used_sums.add(2 * x)
        for y in selected:
            used_sums.add(x + y)

    def get_addable(selected, used_sums):
        addable = []
        for x in range(1, N + 1):
            if can_add(x, selected, used_sums):
                addable.append(x)
        return addable

    def smart_greedy():
        selected = []
        used_sums = set()
        
        while True:
            if time.time() > deadline:
                break
                
            addable = get_addable(selected, used_sums)
            if not addable:
                break
            
            # If only one choice, take it
            if len(addable) == 1:
                x = addable[0]
                add_element(x, selected, used_sums)
                continue
            
            # Evaluate each candidate to see how many others it blocks
            best_candidates = []
            min_remaining = N + 1
            
            for x in addable:
                # Simulate adding x
                # We need to count how many other addable elements remain addable
                # after adding x.
                
                # Create temporary copies
                temp_selected = selected + [x]
                temp_used_sums = set(used_sums)
                temp_used_sums.add(2 * x)
                for z in selected:
                    temp_used_sums.add(x + z)
                
                count_remaining = 0
                for y in addable:
                    if y == x:
                        continue
                    if can_add(y, temp_selected, temp_used_sums):
                        count_remaining += 1
                
                if count_remaining < min_remaining:
                    min_remaining = count_remaining
                    best_candidates = [x]
                elif count_remaining == min_remaining:
                    best_candidates.append(x)
            
            # Pick randomly among the best candidates to add diversity
            best_x = random.choice(best_candidates)
            add_element(best_x, selected, used_sums)
            
        return selected

    # Run smart greedy
    best_set = smart_greedy()
    
    # If we have time, we could try random restarts with smart greedy logic
    # but smart greedy is deterministic-ish (random tie break), so it's likely stable.
    # Let's just return the result.
    
    return best_set
