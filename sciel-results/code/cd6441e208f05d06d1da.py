import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def can_add(x, selected, existing_sums):
        """Check if x can be added to the set without creating duplicate triple sums."""
        # 1. x, x, x -> 3x
        if 3 * x in existing_sums:
            return False
        
        # 2. x, x, y -> 2x + y
        for y in selected:
            if 2 * x + y in existing_sums:
                return False
        
        # 3. x, y, z -> x + y + z
        k = len(selected)
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                if x + y + z in existing_sums:
                    return False
        
        return True
    
    def add_element(x, selected, existing_sums):
        """Add x to selected and update existing_sums."""
        selected.append(x)
        # Add 3x
        existing_sums.add(3 * x)
        # Add 2x + y
        for y in selected[:-1]:
            existing_sums.add(2 * x + y)
        # Add x + y + z
        k_old = len(selected) - 1
        for i in range(k_old):
            y = selected[i]
            for j in range(i, k_old):
                z = selected[j]
                existing_sums.add(x + y + z)
    
    def rebuild_sums(selected):
        """Rebuild existing_sums from scratch."""
        existing_sums = set()
        for i in range(len(selected)):
            x = selected[i]
            existing_sums.add(3 * x)
            for j in range(i):
                y = selected[j]
                existing_sums.add(2 * x + y)
                for k in range(j):
                    z = selected[k]
                    existing_sums.add(x + y + z)
        return existing_sums
    
    def try_greedy(start=1, end=4096, step=1, skip_initial=None):
        selected = []
        existing_sums = set()
        
        for x in range(start, end + 1, step):
            if time.time() > deadline:
                return selected
            
            if skip_initial and x in skip_initial:
                continue
            
            if can_add(x, selected, existing_sums):
                add_element(x, selected, existing_sums)
        
        return selected
    
    def local_search(selected, max_iters=100):
        """Try to improve the set by swapping elements or adding new ones."""
        selected = list(selected)
        existing_sums = rebuild_sums(selected)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2.0:
            improved = False
            iters += 1
            
            # Try to add a new element
            for x in range(1, 4097):
                if x in selected:
                    continue
                if can_add(x, selected, existing_sums):
                    add_element(x, selected, existing_sums)
                    improved = True
                    break
            
            # Try to swap an element with a larger one
            for i in range(len(selected)):
                old_x = selected[i]
                # Try replacing old_x with a larger value
                for new_x in range(old_x + 1, 4097):
                    if new_x in selected:
                        continue
                    
                    # Temporarily remove old_x
                    selected[i] = 0  # mark as removed
                    remaining = [v for v in selected if v != 0]
                    remaining_sums = rebuild_sums(remaining)
                    
                    if can_add(new_x, remaining, remaining_sums):
                        # Do the swap
                        selected[i] = new_x
                        existing_sums = rebuild_sums(selected)
                        improved = True
                        break
                    
                    # Restore
                    selected[i] = old_x
            
            if not improved:
                break
        
        return sorted(selected)
    
    # Standard greedy
    best_set = try_greedy(1, 4096, 1)
    best_len = len(best_set)
    
    # Try a few variations with small skips to see if we can get a larger set
    variations = [
        {2}, {3}, {4}, {2, 3}, {5}, {1, 2}
    ]
    
    for skip in variations:
        if time.time() > deadline - 5.0:
            break
        candidate = try_greedy(1, 4096, 1, skip_initial=skip)
        if len(candidate) > best_len:
            best_set = candidate
            best_len = len(candidate)
    
    # If we still have time, try random restarts with random skips
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 8.0:
                break
            skip_count = random.randint(1, 3)
            skips = set(random.sample(range(1, 20), min(skip_count, 19)))
            candidate = try_greedy(1, 4096, 1, skip_initial=skips)
            if len(candidate) > best_len:
                best_set = candidate
                best_len = len(candidate)
    
    # Apply local search to the best set found
    if time.time() < deadline - 5.0:
        improved_set = local_search(best_set, max_iters=50)
        if len(improved_set) > best_len:
            best_set = improved_set
            best_len = len(improved_set)
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
