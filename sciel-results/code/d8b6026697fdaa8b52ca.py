import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def is_valid(x, selected, used_sums):
        # Check 3x
        if 3 * x in used_sums:
            return False
        
        # Check 2x + y
        for y in selected:
            if 2 * x + y in used_sums:
                return False
        
        # Check x + y + z
        k = len(selected)
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                if x + y + z in used_sums:
                    return False
        return True

    def add_sums(x, selected, used_sums):
        used_sums.add(3 * x)
        for y in selected:
            used_sums.add(2 * x + y)
        k = len(selected)
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                used_sums.add(x + y + z)

    def remove_sums(x, selected, used_sums):
        used_sums.remove(3 * x)
        for y in selected:
            used_sums.remove(2 * x + y)
        k = len(selected)
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                used_sums.remove(x + y + z)

    def try_greedy_with_lookahead(skip_set=None, lookahead_depth=1):
        if skip_set is None:
            skip_set = set()
        
        selected = []
        used_sums = set()
        
        for x in range(1, 4097):
            if time.time() > deadline:
                break
                
            if x in skip_set:
                continue
                
            if is_valid(x, selected, used_sums):
                # Lookahead: Check if adding x allows at least one more number
                if lookahead_depth > 0:
                    # Simulate adding x
                    temp_selected = selected + [x]
                    temp_used = used_sums.copy()
                    add_sums(x, selected, temp_used)
                    
                    valid_next = False
                    # Try to find any valid y > x
                    for y in range(x + 1, min(4097, x + 100)): # Limit search range for speed
                        if is_valid(y, temp_selected, temp_used):
                            valid_next = True
                            break
                    
                    if not valid_next and len(selected) > 0:
                        # If we can't add anything after x, and we already have some elements,
                        # it might be better to skip x to allow a denser packing later?
                        # This is a heuristic. If x is small, skipping it might be bad.
                        # We only skip if we are not at the very beginning.
                        if len(selected) >= 2:
                            continue
                    
                    # If valid_next is True, or we are at the start, we proceed with x
                    # Note: We don't actually add y here, we just check if x is a "dead end".
                
                # Commit to x
                selected.append(x)
                add_sums(x, selected[:-1], used_sums)
                
        return selected

    best_set = []
    
    # 1. Standard greedy with lookahead
    candidate = try_greedy_with_lookahead(lookahead_depth=1)
    if len(candidate) > len(best_set):
        best_set = candidate
        
    # 2. Try skipping small numbers
    # Skipping 1, 2, etc. can sometimes allow a better structure
    for skip in [1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3, 4]:
        if time.time() > deadline - 1.0:
            break
        skip_set = {skip}
        candidate = try_greedy_with_lookahead(skip_set=skip_set, lookahead_depth=1)
        if len(candidate) > len(best_set):
            best_set = candidate
            
    # 3. Try skipping pairs of small numbers
    if time.time() < deadline - 5.0:
        pairs = [(1,2), (1,3), (2,3), (1,4), (2,4), (3,4), (1,5), (2,5), (3,5), (4,5)]
        for p in pairs:
            if time.time() > deadline - 1.0:
                break
            skip_set = set(p)
            candidate = try_greedy_with_lookahead(skip_set=skip_set, lookahead_depth=1)
            if len(candidate) > len(best_set):
                best_set = candidate

    # 4. Randomized restarts with random skips
    if time.time() < deadline - 2.0:
        for _ in range(20):
            if time.time() > deadline - 1.0:
                break
            # Skip 1-3 random numbers from 1 to 50
            skip_count = random.randint(1, 3)
            skips = set(random.sample(range(1, 51), skip_count))
            candidate = try_greedy_with_lookahead(skip_set=skips, lookahead_depth=1)
            if len(candidate) > len(best_set):
                best_set = candidate

    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
