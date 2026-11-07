import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s limit
    
    N = 128
    
    def is_valid(s_set):
        """Check if no three distinct a,b,c in s_set have all pair sums in s_set."""
        elems = sorted(s_set)
        n = len(elems)
        in_set = [False] * (2 * N + 1)
        for x in elems:
            if x <= 2 * N:
                in_set[x] = True
        
        # For each pair (i, j) with i < j, if elems[i]+elems[j] is in set,
        # check if there exists k > j such that elems[i]+elems[k] in set AND elems[j]+elems[k] in set
        for i in range(n):
            for j in range(i + 1, n):
                s1 = elems[i] + elems[j]
                if s1 > 2 * N or not in_set[s1]:
                    continue
                # Now check if there's a k > j with both sums in set
                for k in range(j + 1, n):
                    s2 = elems[i] + elems[k]
                    s3 = elems[j] + elems[k]
                    if s2 <= 2 * N and in_set[s2] and s3 <= 2 * N and in_set[s3]:
                        return False
        return True
    
    # Initial candidate: upper half
    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)
    
    # Try to improve by local search
    # Strategy: try adding elements from lower half, if invalid, try removing some upper elements
    
    # Precompute: for the initial set, all sums > N so it's valid
    # Try to add lower elements one by one
    
    # Build a list of candidate elements to try adding (lower half)
    lower_candidates = list(range(1, N // 2 + 1))
    random.seed(42)
    random.shuffle(lower_candidates)
    
    # Also consider removing some upper elements to make room
    # Use a greedy approach with local search
    
    # Let's try a more systematic approach:
    # Start with upper half, try to add each lower element
    # If adding makes it invalid, try removing one upper element to fix
    
    current = set(best_set)
    
    # Try adding lower elements in random order
    for elem in lower_candidates:
        if elem in current:
            continue
        trial = current | {elem}
        if is_valid(trial):
            current = trial
            if len(current) > best_size:
                best_set = set(current)
                best_size = len(current)
            if time.time() > deadline - 5:
                break
    
    # Now try local search: swap operations
    # For each element in current, try replacing it with an element not in current
    # to see if we can maintain validity and increase size (though swap keeps same size)
    # Actually, we want to increase size, so we should try adding elements
    
    # Let's do multiple restarts with different initial sets
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Different initial strategies
        if restart == 0:
            # Already done above
            continue
        
        # Try: take upper half, remove some random elements, add lower elements
        init = set(range(N // 2 + 1, N + 1))
        
        # Remove some random upper elements
        upper_list = list(range(N // 2 + 1, N + 1))
        num_remove = random.randint(0, 10)
        to_remove = set(random.sample(upper_list, num_remove))
        init -= to_remove
        
        # Add lower elements greedily
        lower_list = list(range(1, N // 2 + 1))
        random.shuffle(lower_list)
        
        current = set(init)
        for elem in lower_list:
            if elem in current:
                continue
            trial = current | {elem}
            if is_valid(trial):
                current = trial
        
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
    
    # Final local search on best_set: try to add any remaining element
    remaining = set(range(1, N + 1)) - best_set
    for elem in remaining:
        if time.time() > deadline - 2:
            break
        trial = best_set | {elem}
        if is_valid(trial):
            best_set = trial
    
    # Ensure return type is correct
    result = list(best_set)
    # Verify all are Python int
    result = [int(x) for x in result]
    return result
