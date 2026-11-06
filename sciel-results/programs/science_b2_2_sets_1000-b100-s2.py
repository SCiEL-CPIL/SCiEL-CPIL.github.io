import random
import time

def construct():
    N = 1000
    start_time = time.time()
    limit = 115  # Leave 5s buffer

    # Helper to check if x can be added to A given counts
    def is_valid_add(A, x, counts):
        if x in A:
            return False
        if counts.get(2*x, 0) >= 2:
            return False
        for y in A:
            if counts.get(x+y, 0) >= 2:
                return False
        return True

    # Helper to add x to A and update counts
    def add_element(A, A_set, counts, x):
        A.append(x)
        A_set.add(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in A[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    # Helper to remove x from A and update counts
    def remove_element(A, A_set, counts, x):
        A.remove(x)
        A_set.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in A:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    # 1. Greedy Initialization
    A = []
    A_set = set()
    counts = {}
    
    for x in range(1, N + 1):
        if is_valid_add(A, x, counts):
            add_element(A, A_set, counts, x)
            
    best_A = list(A)
    best_size = len(A)
    
    # 2. Local Search with Swaps and Adds
    # We maintain A (list), A_set (set), counts (dict)
    
    while time.time() - start_time < limit:
        moved = False
        
        # Try to Add
        # Find a valid x to add
        # To speed up, we can pre-compute valid candidates or just iterate
        valid_adds = []
        for x in range(1, N + 1):
            if x not in A_set:
                if is_valid_add(A, x, counts):
                    valid_adds.append(x)
        
        if valid_adds:
            x = random.choice(valid_adds)
            add_element(A, A_set, counts, x)
            if len(A) > best_size:
                best_size = len(A)
                best_A = list(A)
            continue # Loop again, maybe another add is possible
        
        # No adds possible. Try Swaps.
        # A swap is: remove a, add b.
        # We need b to be invalid now, but valid after removing a.
        
        # Pick random a
        if not A:
            break
        a = random.choice(A)
        
        # Temporarily remove a
        remove_element(A, A_set, counts, a)
        
        # Now look for any valid b to add
        # We can try a few random candidates
        found_b = None
        # To avoid bias, we can shuffle a subset or just pick randoms
        # Let's try up to 10 random candidates
        for _ in range(10):
            b = random.randint(1, N)
            if b in A_set:
                continue
            if is_valid_add(A, b, counts):
                found_b = b
                break
        
        if found_b is not None:
            # Perform the swap: a is already removed, add b
            add_element(A, A_set, counts, found_b)
            moved = True
        else:
            # No swap possible with this a. Re-add a.
            add_element(A, A_set, counts, a)
            # If we couldn't swap, try a Remove step (Perturbation)
            # Remove a random element to break local optimum
            if A:
                a_remove = random.choice(A)
                remove_element(A, A_set, counts, a_remove)
                # The next iteration will likely find an Add.

    return best_A
