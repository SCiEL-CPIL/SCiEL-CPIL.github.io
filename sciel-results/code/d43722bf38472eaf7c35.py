import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_size = 0
    best_set = []
    
    # Helper to check and add - FIXED BUG: iterate over S before appending x
    def greedy_build(order):
        S = []
        counts = [0] * (2 * N + 1)
        for x in order:
            valid = True
            # Check sums x + y for y in S (current S, not including x yet)
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            # Check sum 2x
            if valid:
                s = 2 * x
                if counts[s] >= 2:
                    valid = False
            
            if valid:
                # Update counts BEFORE appending x to S
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
        return S

    def try_order(order):
        nonlocal best_size, best_set
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    def local_search(S, max_iter=5000):
        """Try to improve S by swapping elements or inserting new ones."""
        S = list(S)
        S_set = set(S)
        counts = [0] * (2 * N + 1)
        
        # Build counts
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                if i <= j:
                    counts[a + b] += 1
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            # Try inserting each candidate not in S
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                
                # Check if x can be added
                can_add = True
                for y in S:
                    s = x + y
                    if counts[s] >= 2:
                        can_add = False
                        break
                if can_add:
                    s = 2 * x
                    if counts[s] >= 2:
                        can_add = False
                
                if can_add:
                    # Add x
                    for y in S:
                        counts[x + y] += 1
                    counts[2 * x] += 1
                    S.append(x)
                    S_set.add(x)
                    improved = True
            
            # Try swapping: remove one element, add another
            for i, a in enumerate(list(S)):
                # Try removing a
                # First, temporarily remove a from counts
                for j, b in enumerate(S):
                    if i <= j:
                        counts[a + b] -= 1
                counts[2 * a] -= 1
                
                S_temp = S[:i] + S[i+1:]
                S_set_temp = S_set - {a}
                
                # Try adding each candidate
                for x in range(1, N + 1):
                    if x in S_set_temp:
                        continue
                    
                    can_add = True
                    for y in S_temp:
                        s = x + y
                        if counts[s] >= 2:
                            can_add = False
                            break
                    if can_add:
                        s = 2 * x
                        if counts[s] >= 2:
                            can_add = False
                    
                    if can_add:
                        # This swap is valid - apply it
                        for y in S_temp:
                            counts[x + y] += 1
                        counts[2 * x] += 1
                        S = S_temp + [x]
                        S_set = S_set_temp | {x}
                        improved = True
                        break
                
                # If no improvement, restore counts for a
                if not improved:
                    for j, b in enumerate(S):
                        if i <= j:
                            counts[a + b] += 1
                    counts[2 * a] += 1
                    S_set_temp = S_set
        
        return S

    # 1. Natural order
    try_order(list(range(1, N + 1)))

    # 2. Reverse order
    if time.time() < deadline:
        try_order(list(range(N, 0, -1)))

    # 3. Try several structured orderings
    if time.time() < deadline:
        # Primes first, then composites
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = [x for x in range(1, N+1) if is_prime(x)]
        composites = [x for x in range(1, N+1) if not is_prime(x)]
        try_order(primes + composites)

    # 4. Sparse pattern: powers of 2 first, then fill in
    if time.time() < deadline:
        powers = [2**k for k in range(10) if 2**k <= N]
        rest = [x for x in range(1, N+1) if x not in set(powers)]
        try_order(powers + rest)

    # 5. Random shuffles with different seeds
    rng = random.Random(42)
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        try_order(order)
        if best_size >= 750:  # good enough threshold
            break

    # 6. Local search on best solution
    if time.time() < deadline and best_set:
        improved = local_search(best_set, max_iter=3000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved

    # 7. More random restarts with local search if time permits
    rng2 = random.Random(123)
    while time.time() < deadline - 5:
        order = list(range(1, N + 1))
        rng2.shuffle(order)
        S = greedy_build(order)
        if len(S) >= best_size:
            S_improved = local_search(S, max_iter=2000)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = list(S_improved)

    return best_set
