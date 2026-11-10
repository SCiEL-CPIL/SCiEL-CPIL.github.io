import time
import random
import numpy as np

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    best_size = 0
    best_set = []
    
    def compute_counts(S):
        """Compute sum counts using numpy for speed."""
        arr = np.zeros(2 * N + 1, dtype=np.int32)
        if not S:
            return arr
        S_arr = np.array(S, dtype=np.int32)
        # Use outer sum approach
        for x in S:
            arr[x + S] += 1
        return arr

    def is_valid_addition(x, S, counts):
        """Check if x can be added to S without violating B2[2] constraint."""
        if not S:
            return True
        for y in S:
            if counts[x + y] >= 2:
                return False
        if counts[2 * x] >= 2:
            return False
        return True

    def greedy_build(order):
        S = []
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        for x in order:
            if is_valid_addition(x, S, counts):
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
        return S

    def local_search(S, max_rounds=10):
        """Try to improve set by removing and re-adding elements."""
        if not S:
            return S
        
        current_set = list(S)
        in_set = set(current_set)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 10.0:
                break
            
            # Rebuild counts
            counts = np.zeros(2 * N + 1, dtype=np.int32)
            for x in current_set:
                for y in current_set:
                    if x <= y:
                        counts[x + y] += 1
            
            improved = False
            # Try removing each element and see if we can add more
            for i in range(len(current_set)):
                if time.time() > deadline - 10.0:
                    break
                
                elem = current_set[i]
                temp_set = current_set[:i] + current_set[i+1:]
                temp_in_set = in_set - {elem}
                
                # Rebuild counts for temp_set
                temp_counts = np.zeros(2 * N + 1, dtype=np.int32)
                for x in temp_set:
                    for y in temp_set:
                        if x <= y:
                            temp_counts[x + y] += 1
                
                # Try to add back elements not in temp_set
                added = 0
                for x in range(1, N + 1):
                    if x in temp_in_set:
                        continue
                    
                    valid = True
                    for y in temp_set:
                        if temp_counts[x + y] >= 2:
                            valid = False
                            break
                    if valid and temp_counts[2 * x] >= 2:
                        valid = False
                    
                    if valid:
                        for y in temp_set:
                            temp_counts[x + y] += 1
                        temp_counts[2 * x] += 1
                        temp_set.append(x)
                        temp_in_set.add(x)
                        added += 1
                
                if added > 1:
                    current_set = temp_set
                    in_set = temp_in_set
                    improved = True
            
            if not improved:
                break
        
        return current_set

    def try_order(order):
        nonlocal best_size, best_set
        S = greedy_build(order)
        S_improved = local_search(S)
        
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved

    # 1. Natural order
    try_order(list(range(1, N + 1)))

    # 2. Reverse order
    if time.time() < deadline - 20:
        try_order(list(range(N, 0, -1)))

    # 3. Primes first
    if time.time() < deadline - 20:
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

    # 4. Random shuffles with local search
    rng = random.Random(42)
    no_improve_count = 0
    prev_best = best_size
    
    while time.time() < deadline - 15:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = greedy_build(order)
        S_improved = local_search(S)
        
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > 30:
            break

    # 5. Dense random initialization with local search
    if time.time() < deadline - 10 and best_size < 750:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Start with a dense random subset
            random_subset = rng.sample(range(1, N + 1), 700)
            random_subset.sort()
            
            S = greedy_build(random_subset)
            S_improved = local_search(S, max_rounds=5)
            
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = S_improved

    return best_set
