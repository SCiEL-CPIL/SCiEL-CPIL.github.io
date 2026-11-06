import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, leaving 10s safety margin
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    N = len(smooth_numbers)
    
    # Precompute incompatibility matrix
    # incompat[i][j] is True if smooth_numbers[i] and smooth_numbers[j] cannot be in the same antichain
    incompat = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if b % a == 0 or a % b == 0:
                incompat[i][j] = True
                incompat[j][i] = True
                
    answers = [None] * 129  # Index 0 unused, 1..128
    
    # Helper to check if a list of indices forms an antichain
    def is_antichain(indices):
        n = len(indices)
        for i in range(n):
            for j in range(i + 1, n):
                if incompat[indices[i]][indices[j]]:
                    return False
        return True
        
    # Helper to check time and return if deadline exceeded
    def time_check():
        return time.time() > deadline

    # Exhaustive search for minimal k
    # We iterate k from 1 upwards.
    # For each k, we check all combinations of size k.
    # If it forms an antichain and sum <= 128, and answers[sum] is None, we update it.
    # Since we go by increasing k, the first update is the minimal one.
    
    max_k = 8 # Safe upper bound, most targets will be found with k <= 5
    
    for k in range(1, max_k + 1):
        if time_check():
            break
            
        # If all targets are found, we can stop
        if all(ans is not None for ans in answers[1:]):
            break
            
        for combo in itertools.combinations(range(N), k):
            # Check antichain property
            valid = True
            for i in range(k):
                for j in range(i + 1, k):
                    if incompat[combo[i]][combo[j]]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Calculate sum
            s = sum(smooth_numbers[idx] for idx in combo)
            
            if s <= 128:
                if answers[s] is None:
                    answers[s] = [smooth_numbers[idx] for idx in combo]
                # If answers[s] is not None, it was found with a smaller k, so we ignore this (larger k) solution.

    # Fallback for any missing targets (should not happen if max_k is sufficient)
    for t in range(1, 129):
        if answers[t] is None:
            # Try larger k if needed
            for r in range(max_k + 1, 15):
                found = False
                for combo in itertools.combinations(range(N), r):
                    valid = True
                    for i in range(r):
                        for j in range(i + 1, r):
                            if incompat[combo[i]][combo[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        s = sum(smooth_numbers[idx] for idx in combo)
                        if s == t:
                            answers[t] = [smooth_numbers[idx] for idx in combo]
                            found = True
                            break
                if found:
                    break
                if time_check():
                    break

    return answers[1:]
