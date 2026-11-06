import itertools
import time

def construct():
    start_time = time.time()
    
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
        
    # Iterate over number of terms k
    # We know k=1 covers smooth numbers themselves.
    # k=2 covers sums of two.
    # etc.
    
    # To be efficient, we can just iterate k and find the first valid antichain sum for each target.
    # Since we want minimal k for each target, we process k=1, then k=2, etc.
    # If a target is already assigned, we skip it.
    
    # Optimization: Pre-compute all valid antichains for small k?
    # Or just generate combinations and filter.
    
    # Let's try generating combinations for k=1, 2, 3, 4, 5...
    
    # k=1
    for i in range(N):
        s = smooth_numbers[i]
        if s <= 128 and answers[s] is None:
            answers[s] = [smooth_numbers[i]]
            
    # Check if all found
    if all(ans is not None for ans in answers[1:]):
        return answers[1:]

    # k=2
    for i in range(N):
        for j in range(i + 1, N):
            if incompat[i][j]:
                continue
            s = smooth_numbers[i] + smooth_numbers[j]
            if s > 128:
                break
            if answers[s] is None:
                answers[s] = [smooth_numbers[i], smooth_numbers[j]]
                
    # Check if all found
    if all(ans is not None for ans in answers[1:]):
        return answers[1:]

    # k=3
    for i in range(N):
        for j in range(i + 1, N):
            if incompat[i][j]:
                continue
            for k in range(j + 1, N):
                if incompat[i][k] or incompat[j][k]:
                    continue
                s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k]
                if s > 128:
                    break
                if answers[s] is None:
                    answers[s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k]]
                    
    # Check if all found
    if all(ans is not None for ans in answers[1:]):
        return answers[1:]

    # k=4
    # This might be the heavy part, but N is small (~50).
    # Number of combinations of 4 from 50 is ~230k. Very fast.
    for i in range(N):
        for j in range(i + 1, N):
            if incompat[i][j]:
                continue
            for k in range(j + 1, N):
                if incompat[i][k] or incompat[j][k]:
                    continue
                for l in range(k + 1, N):
                    if incompat[i][l] or incompat[j][l] or incompat[k][l]:
                        continue
                    s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[l]
                    if s > 128:
                        break
                    if answers[s] is None:
                        answers[s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[l]]
                        
    # Check if all found
    if all(ans is not None for ans in answers[1:]):
        return answers[1:]

    # k=5
    # If we get here, some targets need 5 terms.
    # Combinations of 5 from 50 is ~2.1M. Might be slow in pure Python loops.
    # Let's add a time check.
    if time.time() - start_time > 50:
        # Fallback for remaining targets
        for t in range(1, 129):
            if answers[t] is None:
                # Brute force with itertools
                for r in range(5, 10):
                    for combo in itertools.combinations(range(N), r):
                        # Check antichain
                        valid = True
                        for a in range(r):
                            for b in range(a+1, r):
                                if incompat[combo[a]][combo[b]]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            s = sum(smooth_numbers[idx] for idx in combo)
                            if s == t:
                                answers[t] = [smooth_numbers[idx] for idx in combo]
                                break
                    if answers[t] is not None:
                        break
        return answers[1:]

    for i in range(N):
        for j in range(i + 1, N):
            if incompat[i][j]:
                continue
            for k in range(j + 1, N):
                if incompat[i][k] or incompat[j][k]:
                    continue
                for l in range(k + 1, N):
                    if incompat[i][l] or incompat[j][l] or incompat[k][l]:
                        continue
                    for m in range(l + 1, N):
                        if incompat[i][m] or incompat[j][m] or incompat[k][m] or incompat[l][m]:
                            continue
                        s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[l] + smooth_numbers[m]
                        if s > 128:
                            break
                        if answers[s] is None:
                            answers[s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[l], smooth_numbers[m]]
                            
    # Check if all found
    if all(ans is not None for ans in answers[1:]):
        return answers[1:]

    # Final fallback for any missing
    for t in range(1, 129):
        if answers[t] is None:
            for r in range(5, 15):
                for combo in itertools.combinations(range(N), r):
                    valid = True
                    for a in range(r):
                        for b in range(a+1, r):
                            if incompat[combo[a]][combo[b]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        s = sum(smooth_numbers[idx] for idx in combo)
                        if s == t:
                            answers[t] = [smooth_numbers[idx] for idx in combo]
                            break
                if answers[t] is not None:
                    break
                    
    return answers[1:]
