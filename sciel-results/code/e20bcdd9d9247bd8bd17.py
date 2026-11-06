import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    m = len(smooth)
    
    # Precompute divisibility matrix for smooth numbers to speed up antichain check
    # div[i][j] is True if smooth[i] divides smooth[j]
    # Note: distinct terms, so i != j.
    # We need to check if any term divides another.
    # For a set of terms, we check pairs.
    
    # Precompute antichain status for pairs, triples, etc.
    # To minimize total summands, we should prioritize finding solutions with fewer terms.
    # The current program searches for k=1, then k=2, etc. for each n.
    # This is correct for minimizing terms per target, and thus total terms.
    
    # However, the "Inspiration 1" approach pre-generates all valid antichains of size 2 and 3
    # and assigns them to targets. This is slightly more efficient because it avoids
    # re-checking the same pairs/triples for every target n.
    # But the current program's approach of checking k=1 first is crucial for smooth numbers.
    
    # Let's refine the current program's approach by pre-computing valid antichains
    # for k=2 and k=3 and storing them in a dictionary by sum.
    # Then for each n, we check:
    # 1. Is n smooth? -> [n]
    # 2. Is there a valid antichain pair summing to n?
    # 3. Is there a valid antichain triple summing to n?
    # 4. Fallback to search for k=4 if needed.
    
    # This is faster because we don't iterate over all pairs for each n, but rather
    # iterate over all pairs once and store the results.
    
    # Precompute valid antichain pairs
    pair_solutions = {}
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            # Check antichain: a does not divide b and b does not divide a
            # Since a < b, we only need to check b % a != 0
            if b % a != 0:
                # If multiple pairs sum to the same s, we want to keep one.
                # Since we are minimizing terms, any valid pair is equivalent in count.
                # We can just store the first one found.
                if s not in pair_solutions:
                    pair_solutions[s] = [a, b]
    
    # Precompute valid antichain triples
    triple_solutions = {}
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            # Pruning: if a + b + (next smallest) > 128, break
            # But m is small, so just iterate
            for k in range(j + 1, m):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    break
                # Check antichain for (a, b, c)
                # a < b < c
                # Check pairs: (a,b), (a,c), (b,c)
                # b % a != 0
                # c % a != 0
                # c % b != 0
                if b % a != 0 and c % a != 0 and c % b != 0:
                    if s not in triple_solutions:
                        triple_solutions[s] = [a, b, c]
    
    answers = []
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback: if we run out of time, we must return something valid.
            # But given the precomputation, this loop should be very fast.
            # If we hit the deadline, we might have incomplete answers.
            # However, the precomputation happens before this loop, so the loop itself
            # is just dictionary lookups, which are O(1).
            pass
        
        found = None
        
        # Try k=1
        if n in smooth_set:
            found = [n]
        
        # Try k=2
        if found is None:
            if n in pair_solutions:
                found = pair_solutions[n]
        
        # Try k=3
        if found is None:
            if n in triple_solutions:
                found = triple_solutions[n]
        
        # Try k=4 (Safety net, should rarely be needed)
        if found is None:
            # Brute force for k=4 for this specific n
            # a < b < c < d
            for i in range(m):
                a = smooth[i]
                if a * 4 > n:
                    break
                for j in range(i + 1, m):
                    b = smooth[j]
                    if a + b + 2 * (b + 1) > n: # Loose bound
                        break
                    for k in range(j + 1, m):
                        c = smooth[k]
                        if a + b + c >= n:
                            break
                        d = n - a - b - c
                        if d in smooth_set and d > c:
                            # Check antichain for all pairs
                            terms = [a, b, c, d]
                            is_antichain = True
                            for x_idx in range(4):
                                for y_idx in range(x_idx + 1, 4):
                                    if terms[y_idx] % terms[x_idx] == 0:
                                        is_antichain = False
                                        break
                                if not is_antichain:
                                    break
                            if is_antichain:
                                found = terms
                                break
                        if found is not None:
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
        
        # Fallback if still None (should not happen for n<=128)
        if found is None:
            # Try k=5? Or just return an empty list?
            # The task requires valid representations.
            # Let's try a quick search for k=5
            found = None
            for i in range(m):
                a = smooth[i]
                if a * 5 > n:
                    break
                for j in range(i + 1, m):
                    b = smooth[j]
                    if a + b + 3 * (b + 1) > n:
                        break
                    for k in range(j + 1, m):
                        c = smooth[k]
                        if a + b + c + 2 * (c + 1) > n:
                            break
                        for l in range(k + 1, m):
                            d = smooth[l]
                            if a + b + c + d >= n:
                                break
                            e = n - a - b - c - d
                            if e in smooth_set and e > d:
                                terms = [a, b, c, d, e]
                                is_antichain = True
                                for x_idx in range(5):
                                    for y_idx in range(x_idx + 1, 5):
                                        if terms[y_idx] % terms[x_idx] == 0:
                                            is_antichain = False
                                            break
                                    if not is_antichain:
                                        break
                                if is_antichain:
                                    found = terms
                                    break
                            if found is not None:
                                break
                        if found is not None:
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
        
        if found is None:
            # This should not happen. If it does, return an empty list which is invalid.
            # But we assume it won't.
            found = []
            
        answers.append(found)
        
    return answers
