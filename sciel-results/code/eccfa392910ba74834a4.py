import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 4096
    max_sum = 3 * n  # Maximum possible triple sum

    best_set = []
    best_len = 0

    def is_valid_set(S):
        """Check if set S has all unique unordered triple sums."""
        S = list(S)
        k = len(S)
        used = bytearray(max_sum + 1)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = S[i] + S[j] + S[l]
                    if used[s]:
                        return False
                    used[s] = 1
        return True

    def build_greedy_random(seed_val):
        """Build a set using randomized greedy approach."""
        random.seed(seed_val)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        used = bytearray(max_sum + 1)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check if x can be added
            # New sums: 3x, 2x+y for y in S, x+y+z for y<=z in S
            valid = True
            
            # Check 3x
            s = 3 * x
            if used[s]:
                valid = False
            
            if valid:
                # Check 2x + y for each y in S
                for y in S:
                    s = 2 * x + y
                    if used[s]:
                        valid = False
                        break
            
            if valid:
                # Check x + y + z for y <= z in S
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        s = x + S[i] + S[j]
                        if used[s]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Add x and update used
                S.append(x)
                used[3 * x] = 1
                for y in S[:-1]:
                    used[2 * x + y] = 1
                for i in range(len(S) - 1):
                    for j in range(i, len(S) - 1):
                        used[x + S[i] + S[j]] = 1
        
        return S

    def local_search_add(S, used):
        """Try to add more elements to S."""
        S = list(S)
        used = bytearray(used)
        
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            
            if x in S:
                continue
            
            valid = True
            s = 3 * x
            if used[s]:
                valid = False
            
            if valid:
                for y in S:
                    s = 2 * x + y
                    if used[s]:
                        valid = False
                        break
            
            if valid:
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        s = x + S[i] + S[j]
                        if used[s]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                S.append(x)
                used[3 * x] = 1
                for y in S[:-1]:
                    used[2 * x + y] = 1
                for i in range(len(S) - 1):
                    for j in range(i, len(S) - 1):
                        used[x + S[i] + S[j]] = 1
        
        return S

    def local_search_swap(S, used, max_swaps=100):
        """Try swap operations to potentially increase set size."""
        S = list(S)
        used = bytearray(used)
        k = len(S)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            # Try removing a random element and adding a new one
            idx = random.randint(0, len(S) - 1)
            removed = S[idx]
            S.pop(idx)
            
            # Recompute used after removal
            used = bytearray(max_sum + 1)
            for i in range(len(S)):
                for j in range(i, len(S)):
                    for l in range(j, len(S)):
                        used[S[i] + S[j] + S[l]] = 1
            
            # Try to add a new element
            added = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in S:
                    continue
                
                valid = True
                s = 3 * x
                if used[s]:
                    valid = False
                
                if valid:
                    for y in S:
                        s = 2 * x + y
                        if used[s]:
                            valid = False
                            break
                
                if valid:
                    for i in range(len(S)):
                        for j in range(i, len(S)):
                            s = x + S[i] + S[j]
                            if used[s]:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    S.append(x)
                    used[3 * x] = 1
                    for y in S[:-1]:
                        used[2 * x + y] = 1
                    for i in range(len(S) - 1):
                        for j in range(i, len(S) - 1):
                            used[x + S[i] + S[j]] = 1
                    added = True
                    break
            
            if not added:
                # Restore
                S.insert(idx, removed)
                used = bytearray(max_sum + 1)
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        for l in range(j, len(S)):
                            used[S[i] + S[j] + S[l]] = 1
        
        return S

    # Strategy 1: Power of 3 base
    power3_set = [3**i for i in range(8) if 3**i <= n]
    if is_valid_set(power3_set):
        best_set = list(power3_set)
        best_len = len(power3_set)

    # Strategy 2: Greedy smallest-first
    S_greedy = []
    used_greedy = bytearray(max_sum + 1)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        valid = True
        s = 3 * x
        if used_greedy[s]:
            valid = False
        if valid:
            for y in S_greedy:
                s = 2 * x + y
                if used_greedy[s]:
                    valid = False
                    break
        if valid:
            for i in range(len(S_greedy)):
                for j in range(i, len(S_greedy)):
                    s = x + S_greedy[i] + S_greedy[j]
                    if used_greedy[s]:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            S_greedy.append(x)
            used_greedy[3 * x] = 1
            for y in S_greedy[:-1]:
                used_greedy[2 * x + y] = 1
            for i in range(len(S_greedy) - 1):
                for j in range(i, len(S_greedy) - 1):
                    used_greedy[x + S_greedy[i] + S_greedy[j]] = 1

    if len(S_greedy) > best_len:
        best_set = S_greedy
        best_len = len(S_greedy)

    # Strategy 3: Multiple randomized greedy trials
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        S = build_greedy_random(trial)
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
            # Try to improve with local search
            used = bytearray(max_sum + 1)
            for i in range(len(S)):
                for j in range(i, len(S)):
                    for l in range(j, len(S)):
                        used[S[i] + S[j] + S[l]] = 1
            
            S = local_search_add(S, used)
            if len(S) > best_len:
                best_set = S
                best_len = len(S)
            
            if best_len < 80:
                S = local_search_swap(S, used, max_swaps=50)
                if len(S) > best_len:
                    best_set = S
                    best_len = len(S)

    # Strategy 4: More aggressive local search from current best
    if best_len < 80 and time.time() < deadline - 10.0:
        used = bytearray(max_sum + 1)
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                for l in range(j, len(best_set)):
                    used[best_set[i] + best_set[j] + best_set[l]] = 1
        
        S = local_search_add(best_set, used)
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
        
        if best_len < 80:
            used = bytearray(max_sum + 1)
            for i in range(len(best_set)):
                for j in range(i, len(best_set)):
                    for l in range(j, len(best_set)):
                        used[best_set[i] + best_set[j] + best_set[l]] = 1
            S = local_search_swap(best_set, used, max_swaps=200)
            if len(S) > best_len:
                best_set = S
                best_len = len(S)

    # Final validation
    if best_set and not is_valid_set(best_set):
        best_set = [3**i for i in range(8) if 3**i <= n]

    return best_set
