import time
import random
from itertools import combinations

def count_conflicts(S):
    sums = {}
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    total = 0
    for c in sums.values():
        if c > 1:
            total += c * (c - 1) // 2
    return total

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    n = len(S)
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_S = [None]
    best_true_size = [12]
    
    def check_and_update(S):
        true_size = max_sidon_size(S)
        if true_size < best_true_size[0]:
            best_true_size[0] = true_size
            best_S[0] = list(S)
            return true_size
        return true_size

    def local_search(S, max_iter=500):
        S = sorted(S)
        current_conflicts = count_conflicts(S)
        best_conflicts = current_conflicts
        best_S_local = list(S)
        n = len(S)
        universe = list(range(129))
        check_counter = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            idx = random.randrange(n)
            old_val = S[idx]
            
            while True:
                new_val = random.choice(universe)
                if new_val not in S:
                    break
            
            S[idx] = new_val
            new_conflicts = count_conflicts(S)
            
            if new_conflicts > best_conflicts:
                best_conflicts = new_conflicts
                best_S_local = list(S)
                check_counter += 1
                if check_counter % 5 == 0:
                    true_size = check_and_update(best_S_local)
                    if true_size == 3:
                        return best_S_local, best_conflicts
            elif new_conflicts >= current_conflicts:
                current_conflicts = new_conflicts
            else:
                S[idx] = old_val
                
        # Final check
        check_and_update(best_S_local)
        return best_S_local, best_conflicts

    # Initial candidates
    initial_sets = [
        list(range(12)),
        [i * 11 for i in range(12)],
        [i * 10 for i in range(12)],
        [i * 12 for i in range(12)],
    ]
    for _ in range(5):
        initial_sets.append(sorted(random.sample(range(129), 12)))
        
    for S0 in initial_sets:
        if time.time() > deadline:
            break
        check_and_update(S0)
        if best_true_size[0] == 3:
            break
        S, conf = local_search(S0, max_iter=300)
        
    # More random restarts
    while time.time() < deadline - 2.0:
        S0 = sorted(random.sample(range(129), 12))
        check_and_update(S0)
        if best_true_size[0] == 3:
            break
        S, conf = local_search(S0, max_iter=150)

    if best_S[0] is None:
        return list(range(12))
        
    return sorted(best_S[0])
