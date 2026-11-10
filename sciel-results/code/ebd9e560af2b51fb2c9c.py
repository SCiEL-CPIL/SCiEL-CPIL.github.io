import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def is_valid_set(S):
        if not S:
            return True
        n = len(S)
        K = [set() for _ in range(n + 1)]
        K[0].add(0)
        for x in S:
            for k in range(n - 1, -1, -1):
                if K[k]:
                    new_sums = {s + x for s in K[k]}
                    K[k + 1].update(new_sums)
        seen = {}
        for k in range(n + 1):
            for s in K[k]:
                if s in seen and seen[s] != k:
                    return False
                seen[s] = k
        return True

    def count_conflicts(S):
        if not S:
            return 0
        n = len(S)
        K = [set() for _ in range(n + 1)]
        K[0].add(0)
        for x in S:
            for k in range(n - 1, -1, -1):
                if K[k]:
                    new_sums = {s + x for s in K[k]}
                    K[k + 1].update(new_sums)
        sum_to_card = {}
        conflicts = 0
        for k in range(n + 1):
            for s in K[k]:
                if s in sum_to_card:
                    if sum_to_card[s] != k:
                        conflicts += 1
                else:
                    sum_to_card[s] = k
        return conflicts

    # Build baseline using consecutive offsets
    # S = {B+1, ..., B+n} where B > n*(n+1)/2
    best_n = 0
    for n in range(1, 30):
        min_B = n * (n + 1) // 2 + 1
        if min_B + n <= 256:
            best_n = n
        else:
            break
    
    baseline_n = best_n
    min_B = baseline_n * (baseline_n + 1) // 2 + 1
    S_baseline = list(range(min_B, min_B + baseline_n))
    
    if not is_valid_set(S_baseline):
        S_baseline = [2**i for i in range(9)]
    
    current_S = S_baseline[:]
    used = set(current_S)
    
    random.seed(42)
    
    # Phase 1: Greedy add
    for _ in range(300):
        if time.time() > deadline - 10:
            break
        if len(current_S) >= 60:
            break
            
        best_x = None
        best_conf = float('inf')
        candidates = [x for x in range(1, 257) if x not in used]
        
        if len(candidates) > 80:
            candidates = random.sample(candidates, 80)
        
        for x in candidates:
            if time.time() > deadline - 10:
                break
            test_S = current_S + [x]
            c = count_conflicts(test_S)
            if c == 0:
                best_x = x
                break
            elif c < best_conf:
                best_conf = c
                best_x = x
        
        if best_x is not None and best_conf == 0:
            current_S.append(best_x)
            used.add(best_x)
    
    # Phase 2: Local search - swap operations
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        if len(current_S) >= 60:
            break
            
        # Try to swap an element out for a new one
        swapped = False
        for i in range(len(current_S)):
            if time.time() > deadline - 5:
                break
            old_x = current_S[i]
            old_used = set(current_S[:i] + current_S[i+1:])
            
            candidates = [x for x in range(1, 257) if x not in old_used]
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)
            
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                test_S = current_S[:i] + [x] + current_S[i+1:]
                if count_conflicts(test_S) == 0:
                    current_S[i] = x
                    used.discard(old_x)
                    used.add(x)
                    swapped = True
                    break
            if swapped:
                break
        
        if not swapped:
            break
    
    # Final validation
    if is_valid_set(current_S):
        return [int(x) for x in current_S]
    
    return [int(x) for x in S_baseline]

if __name__ == "__main__":
    s = construct()
    print(f"Set size: {len(s)}")
    print(f"Valid: {is_valid_set(s)}")
