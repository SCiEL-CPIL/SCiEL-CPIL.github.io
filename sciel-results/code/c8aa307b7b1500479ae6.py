import random
import time

def construct():
    initial = [1, 2, 5, 11]
    best_set = None
    best_size = 0
    
    start_time = time.time()
    time_limit = 115.0
    
    def compute_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def try_greedy(start_S):
        S = list(start_S)
        sums = compute_sums(S)
        
        # Use a set for S for O(1) lookup in candidates generation?
        # S is small, list is fine.
        candidates = [x for x in range(1, 285) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            # Check self sum
            if 2 * x in sums:
                valid = False
            else:
                # Check cross sums
                for y in S:
                    if x + y in sums:
                        valid = False
                        break
            if valid:
                S.append(x)
                sums.add(2 * x)
                for y in S[:-1]:
                    sums.add(x + y)
        
        return S

    # Phase 1: Randomized Greedy
    # We run until time limit or a fixed number of iterations
    iterations = 0
    while time.time() - start_time < time_limit * 0.5:
        S = try_greedy(initial)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        iterations += 1
        
    # Phase 2: Perturbation
    if best_set:
        while time.time() - start_time < time_limit:
            # Pick a random element to remove (not from initial set)
            removable = [x for x in best_set if x not in initial]
            if not removable:
                break
            x_remove = random.choice(removable)
            
            # Create perturbed start
            start_S = [x for x in best_set if x != x_remove]
            
            S = try_greedy(start_S)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            elif len(S) == best_size:
                # Keep the set if it's the same size, maybe helps diversity?
                # Actually, just keeping the first best is fine, 
                # but swapping might help in subsequent perturbations?
                # Let's stick to strict improvement to avoid thrashing.
                pass

    return best_set
