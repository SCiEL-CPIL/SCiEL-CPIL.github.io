# EVOLVE-BLOCK-START
import random
from itertools import combinations_with_replacement

def construct():
    n = 4096
    
    def can_add(S, T, x):
        """
        Check if x can be added to S without creating duplicate triple sums.
        S: list of current elements.
        T: set of current triple sums.
        x: candidate element.
        Returns: (bool, new_T) where new_T is the updated set of sums if valid, else None.
        """
        new_sums_list = []
        
        # 1. x, x, x
        new_sums_list.append(3 * x)
        
        # 2. x, x, s
        for s in S:
            new_sums_list.append(2 * x + s)
            
        # 3. x, s1, s2
        k = len(S)
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                new_sums_list.append(x + s1 + s2)
                
        # Check for internal collisions
        if len(new_sums_list) != len(set(new_sums_list)):
            return False, None
            
        new_sums_set = set(new_sums_list)
        
        # Check collision with existing sums
        if new_sums_set & T:
            return False, None
            
        # Valid
        new_T = T.union(new_sums_set)
        return True, new_T

    def build_set(skeleton, candidates, max_size=40):
        S = list(skeleton)
        # Compute initial sums
        T = set()
        for t in combinations_with_replacement(S, 3):
            T.add(sum(t))
            
        for x in candidates:
            if x in S:
                continue
            valid, new_T = can_add(S, T, x)
            if valid:
                S.append(x)
                T = new_T
                if len(S) >= max_size:
                    break
                    
        return S

    # Define skeletons
    # Skeleton 1: Powers of 4
    skeleton1 = [1, 4, 16, 64, 256, 1024, 4096]
    
    # Candidates: All numbers 1..4096
    all_candidates = list(range(1, n + 1))
    
    best_set = []
    
    # Strategy 1: Skeleton1 + Ascending
    cands_asc = [x for x in all_candidates if x not in skeleton1]
    set1 = build_set(skeleton1, cands_asc)
    if len(set1) > len(best_set):
        best_set = set1
        
    # Strategy 2: Skeleton1 + Descending
    cands_desc = [x for x in reversed(all_candidates) if x not in skeleton1]
    set2 = build_set(skeleton1, cands_desc)
    if len(set2) > len(best_set):
        best_set = set2
        
    # Strategy 3: Randomized trials
    # We will run multiple random trials to find a larger set.
    # The check is fast, so we can run many trials.
    import time
    start_time = time.time()
    time_limit = 55.0 # Leave some buffer for overhead
    
    random.seed(42)
    trial_count = 0
    while time.time() - start_time < time_limit:
        trial_count += 1
        # Random candidate order
        cands_rand = all_candidates[:]
        random.shuffle(cands_rand)
        
        # Try with skeleton1
        cands_rand2 = [x for x in cands_rand if x not in skeleton1]
        s_rand = build_set(skeleton1, cands_rand2)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            if len(best_set) >= 40:
                break
                
        # Try with empty set (pure random greedy)
        s_rand_empty = build_set([], cands_rand)
        if len(s_rand_empty) > len(best_set):
            best_set = s_rand_empty
            if len(best_set) >= 40:
                break

    return best_set
# EVOLVE-BLOCK-END
