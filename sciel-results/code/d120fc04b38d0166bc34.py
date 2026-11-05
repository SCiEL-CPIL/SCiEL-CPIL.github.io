import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        # Check if adding x to S_set keeps it sum-free
        # Conflicts:
        # 1. a + b = x for a, b in S_set
        # 2. a + x = b for a, b in S_set
        # 3. x + x = c for c in S_set
        
        # 1. a + b = x => b = x - a
        for a in S_set:
            if (x - a) in S_set:
                return False
        
        # 2. a + x = b => b = a + x
        for a in S_set:
            if (a + x) in S_set:
                return False
        
        # 3. x + x = c
        if (2 * x) in S_set:
            return False
            
        return True

    def greedy_improve(S, time_limit):
        S_set = set(S)
        remaining = [x for x in S if x not in S_set]
        
        start_time = time.time()
        while True:
            if time.time() - start_time > time_limit:
                break
            
            added_any = False
            # Shuffle to avoid order bias
            random.shuffle(remaining)
            
            for x in remaining[:]:  # Iterate over a copy
                if x in S_set:
                    continue
                if can_add(x, S_set):
                    S_set.add(x)
                    remaining.remove(x)
                    added_any = True
            
            if not added_any:
                break
        
        return list(S_set)

    def get_suffix(S):
        # S is sorted
        n = len(S)
        # Find largest k such that S[n-k] + S[n-k+1] > S[n-1]
        # Note: indices 0 to n-1
        # If k=1, set is {S[n-1]}. 2*S[n-1] > S[n-1] always true for non-zero.
        # We want the largest subset of the suffix.
        # Let the subset be S[n-k : n].
        # Condition: min(S[n-k:]) + min(S[n-k:]) > max(S[n-k:])
        # Since sorted, min is S[n-k], max is S[n-1].
        # So S[n-k] + S[n-k] > S[n-1].
        
        best_k = 1
        for k in range(1, n + 1):
            # Subset is S[n-k : n]
            # Smallest is S[n-k]
            # Largest is S[n-1]
            if 2 * S[n - k] > S[n - 1]:
                best_k = k
            else:
                # Since S is sorted, if this fails, smaller k (larger index, smaller values) might fail too?
                # Wait. As k increases, n-k decreases, so S[n-k] decreases.
                # So 2*S[n-k] decreases.
                # So if it fails for some k, it will fail for all larger k.
                break
        
        return S[n - best_k : n]

    def solve_instance(S, time_limit):
        # 1. Odds
        odds = [x for x in S if x % 2 != 0]
        
        # 2. Suffix
        suffix = get_suffix(S)
        
        # 3. Greedy (sorted)
        greedy_init = []
        S_set = set()
        for x in S:
            if can_add(x, S_set):
                S_set.add(x)
                greedy_init.append(x)
        
        # Pick best initial
        candidates = [odds, suffix, greedy_init]
        best_init = max(candidates, key=len)
        
        # Improve
        improved = greedy_improve(best_init, time_limit)
        
        return improved

    # Instances
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    total_time = 120
    start = time.time()
    
    for S in instances:
        # Allocate time based on size? Or equal?
        # 240 is the hardest.
        # Let's give 20s to 60, 30s to 120, 60s to 240.
        # Or just run until deadline.
        elapsed = time.time() - start
        remaining_time = total_time - elapsed - 5 # 5s safety margin
        
        if remaining_time <= 0:
            # Fallback to odds
            results.append([x for x in S if x % 2 != 0])
            continue
            
        # Run solver
        res = solve_instance(S, remaining_time)
        results.append(res)
        
    return results

# EVOLVE-BLOCK-END
