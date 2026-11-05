import time

def construct():
    start_time = time.time()
    
    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        if not subset:
            return True
        s_set = set(subset)
        n = len(subset)
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                b = subset[j]
                c = a + b
                if c in s_set:
                    return False
        return True

    def best_k_subset(S):
        candidates = set([0])
        for x in S:
            candidates.add(abs(x))
        
        best_subset = []
        # Sort candidates to try smallest K first (largest subset)
        for K in sorted(candidates):
            if time.time() - start_time > 115:
                break
            S_K = [x for x in S if abs(x) > K]
            if is_sum_free(S_K):
                return S_K
        return []

    def greedy_max_abs(S):
        # Sort by absolute value descending
        S_sorted = sorted(S, key=lambda x: abs(x), reverse=True)
        C = []
        C_set = set()
        
        for x in S_sorted:
            # Check if adding x to C creates a violation
            # Violations must involve x
            valid = True
            
            # 1. Is x the sum of two elements in C?
            # a + b = x => b = x - a
            # We can check if there exists a in C such that x-a is in C
            # To avoid O(N^2) here, we can just iterate C
            for a in C:
                if (x - a) in C_set:
                    valid = False
                    break
            
            if valid:
                # 2. Is x an addend?
                # x + a = b => b = x + a
                # Check if x + a is in C for any a in C
                for a in C:
                    if (x + a) in C_set:
                        valid = False
                        break
            
            if valid:
                # 3. Is 2x in C?
                if (2 * x) in C_set:
                    valid = False
            
            # Note: If valid is still True, we add x
            if valid:
                C.append(x)
                C_set.add(x)
                
        return C

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Candidate 1: Odds
        S_odd = [x for x in S if x % 2 != 0]
        
        # Candidate 2: Best S_K
        S_best_k = best_k_subset(S)
        
        # Candidate 3: Upper half
        M = max(S)
        S_upper = [x for x in S if x > M / 2.0]
        
        # Candidate 4: Lower half
        m = min(S)
        S_lower = [x for x in S if x < m / 2.0]
        
        # Candidate 5: Greedy Max Abs
        S_greedy = greedy_max_abs(S)
        
        candidates = [S_odd, S_best_k, S_upper, S_lower, S_greedy]
        
        # Select the largest valid sum-free subset
        best = max(candidates, key=len)
        
        # Final check for safety
        if not is_sum_free(best):
             best = S_odd

        results.append(best)
        
    return results
