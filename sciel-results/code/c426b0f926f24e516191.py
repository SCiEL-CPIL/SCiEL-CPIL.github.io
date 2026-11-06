import random
import time

def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30, minimizing max element.
    Uses Randomized Local Search (Simulated Annealing) to find a valid set for a given M,
    then binary searches for the minimal M.
    """
    target_diffs = set(range(1, 31))
    
    def calculate_cost(S, M):
        """
        Returns a cost value. Lower is better.
        Components:
        1. Sidon violations (duplicate differences).
        2. Missing target differences (1..30).
        3. Size constraint (must be <= 32).
        4. Max element constraint (must be <= M).
        """
        if 0 not in S:
            return float('inf')
        
        S = sorted(list(set(S)))
        n = len(S)
        
        if n > 32:
            return float('inf')
            
        if max(S) > M:
            return float('inf')
            
        # Calculate differences
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(S[j] - S[i])
        
        # Check Sidon property
        if len(set(diffs)) != len(diffs):
            # Count duplicates
            from collections import Counter
            c = Counter(diffs)
            dup_count = sum(v - 1 for v in c.values() if v > 1)
            return 10000 + dup_count # Heavy penalty for non-Sidon
            
        diff_set = set(diffs)
        
        # Check coverage
        missing = target_diffs - diff_set
        if missing:
            return 5000 + len(missing) # Penalty for missing diffs
            
        # Valid solution
        # We want to minimize max(S) later, but for now, cost is 0.
        # To help search, we can prefer smaller max(S) within the same M?
        # No, we are fixing M. If valid, cost is 0.
        return 0

    def find_solution_for_M(M, max_iters=50000):
        """
        Uses Simulated Annealing to find a Sidon set in [0, M] covering 1..30.
        """
        # Initialize
        # We need at least enough elements to cover 30 diffs.
        # Min size N: N(N-1)/2 >= 30 => N=9 (36 pairs).
        # Let's start with size 9 or 10.
        min_size = 9
        max_size = 32
        
        # Random initialization
        # Ensure 0 is in the set.
        # Pick random size between min_size and min(max_size, M)
        # If M is small, size is limited.
        
        best_S = None
        best_cost = float('inf')
        
        # Try a few different initial sizes
        for target_size in range(min_size, min(max_size, M) + 1):
            if target_size > M: break # Can't have more elements than values in 1..M
            
            # Initial set: 0 + random distinct values from 1..M
            if M < target_size - 1:
                continue
                
            S_init = {0}
            # Randomly pick target_size - 1 elements from 1..M
            candidates = list(range(1, M + 1))
            random.shuffle(candidates)
            S_init.update(candidates[:target_size - 1])
            S = list(S_init)
            
            current_cost = calculate_cost(S, M)
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = list(S)
                if best_cost == 0:
                    return best_S
            
            # Simulated Annealing
            T = 100.0
            cooling = 0.995
            no_improve = 0
            
            for i in range(max_iters):
                if no_improve > 5000:
                    break
                if T < 0.1:
                    break
                    
                # Generate neighbor
                # Move types:
                # 1. Add random x in [1, M] not in S
                # 2. Remove random x in S (x != 0)
                # 3. Swap random x in S (x != 0) with random y in [1, M] not in S
                
                S_set = set(S)
                move_type = random.choice([1, 2, 3])
                
                new_S = list(S)
                
                if move_type == 1: # Add
                    if len(S) < 32:
                        # Pick random x
                        x = random.randint(1, M)
                        if x not in S_set:
                            new_S.append(x)
                        else:
                            continue # Skip if no change
                            
                elif move_type == 2: # Remove
                    if len(S) > 1:
                        # Pick random index not 0
                        non_zero_indices = [idx for idx, val in enumerate(S) if val != 0]
                        if non_zero_indices:
                            idx = random.choice(non_zero_indices)
                            new_S.pop(idx)
                        else:
                            continue
                            
                elif move_type == 3: # Swap
                    if len(S) > 1 and M > len(S):
                        non_zero_indices = [idx for idx, val in enumerate(S) if val != 0]
                        if non_zero_indices:
                            idx = random.choice(non_zero_indices)
                            old_val = S[idx]
                            new_val = random.randint(1, M)
                            if new_val not in S_set:
                                new_S[idx] = new_val
                            else:
                                continue
                            
                new_cost = calculate_cost(new_S, M)
                
                # Acceptance criterion
                if new_cost < current_cost:
                    S = new_S
                    current_cost = new_cost
                    no_improve = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_S = list(S)
                        if best_cost == 0:
                            return best_S
                else:
                    # Probabilistic acceptance
                    delta = new_cost - current_cost
                    if delta > 0:
                        prob = pow(2.71828, -delta / T)
                        if random.random() < prob:
                            S = new_S
                            current_cost = new_cost
                    else:
                        # Should not happen if cost is non-negative and we checked <
                        S = new_S
                        current_cost = new_cost
                    no_improve += 1
                
                T *= cooling
        
        return best_S if best_cost == 0 else None

    start_time = time.time()
    max_time = 115
    
    # Binary search for minimal M
    # Lower bound: 
    # Smallest Golomb ruler of length 30?
    # Known: Length 30 requires at least 9 marks?
    # Minimal max element for Sidon set covering 1..30.
    # Let's start from 40.
    lo = 40
    hi = 200
    
    best_solution = None
    
    # First, find any solution in [hi] to ensure we have a fallback
    # Actually, let's just iterate up from lo, it's faster if the answer is small.
    # But binary search is better if the answer is large.
    # Given N=9, max element is likely < 100.
    
    # Let's try linear search from lo, but with a step?
    # Or binary search.
    
    # To be safe, find a solution for a large M first to get a valid set (fallback)
    # Then try to minimize.
    
    # Let's just iterate M from lo to hi. If we find one, we return it?
    # No, we want MINIMAL M.
    # So we must check if M-1 has a solution.
    # Thus, we should iterate M from lo upwards and return the FIRST one found.
    # This is optimal for minimization if we are confident the search is complete.
    # RLS is not complete. It might miss a solution for M=50 but find one for M=51.
    # So we should run RLS for M, and if it fails, try M+1.
    # To improve reliability, run RLS multiple times for each M?
    
    # Strategy:
    # For M in range(lo, hi+1):
    #    if time > limit: break
    #    sol = find_solution_for_M(M)
    #    if sol: return sol
    
    # This might be slow if RLS takes long per M.
    # Let's limit iterations per M.
    
    for M in range(lo, hi + 1):
        if time.time() - start_time > max_time:
            break
            
        # Run search
        sol = find_solution_for_M(M, max_iters=20000)
        
        if sol is not None:
            # Verify
            if 0 in sol and len(set(sol)) == len(sol):
                diffs = set()
                n = len(sol)
                sol = sorted(sol)
                for i in range(n):
                    for j in range(i + 1, n):
                        diffs.add(sol[j] - sol[i])
                if all(d in diffs for d in range(1, 31)) and len(diffs) == n*(n-1)//2:
                    return sol

    # Fallback: Return a known valid set or generate one greedily
    # If we reach here, we failed to find a minimal one within time.
    # We should return the best valid one we found?
    # The problem says "Minimize the largest element".
    # If we didn't find the minimal one, we should return a valid one.
    # But we don't store the best valid one across M's if we return immediately on first success.
    # If we fail to find for small M, we go to larger M.
    # If we find for M=100, we return it. Is it minimal?
    # If M=99 had a solution but RLS missed it, we return 100. This is suboptimal but valid.
    # To improve, we could keep the best solution found so far?
    # No, if we find one at M=50, it's better than one at M=100.
    # So linear search from low to high is correct IF the search is reliable.
    
    # If no solution found in range, return a hardcoded one?
    # Let's generate a valid one using a simple greedy method as fallback.
    
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in set(diffs_list(sorted(a))):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                    a = trial
                    break
                y += 1
                
    return a
