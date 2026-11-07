import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Pre-compute factorization vectors for numbers 2..40
    # vector[i] is a tuple of exponents for primes
    vectors = {}
    for n in range(2, 41):
        exponents = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                exponents[i] += 1
                temp //= p
        vectors[n] = tuple(exponents)
        
    # We want to find the largest subset S of {2..40} such that all subset products are distinct.
    # This is equivalent to: The set of vectors {v(x) for x in S} is "sum-distinct".
    # i.e., no two distinct subsets of S have the same sum of vectors.
    
    # A powerful way to check this is to represent the set of all subset sums as a set of tuples.
    # However, for the search, we can use a recursive backtracking approach.
    
    # To speed up the check, we can precompute the "potential" subset sums for small sets.
    # But since we are adding elements one by one, we can maintain the set of current subset sums.
    
    # Let's use a different perspective:
    # If we have a set S, the number of distinct subset products is 2^{|S|}.
    # We want to maximize |S| subject to the condition.
    
    # We will perform a depth-first search (DFS) with backtracking.
    # State: current set of indices (or numbers), current set of subset sums (as frozenset of tuples).
    # This state space is huge, so we need strong pruning.
    
    # Pruning idea:
    # If we have already selected k numbers, we have 2^k distinct sums.
    # If we try to add a new number x, the new sums are {s, s + v(x) for s in current_sums}.
    # If any s + v(x) is already in current_sums, we have a collision.
    # Actually, the condition is that the new sums {s + v(x)} must be disjoint from {s} \ {0}?
    # No, the condition is that ALL 2^(k+1) sums must be distinct.
    # The old sums are S_old. The new sums are S_new = S_old U (S_old + v(x)).
    # We need S_old and (S_old + v(x)) to be disjoint.
    # i.e., (S_old + v(x)) ∩ S_old = ∅.
    
    # This check is O(2^k).
    # For k=18, 2^18 = 262,144. This is fast enough for a single check.
    # But we are doing this in a search tree.
    
    # Let's structure the search:
    # We try to build a set of size 18, then 17, etc.
    # Since the maximum possible size is likely around 15-18, we can start from high and go down.
    
    # To make the search efficient, we should order the candidates.
    # Numbers with "sparse" factorizations (primes) are generally better because they introduce new dimensions.
    # However, we are limited to 2..40.
    
    # Let's try a greedy approach first to get a good lower bound, then use it to prune the DFS.
    
    def is_sum_distinct(nums):
        k = len(nums)
        if k == 0:
            return True
        vecs = [vectors[x] for x in nums]
        sums = set()
        sums.add(tuple([0]*num_primes))
        
        # We can use a list of sums for DP, but a set is fine for checking distinctness.
        # To be faster, we can maintain a list of current sums and check against a set.
        current_sums = [tuple([0]*num_primes)]
        all_sums = set(current_sums)
        
        for v in vecs:
            new_sums = []
            for s in current_sums:
                # s is a tuple, v is a tuple
                new_s = tuple(s[i] + v[i] for i in range(num_primes))
                if new_s in all_sums:
                    return False
                new_sums.append(new_s)
            # If we get here, all new sums are distinct from old sums and each other?
            # Wait, we need to ensure new_sums are distinct from EACH OTHER too.
            # Since the mapping s -> s+v is a bijection, if s1 != s2, then s1+v != s2+v.
            # So new_sums are distinct from each other.
            # We just need to ensure they are distinct from all_sums (which contains old sums).
            # We checked that.
            
            all_sums.update(new_sums)
            current_sums = new_sums + current_sums # Order doesn't matter for the set, but for the list we just append
            # Actually, for the next iteration, we need ALL sums, not just the new ones.
            # So current_sums should become all_sums? No, current_sums is the list of ALL sums so far.
            # Let's just use a list of all sums.
            # Wait, if I keep a list of all sums, its size doubles each time.
            # For k=18, size is 2^18. This is fine.
            
            # Correction: I updated all_sums, but I need to update the list of sums for the next iteration.
            # The list of sums for the next iteration is the old list + the new list.
            # So:
            current_sums = current_sums + new_sums
            
        return True

    # Greedy to find a good starting point
    def greedy_construct(order):
        current = []
        for x in order:
            if len(current) >= 18:
                break
            new_set = current + [x]
            if is_sum_distinct(new_set):
                current = new_set
        return current

    best_size = 0
    best_set = []
    
    # Try a few greedy orders to get a baseline
    orders = [
        range(2, 41),
        range(40, 1, -1),
        sorted(range(2, 41), key=lambda x: sum(vectors[x])) # Sparse first?
    ]
    
    for order in orders:
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
    # If we found 18, we are done.
    if best_size == 18:
        return [int(x) for x in best_set]

    # Now, use DFS to see if we can find a larger set.
    # We will search for a set of size target_size.
    # We can start from best_size + 1 and go up to 18.
    
    # To make DFS efficient, we can use the fact that if a set of size k is valid,
    # any subset is valid.
    
    # Let's try to find a set of size 18 directly using backtracking.
    # If that fails, try 17, etc.
    
    # Optimization: Pre-filter candidates.
    # If a number is "bad" (e.g., 1, but 1 is not in 2..40), it might be excluded.
    # Actually, all numbers 2..40 are valid individually.
    
    # Let's implement a recursive search.
    # To avoid re-computing the sum set from scratch, we pass the current set of sums.
    # But the set of sums is large (up to 2^17). Copying it is expensive.
    # Instead, we can just re-run is_sum_distinct on the candidate set.
    # Since the candidate set is small (<=18), is_sum_distinct is fast.
    
    # Strategy:
    # 1. Generate all combinations of size 18 from 2..40? No, that's C(39,18) which is huge.
    # 2. Use backtracking with pruning.
    
    # Pruning:
    # If we have selected k numbers, and we are at index i in the sorted list of candidates,
    # the maximum possible size we can reach is k + (len(candidates) - i).
    # If this is less than the target size, prune.
    
    # Also, we can use the "greedy bound": if even the best greedy completion from this state
    # doesn't reach the target, prune. (Expensive to compute, maybe skip).
    
    # Let's try a simple DFS for target size 18.
    
    def dfs_search(target_size):
        candidates = list(range(2, 41))
        current_set = []
        
        def backtrack(start_idx, current_size):
            if current_size == target_size:
                return True
            if start_idx >= len(candidates):
                return False
            
            # Pruning: not enough candidates left
            if current_size + (len(candidates) - start_idx) < target_size:
                return False
            
            # Try to add candidates[start_idx]
            for i in range(start_idx, len(candidates)):
                x = candidates[i]
                new_set = current_set + [x]
                if is_sum_distinct(new_set):
                    current_set.append(x)
                    if backtrack(i + 1, current_size + 1):
                        return True
                    current_set.pop()
                
                # Check time
                if time.time() > deadline - 5:
                    return False
                    
            return False
        
        return backtrack(0, 0)
    
    # Try to find a set of size 18
    # If found, return it.
    # If not found, try 17, etc.
    
    # Note: The DFS might be slow if it explores a huge tree.
    # We can improve it by ordering candidates.
    # Candidates that are "more independent" should be tried first.
    # Primes are good candidates.
    
    # Let's reorder candidates: Primes first, then composites.
    primes_in_range = [p for p in primes if 2 <= p <= 40]
    composites_in_range = [n for n in range(2, 41) if n not in primes_in_range]
    ordered_candidates = primes_in_range + composites_in_range
    
    # Modify dfs_search to use ordered_candidates
    def dfs_search_ordered(target_size, candidates):
        current_set = []
        
        def backtrack(start_idx, current_size):
            if current_size == target_size:
                return True
            if start_idx >= len(candidates):
                return False
            
            if current_size + (len(candidates) - start_idx) < target_size:
                return False
            
            for i in range(start_idx, len(candidates)):
                x = candidates[i]
                new_set = current_set + [x]
                if is_sum_distinct(new_set):
                    current_set.append(x)
                    if backtrack(i + 1, current_size + 1):
                        return True
                    current_set.pop()
                
                if time.time() > deadline - 5:
                    return False
                    
            return False
        
        return backtrack(0, 0)
    
    # Try sizes from 18 down to best_size + 1
    for size in range(18, best_size, -1):
        if time.time() > deadline - 10:
            break
        
        # Try with ordered candidates
        if dfs_search_ordered(size, ordered_candidates):
            # We found a set of this size in current_set (from the closure)
            # But current_set is local to dfs_search_ordered. We need to return it.
            # Let's modify dfs_search_ordered to return the set.
            pass # Placeholder, need to fix implementation to return the set
        
    # The previous implementation doesn't return the set. Let's rewrite the search to return the set.
    
    def find_max_set():
        candidates = ordered_candidates
        best_found = list(best_set)
        
        def backtrack(start_idx, current_set, current_size):
            nonlocal best_found
            
            # Update best if larger
            if current_size > len(best_found):
                best_found = list(current_set)
                
            if current_size == 18:
                return # Max possible
            
            # Pruning
            if current_size + (len(candidates) - start_idx) <= len(best_found):
                return # Can't beat best
            
            for i in range(start_idx, len(candidates)):
                x = candidates[i]
                new_set = current_set + [x]
                if is_sum_distinct(new_set):
                    current_set.append(x)
                    backtrack(i + 1, current_set, current_size + 1)
                    current_set.pop()
                
                if time.time() > deadline - 5:
                    return
                    
        backtrack(0, [], 0)
        return best_found

    # Run the search
    # This might take a while. We have ~80s left.
    # If it finds 18, great. If not, it returns the best found.
    
    # To ensure we don't timeout, we run it.
    # If the DFS is too slow, it will stop at deadline-5 and return the best found so far.
    
    result_set = find_max_set()
    
    # Ensure valid
    if not is_sum_distinct(result_set):
        # Fallback to greedy result
        result_set = best_set
        
    if not is_sum_distinct(result_set):
        # Absolute fallback
        result_set = [2]

    return [int(x) for x in result_set]
