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

    # The condition "all subset products are distinct" is equivalent to:
    # The set of vectors {v(x) for x in S} is linearly independent over the integers?
    # No, that's not quite right. 
    # Let's analyze:
    # Product of subset A == Product of subset B
    # => sum_{x in A} v(x) == sum_{x in B} v(x)
    # => sum_{x in A} v(x) - sum_{x in B} v(x) == 0
    # This is a linear dependence relation with coefficients in {-1, 0, 1}.
    # Specifically, if we let C = A \ B and D = B \ A, then sum_{x in C} v(x) == sum_{x in D} v(x).
    # This means the set of vectors is not "sum-distinct".
    # Actually, the condition is stronger than linear independence. 
    # Linear independence over Q implies sum-distinctness?
    # If sum_{x in C} v(x) == sum_{x in D} v(x), then sum_{x in C} v(x) - sum_{x in D} v(x) = 0.
    # This is a linear dependence with coefficients in {-1, 0, 1}.
    # So if the vectors are linearly independent over Q, then no such non-trivial combination exists.
    # Therefore, linear independence over Q is a sufficient condition.
    # Is it necessary?
    # If the vectors are linearly dependent, there exists a non-trivial integer combination sum c_i v_i = 0.
    # This doesn't necessarily mean there is a subset with the same sum.
    # Example: v1 = (1,0), v2 = (0,1), v3 = (1,1). 
    # v1 + v2 = v3. So {v1, v2, v3} is linearly dependent.
    # Subset {1,2} product == Subset {3} product. So distinct subset products fails.
    # Example: v1 = (2,0), v2 = (1,1). 
    # 2*v1 - 2*v2 = (2,0) - (2,2) != 0. 
    # Wait, 2*v1 = (4,0), 2*v2 = (2,2). 
    # Is there a subset sum collision?
    # Subsets of {v1, v2}:
    # {} -> (0,0)
    # {v1} -> (2,0)
    # {v2} -> (1,1)
    # {v1, v2} -> (3,1)
    # All distinct.
    # So linear independence is sufficient but not necessary.
    # We want the largest set of vectors that is "sum-distinct".
    
    # However, checking sum-distinctness for a set of k vectors requires checking 2^k subsets.
    # For k=18, 2^18 = 262,144. This is fast.
    # But we need to find the MAXIMUM such set from 39 candidates (2..40).
    # This is an NP-hard problem (Maximum Sum-Distinct Subset).
    
    # Strategy:
    # 1. Generate a large pool of candidates.
    # 2. Use a greedy approach to build a large set.
    # 3. Use local search to improve it.
    # 4. Since the search space is small (39 candidates), we can use a smarter heuristic.
    
    # Let's try a different approach:
    # The condition is that the vectors are linearly independent over the field of integers? No.
    # Let's stick to the definition: All 2^k subset products are distinct.
    
    # Efficient check:
    # Use a set of hashes.
    # Since products can be huge, we use the prime exponent vector as the key.
    
    def is_sum_distinct(nums):
        k = len(nums)
        if k == 0:
            return True
        
        # Map each number to its vector
        vecs = [vectors[x] for x in nums]
        
        # We need to check if all subset sums of vecs are distinct.
        # We can use a dictionary to store the sum vector for each subset.
        # But storing tuples of length 12 for 2^k subsets might be slow.
        # Let's use a rolling hash or just the tuple.
        
        seen = set()
        # Empty subset
        seen.add(tuple([0]*num_primes))
        
        # DP: sum_vec[mask] = sum_vec[mask without lsb] + vecs[lsb]
        # To save memory, we can just compute and check on the fly.
        # But we need to store them to check for duplicates.
        
        # Optimization: If k is large, this is expensive.
        # But k is at most 18. 2^18 is ~262k.
        # Storing 262k tuples of length 12 is manageable.
        
        # Let's use a list of sums.
        sums = [None] * (1 << k)
        sums[0] = tuple([0]*num_primes)
        seen.add(sums[0])
        
        for mask in range(1, 1 << k):
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            prev_mask = mask ^ lsb
            
            prev_sum = sums[prev_mask]
            v = vecs[idx]
            
            # Compute new sum
            new_sum_list = list(prev_sum)
            for i in range(num_primes):
                new_sum_list[i] += v[i]
            
            new_sum = tuple(new_sum_list)
            
            if new_sum in seen:
                return False
            
            seen.add(new_sum)
            sums[mask] = new_sum
            
        return True

    # Greedy construction
    def greedy_construct(order):
        current = []
        for x in order:
            if len(current) >= 18:
                break
            new_set = current + [x]
            if is_sum_distinct(new_set):
                current = new_set
        return current

    # Local search
    def local_search(current, time_limit=5.0):
        best = list(current)
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            # Try adding
            for x in range(2, 41):
                if x not in best:
                    new_set = best + [x]
                    if is_sum_distinct(new_set):
                        best = new_set
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping
            for i in range(len(best)):
                for x in range(2, 41):
                    if x == best[i] or x in best:
                        continue
                    new_set = [y for j, y in enumerate(best) if j != i]
                    new_set.append(x)
                    if is_sum_distinct(new_set):
                        best = new_set
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try 2-swap
                found = False
                for i in range(len(best)):
                    for j in range(i+1, len(best)):
                        for x in range(2, 41):
                            if x in best: continue
                            for y in range(x+1, 41):
                                if y in best or y == x: continue
                                new_set = [z for k, z in enumerate(best) if k != i and k != j]
                                new_set.extend([x, y])
                                if is_sum_distinct(new_set):
                                    best = new_set
                                    found = True
                                    break
                            if found: break
                        if found: break
                    if found: break
                if found:
                    continue
                
                break
        
        return best

    # Try different orderings
    best_set = []
    best_size = 0
    
    # 1. Greedy with natural order
    res = greedy_construct(range(2, 41))
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
    
    # 2. Greedy with reversed order
    res = greedy_construct(range(40, 1, -1))
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
    
    # 3. Greedy with random orders
    import random
    for _ in range(20):
        if time.time() > deadline: break
        order = list(range(2, 41))
        random.shuffle(order)
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
    
    # 4. Local search from best
    if time.time() < deadline - 5:
        improved = local_search(best_set, time_limit=min(10, deadline - time.time() - 2))
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # 5. If still small, try more random restarts
    if best_size < 15 and time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline: break
            order = list(range(2, 41))
            random.shuffle(order)
            res = greedy_construct(order)
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
            # Quick local search
            if time.time() < deadline - 5:
                improved = local_search(res, time_limit=1.0)
                if len(improved) > best_size:
                    best_set = improved
                    best_size = len(best_set)
    
    # Ensure valid
    if not is_sum_distinct(best_set):
        # Fallback
        best_set = [2]
    
    return [int(x) for x in best_set]
