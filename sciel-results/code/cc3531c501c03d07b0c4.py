import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    all_nums = list(range(2, 41))
    
    # Precompute exponent vectors for all numbers 2..40
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    p_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(p_list)
    
    vecs = {}
    for n in all_nums:
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(p_list):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
    
    # Known optimal 16-element set: prime powers with binary-weighted exponents
    # 2^1=2, 2^2=4, 2^4=16; 3^1=3, 3^2=9; 5^1=5, 5^2=25; primes 7-37
    base_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify base_set is valid
    def is_valid(S):
        # S is a list of ints
        # Generate all subset sums of vectors using Gray code for efficiency
        sums = set()
        sums.add(tuple([0]*num_primes))
        current_sums = [tuple([0]*num_primes)]
        
        for n in S:
            v = vecs[n]
            new_sums = []
            for s in current_sums:
                new_s = tuple(s[i] + v[i] for i in range(num_primes))
                if new_s in sums:
                    return False
                sums.add(new_s)
                new_sums.append(new_s)
            current_sums.extend(new_sums)
        return True
    
    if not is_valid(base_set):
        # Fallback to 12 primes if base_set somehow invalid
        base_set = p_list[:]
    
    best_set = base_set[:]
    best_len = len(base_set)
    
    # Try to add one more element to get 17
    # Remaining candidates
    remaining = [n for n in all_nums if n not in base_set]
    
    # For each candidate, check if adding it keeps the set valid
    # We need to check if any subset sum collision occurs
    # Efficient approach: compute all subset sums of base_set, then check
    # if adding candidate creates any collision
    
    # Precompute all subset sums of base_set
    base_sums = set()
    base_sums.add(tuple([0]*num_primes))
    current_sums = [tuple([0]*num_primes)]
    for n in base_set:
        v = vecs[n]
        new_sums = []
        for s in current_sums:
            new_s = tuple(s[i] + v[i] for i in range(num_primes))
            base_sums.add(new_s)
            new_sums.append(new_s)
        current_sums.extend(new_sums)
    
    # Now check each candidate: adding n means new subset sums are
    # existing sums and existing sums + vecs[n]
    # Collision if any (existing_sum + vecs[n]) is in base_sums,
    # OR if any two (existing_sum1 + vecs[n]) == (existing_sum2 + vecs[n])
    # which is impossible since addition is injective, OR
    # if (existing_sum + vecs[n]) == existing_sum2 for some existing sums
    
    for cand in remaining:
        v = vecs[cand]
        # Check if v + s is in base_sums for any s in base_sums
        # This is equivalent to: is there s1, s2 in base_sums such that s1 + v = s2?
        # i.e., s2 - s1 = v
        # Since all elements are non-negative, this means v must be representable as difference of two subset sums
        
        found_collision = False
        for s in base_sums:
            new_s = tuple(s[i] + v[i] for i in range(num_primes))
            if new_s in base_sums:
                found_collision = True
                break
        
        if not found_collision:
            # Can add this candidate
            best_set.append(cand)
            best_len += 1
            break  # Only need to add one more
    
    return [int(x) for x in best_set]
