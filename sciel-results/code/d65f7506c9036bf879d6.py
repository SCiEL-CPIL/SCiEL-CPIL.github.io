import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def get_subset_products(nums):
        prods = {1}
        for x in nums:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            prods |= new_prods
        return prods
        
    def is_valid(nums):
        if len(nums) != len(set(nums)):
            return False
        if any(x < 2 or x > 40 for x in nums):
            return False
        prods = get_subset_products(nums)
        return len(prods) == 2 ** len(nums)

    # Known optimal set of size 16
    candidate = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if is_valid(candidate):
        return candidate

    # Fallback search if candidate fails (should not happen)
    def can_add(S, c, prods_S):
        for p in prods_S:
            if p * c in prods_S:
                return False
        return True
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [4, 8, 9, 12, 16, 18, 20, 24, 25, 27, 28, 32, 36, 40]
    
    best = primes[:]
    found_18 = [False]
    
    def backtrack(current_set, current_prods, candidates):
        if time.time() > deadline:
            return
        if found_18[0]:
            return
        if len(current_set) > len(best):
            best[:] = current_set[:]
        if len(current_set) >= 18:
            found_18[0] = True
            return
        if not candidates:
            return
        
        for i, c in enumerate(candidates):
            if found_18[0] or time.time() > deadline:
                return
            if can_add(current_set, c, current_prods):
                new_set = current_set + [c]
                new_prods = get_subset_products(new_set)
                backtrack(new_set, new_prods, candidates[i+1:])
    
    backtrack(primes, get_subset_products(primes), composites)
    
    return best
