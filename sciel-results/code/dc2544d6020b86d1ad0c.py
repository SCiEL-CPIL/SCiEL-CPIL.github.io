import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def get_smooth_numbers(primes, limit):
        smooths = set([1])
        nums = [1]
        for p in primes:
            new_nums = []
            for x in nums:
                y = x
                while y * p <= limit:
                    y *= p
                    if y not in smooths:
                        smooths.add(y)
                        new_nums.append(y)
            nums.extend(new_nums)
        return smooths

    def find_solution(primes):
        if time.time() > deadline:
            return None
        
        smooth_set = get_smooth_numbers(primes, 20000)
        
        # Build partners
        partners = [[] for _ in range(10001)]
        for s in smooth_set:
            if s < 3:
                continue
            min_u = max(1, s - 10000)
            max_u = min(10000, (s - 1) // 2)
            for u in range(min_u, max_u + 1):
                v = s - u
                if v <= 10000 and v > u:
                    partners[u].append(v)
        
        # Convert to sets for O(1) lookup
        partners_sets = [set(p) for p in partners]
        
        def backtrack(current_list, last_val):
            if len(current_list) == 12:
                return current_list[:]
            if time.time() > deadline:
                return None
            
            # Find next candidate v > last_val
            # v must be in partners_sets[u] for all u in current_list
            
            # Use the last element's partners as the base set to iterate
            # This is a heuristic; usually the last element constrains the next one most directly in terms of range
            last_p = current_list[-1]
            
            # We need to check membership in all previous partners sets
            # To optimize, we can intersect the sets? 
            # But sets change size. 
            # Given the sparsity, iterating over partners_sets[last_p] is fine.
            
            for v in partners_sets[last_p]:
                if v <= last_val:
                    continue
                
                # Check if v is connected to all other elements
                valid = True
                for u in current_list[:-1]:
                    if v not in partners_sets[u]:
                        valid = False
                        break
                
                if valid:
                    res = backtrack(current_list + [v], v)
                    if res:
                        return res
            return None

        # Try starting with each a1
        # We can skip a1 if it has no partners
        for a1 in range(1, 10001):
            if time.time() > deadline:
                return None
            if not partners[a1]:
                continue
            res = backtrack([a1], a1)
            if res:
                return res
        return None

    # Prime sets to try
    # We know 2 is almost always required.
    prime_sets = [
        [2],
        [2, 3],
        [2, 5],
        [2, 7],
        [2, 11],
        [2, 3, 5],
        [2, 3, 7],
        [2, 5, 7],
        [2, 3, 5, 7]
    ]

    for p_set in prime_sets:
        sol = find_solution(p_set)
        if sol:
            return sol

    return list(range(1, 13))
