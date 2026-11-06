def construct():
    from math import gcd
    import itertools
    
    n = 256
    nums = list(range(2, n+1))
    
    # Build conflict graph: edge if lcm(a,b) <= 256
    # lcm(a,b) = a*b//gcd(a,b)
    conflict = {a: set() for a in nums}
    for i, a in enumerate(nums):
        for b in nums[i+1:]:
            g = gcd(a, b)
            l = a * b // g
            if l <= n:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Weight: 1/a
    weight = {a: 1.0/a for a in nums}
    
    # Greedy: sort by weight descending (i.e., a ascending), pick if no conflict with selected
    # But this might not be optimal. Let's try multiple orderings.
    
    best_set = set()
    best_sum = 0.0
    
    # Try greedy with different orderings
    # Ordering 1: by a ascending (highest weight first)
    # Ordering 2: by degree ascending (least conflicts first)
    # Ordering 3: by weight/degree
    
    def greedy(order):
        selected = set()
        for a in order:
            if not conflict[a] & selected:
                selected.add(a)
        return selected
    
    # Ordering by a ascending
    order1 = sorted(nums, key=lambda x: x)
    s1 = greedy(order1)
    sum1 = sum(weight[a] for a in s1)
    
    # Ordering by degree ascending
    order2 = sorted(nums, key=lambda x: (len(conflict[x]), x))
    s2 = greedy(order2)
    sum2 = sum(weight[a] for a in s2)
    
    # Ordering by weight/degree
    order3 = sorted(nums, key=lambda x: (-weight[x]/max(1, len(conflict[x])), x))
    s3 = greedy(order3)
    sum3 = sum(weight[a] for a in s3)
    
    candidates = [(s1, sum1), (s2, sum2), (s3, sum3)]
    best_set, best_sum = max(candidates, key=lambda x: x[1])
    
    # Local search: try to add a number if possible, or swap
    improved = True
    while improved:
        improved = False
        # Try adding any non-selected number
        for a in nums:
            if a not in best_set and not conflict[a] & best_set:
                best_set.add(a)
                best_sum += weight[a]
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one, add one or more
        for a in list(best_set):
            for b in nums:
                if b in best_set or b == a:
                    continue
                # Try removing a and adding b
                new_set = (best_set - {a}) | {b}
                # Check if b conflicts with any in best_set - {a}
                if conflict[b] & (best_set - {a}):
                    continue
                new_sum = best_sum - weight[a] + weight[b]
                if new_sum > best_sum:
                    best_set = new_set
                    best_sum = new_sum
                    improved = True
                    break
            if improved:
                break
    
    return list(best_set)
