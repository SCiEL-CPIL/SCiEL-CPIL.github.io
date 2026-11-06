def construct():
    from math import gcd
    import random
    
    n = 256
    nums = list(range(2, n+1))
    
    # Build conflict graph: edge if lcm(a,b) <= 256
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
    
    def greedy(order):
        selected = set()
        for a in order:
            if not conflict[a] & selected:
                selected.add(a)
        return selected

    # Generate multiple greedy orders to find a good starting point
    orders = []
    # 1. Weight descending (a ascending)
    orders.append(sorted(nums, key=lambda x: x))
    # 2. Weight/degree
    orders.append(sorted(nums, key=lambda x: (-weight[x]/(1+len(conflict[x])), x)))
    # 3. Degree ascending
    orders.append(sorted(nums, key=lambda x: (len(conflict[x]), x)))
    
    # 4. Randomized weighted orders
    random.seed(42)
    for _ in range(10):
        perturbed_weight = {a: weight[a] * (0.8 + 0.4*random.random()) for a in nums}
        orders.append(sorted(nums, key=lambda x: -perturbed_weight[x]))
        
    best_set = set()
    best_sum = 0.0
    
    for order in orders:
        s = greedy(order)
        s_sum = sum(weight[a] for a in s)
        if s_sum > best_sum:
            best_sum = s_sum
            best_set = s
            
    # Local Search: Steepest Descent 1-for-1 and Add
    improved = True
    while improved:
        improved = False
        
        # 1. Try to add any compatible node
        for a in nums:
            if a not in best_set and not conflict[a] & best_set:
                best_set.add(a)
                best_sum += weight[a]
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Steepest Descent 1-for-1 Swap
        best_swap = None
        best_gain = 0
        for a in best_set:
            for b in nums:
                if b in best_set or b == a:
                    continue
                # Check if b is compatible with all other nodes in best_set except a
                if not conflict[b] & (best_set - {a}):
                    gain = weight[b] - weight[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (a, b)
        
        if best_swap:
            a, b = best_swap
            best_set.remove(a)
            best_set.add(b)
            best_sum += best_gain
            improved = True
            
    return list(best_set)
