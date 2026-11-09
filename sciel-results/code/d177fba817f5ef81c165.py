def construct():
    import random
    from math import gcd

    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def can_add(S_list, x):
        """Check if x can be added to S_list."""
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
                    return False
        return True

    def greedy(order):
        selected = []
        sel_set = set()
        for x in order:
            if x not in sel_set and can_add(selected, x):
                selected.append(x)
                sel_set.add(x)
        return selected

    best = []
    
    # Strategy 1: Greedy with various orderings
    orders = [
        nums,
        nums[::-1],
        sorted(nums, key=lambda x: len([d for d in range(1, x+1) if x%d==0])),
        sorted(nums, key=lambda x: x),
        sorted(nums, key=lambda x: -x)
    ]
    
    def sf_kernel(n):
        res = 1
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                res *= d
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            res *= temp
        return res

    orders.append(sorted(nums, key=sf_kernel))
    orders.append(sorted(nums, key=lambda x: -sf_kernel(x)))

    for order in orders:
        r = greedy(order)
        if len(r) > len(best):
            best = r

    # Strategy 2: Random restarts with Local Search
    random.seed(42)
    for _ in range(200):
        shuffled = nums[:]
        random.shuffle(shuffled)
        current = greedy(shuffled)
        cur_set = set(current)
        
        # Local Search
        for _iter in range(100):
            improved = False
            
            # Try to add
            for x in nums:
                if x not in cur_set and can_add(current, x):
                    current.append(x)
                    cur_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue

            # Try to swap
            current_list = current[:]
            random.shuffle(current_list)
            for u in current_list:
                S_minus_u = [z for z in current if z != u]
                # Try to find a v that can replace u
                for v in nums:
                    if v in cur_set:
                        continue
                    if can_add(S_minus_u, v):
                        current = S_minus_u + [v]
                        cur_set.remove(u)
                        cur_set.add(v)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if len(current) > len(best):
            best = current

    return best
