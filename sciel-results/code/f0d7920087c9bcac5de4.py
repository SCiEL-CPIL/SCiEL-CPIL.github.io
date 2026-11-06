import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    S = set(range(1, 101))
    
    def get_ways_and_subset(elements, target):
        """
        Returns (count, subset).
        """
        elements_list = list(elements)
        n = len(elements_list)
        if n == 0:
            return 0, None
        
        # 1. Count ways
        dp = [0] * (target + 1)
        dp[0] = 1
        for x in elements_list:
            for j in range(target, x - 1, -1):
                if dp[j - x] > 0:
                    dp[j] += dp[j - x]
        
        count = dp[target]
        if count == 0:
            return 0, None
        
        # 2. Find one subset (reconstruction)
        # Use table for backtracking
        table = [[False] * (target + 1) for _ in range(n + 1)]
        table[0][0] = True
        
        for i in range(1, n + 1):
            x = elements_list[i-1]
            for j in range(target + 1):
                if table[i-1][j]:
                    table[i][j] = True
                elif j >= x and table[i-1][j-x]:
                    table[i][j] = True
        
        subset = []
        j = target
        for i in range(n, 0, -1):
            x = elements_list[i-1]
            if j >= x and table[i-1][j-x]:
                subset.append(x)
                j -= x
                
        return count, subset

    # Main optimization loop
    for _ in range(100):
        if time.time() > deadline:
            break
            
        count, subset = get_ways_and_subset(S, 200)
        
        if count == 0:
            return S
            
        # Greedy removal: pick x in subset that minimizes remaining ways
        candidates = list(subset)
        best_x = None
        min_ways = float('inf')
        
        for x in candidates:
            new_S = S - {x}
            ways_x, _ = get_ways_and_subset(new_S, 200)
            if ways_x < min_ways:
                min_ways = ways_x
                best_x = x
                
        if best_x is not None:
            S.remove(best_x)
            
    # Fallback: ensure validity
    while True:
        if time.time() > deadline - 0.5:
             # Force termination with a valid set
             # Just remove largest until valid
             pass
             
        count, subset = get_ways_and_subset(S, 200)
        if count == 0:
            return S
        x = max(subset)
        S.remove(x)
        
    return S
