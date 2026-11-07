import time
import math

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least L > 0 such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    N = 12
    PERIOD = 27720  # LCM(1..12)
    
    # We search over all residues r in 0..PERIOD-1
    # For each r, we find min L such that matching exists.
    # We want to maximize this min L.
    
    def has_matching(r, L):
        # r is the residue, interval is (r, r+L]
        # Left nodes: 1..N
        # Right nodes: indices 0..L-1 corresponding to values r+1 .. r+L
        
        # match_to[j] = i if right node j is matched to left node i, else -1
        match_to = [-1] * L
        
        # Precompute candidates for each i?
        # For a fixed r and L, the candidates for i are indices j in 0..L-1
        # such that i divides (r + 1 + j).
        
        # To speed up DFS, we can precompute the list of valid indices for each i.
        # candidates[i] = list of j such that i | (r + 1 + j)
        
        candidates = [[] for _ in range(N + 1)]
        
        # Iterate through the interval
        # val = r + 1 + j
        # j = val - (r + 1)
        
        # Instead of iterating j, iterate val from r+1 to r+L
        # For each val, find divisors i in 1..N
        # This might be faster than iterating i and checking divisibility for all j?
        # N=12, L<=200.
        # Method 1: For each i, find multiples of i in (r, r+L].
        #   first_mult = r + (i - r % i) % i
        #   if first_mult <= r: first_mult += i
        #   for val in range(first_mult, r + L + 1, i):
        #       j = val - (r + 1)
        #       candidates[i].append(j)
        
        for i in range(1, N + 1):
            # First multiple of i strictly greater than r
            rem = r % i
            if rem == 0:
                first = r + i
            else:
                first = r + (i - rem)
            
            # Add all multiples in (r, r+L]
            # val, val+i, ... <= r+L
            for val in range(first, r + L + 1, i):
                j = val - (r + 1)
                candidates[i].append(j)
        
        # Kuhn's algorithm
        # visited[j] for right nodes
        # We need a visited array for the DFS on the left side?
        # Standard Kuhn:
        # for i in 1..N:
        #   visited = [False] * L
        #   if not dfs(i, visited): return False
        # return True
        
        # dfs(i, visited):
        #   for j in candidates[i]:
        #     if not visited[j]:
        #       visited[j] = True
        #       if match_to[j] == -1 or dfs(match_to[j], visited):
        #         match_to[j] = i
        #         return True
        #   return False
        
        def dfs(i, visited):
            for j in candidates[i]:
                if not visited[j]:
                    visited[j] = True
                    if match_to[j] == -1 or dfs(match_to[j], visited):
                        match_to[j] = i
                        return True
            return False
        
        for i in range(1, N + 1):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_m = 0
    max_L = 0
    
    start_time = time.time()
    # 90s limit, use 80s for safety
    deadline = start_time + 80.0
    
    # We can iterate r in any order.
    # To potentially find a high L earlier, maybe randomize?
    # But exhaustive is required for certainty unless we find 144.
    # The max possible L is bounded. The problem states "at most 144".
    # If we find 144, we can stop.
    
    for r in range(PERIOD):
        if time.time() > deadline:
            break
            
        # Find min L for this r
        # L must be at least N=12? No, but distinct b_i requires at least N numbers.
        # Actually, if L < N, it's impossible to have N distinct integers.
        # So L >= 12.
        found = False
        for L in range(12, 200):
            if has_matching(r, L):
                if L > max_L:
                    max_L = L
                    best_m = r
                    if max_L == 144:
                        return best_m
                found = True
                break
        # If no L found up to 200, something is wrong, but max is 144.
        
    return best_m
