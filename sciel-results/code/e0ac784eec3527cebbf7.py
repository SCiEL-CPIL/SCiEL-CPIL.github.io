import time
import math
from collections import deque

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of the divisors
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    LCM = 1
    for d in divisors:
        LCM = lcm(LCM, d)
    # LCM = 30030
    
    start_time = time.time()
    timeout = 80  # safety margin before 90s kill
    
    best_result = None
    best_range = float('inf')
    
    def hopcroft_karp(adj, n_left, n_right):
        """
        Hopcroft-Karp maximum bipartite matching.
        adj: list of lists, adj[i] = list of right nodes adjacent to left node i.
        Returns (matching_size, match_left, match_right)
        match_left[i] = j if left i matched to right j, else -1
        match_right[j] = i if right j matched to left i, else -1
        """
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = match_right[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == float('inf'):
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                mu = match_right[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, match_left, match_right
    
    def get_offsets_for_window(d, L, W):
        """Return list of offsets x-L where x is a multiple of d in [L, L+W]"""
        k = (L + d - 1) // d
        first = k * d
        if first > L + W:
            return []
        offsets = []
        x = first
        while x <= L + W:
            offsets.append(x - L)
            x += d
        return offsets
    
    def can_assign(L, W):
        """
        Check if we can assign distinct offsets to each divisor such that
        each divisor d_i has a multiple in [L, L+W].
        Uses Hopcroft-Karp bipartite matching.
        Returns (bool, assignment) where assignment[i] = offset for divisor i, or None.
        """
        # Build adjacency: left = divisors (0..n-1), right = unique offsets in [0, W]
        # But offsets can repeat across divisors, so we need to handle that.
        # Actually, the right side should be the set of distinct offset values [0, W].
        # Two divisors can't use the same offset value.
        
        # Left side: n divisors
        # Right side: offsets 0..W (W+1 possible values)
        n_right = W + 1
        
        adj = []
        for i in range(n):
            offsets = get_offsets_for_window(divisors[i], L, W)
            if not offsets:
                return False, None
            adj.append(offsets)
        
        matching, match_left, match_right = hopcroft_karp(adj, n, n_right)
        
        if matching < n:
            return False, None
        
        assignment = [match_left[i] for i in range(n)]
        return True, assignment
    
    # For each possible L in [1, LCM], binary search minimal W
    # But we need to be smart: only consider L values where all divisors have a multiple
    # in some reasonable window.
    
    # Precompute for each divisor d, the valid L ranges (L where [L, L+W] contains a multiple of d)
    # For a given W, L is valid for d if there exists k with k*d in [L, L+W],
    # i.e., L in [k*d - W, k*d].
    
    # Strategy: iterate W from small to large, find all valid L for that W,
    # then for each valid L, check if matching succeeds.
    
    # To speed up: for each W, use difference array to find L where ALL divisors are valid.
    
    for W in range(9, 200):
        if time.time() - start_time > timeout:
            break
        
        # Difference array to count how many divisors are "covered" at each L
        # L is in [1, LCM] (periodicity)
        diff = [0] * (LCM + 2)
        
        for d in divisors:
            k_min = 1
            k_max = (LCM + W) // d
            for k in range(k_min, k_max + 1):
                s = k * d - W
                e = k * d
                if s < 1:
                    s = 1
                if e > LCM:
                    e = LCM
                if s <= e:
                    diff[s] += 1
                    diff[e + 1] -= 1
        
        # Find L where count == n
        valid_Ls = []
        count = 0
        for L in range(1, LCM + 1):
            count += diff[L]
            if count == n:
                valid_Ls.append(L)
        
        if not valid_Ls:
            continue
        
        found = False
        for L in valid_Ls:
            if time.time() - start_time > timeout:
                break
            
            ok, assignment = can_assign(L, W)
            if ok:
                values = [L + assignment[i] for i in range(n)]
                # Verify
                if len(set(values)) == n and all(1 <= v <= 1000000 for v in values) and \
                   all(values[i] % divisors[i] == 0 for i in range(n)):
                    current_range = max(values) - min(values)
                    if current_range < best_range:
                        best_range = current_range
                        best_result = values
                    found = True
                    break
        
        if found:
            break
    
    # If no solution found, try a fallback: just use the divisors themselves
    if best_result is None:
        return divisors[:]
    
    return best_result
