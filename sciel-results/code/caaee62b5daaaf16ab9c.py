import itertools

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # We search for the minimal range R
    # We iterate over possible min values and check if a valid set exists
    # within [min_val, min_val + R]
    
    # Since the optimal range is small (around 19), we can just iterate R from 1 upwards
    # and for each R, iterate min_val from 1 to some limit (e.g., 1000)
    
    def has_valid_assignment(min_val, max_val):
        # Build bipartite graph
        # Left: 0..n-1 (divisors)
        # Right: integers in [min_val, max_val]
        
        candidates = list(range(min_val, max_val + 1))
        
        # adj[i] is list of indices in candidates that are divisible by divisors[i]
        adj = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            for j, val in enumerate(candidates):
                if val % d == 0:
                    adj[i].append(j)
        
        # Check if perfect matching exists using DFS (Hungarian-like for small graphs)
        # We use a simple backtracking or Hopcroft-Karp
        # Given small size, simple DFS is fine
        
        match = [-1] * len(candidates) # match[j] = i
        
        def dfs(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match[j] == -1 or dfs(match[j], seen):
                    match[j] = i
                    return True
            return False
        
        for i in range(n):
            seen = [False] * len(candidates)
            if not dfs(i, seen):
                return None # No matching
        
        # If matching exists, retrieve the values
        result = [0] * n
        for j in range(len(candidates)):
            if match[j] != -1:
                result[match[j]] = candidates[j]
        
        # Check distinctness (guaranteed by matching on distinct values)
        # Check types
        return result

    # Search for minimal range
    best_result = None
    best_range = float('inf')
    
    # We know range is small. Let's search up to 50.
    for R in range(1, 50):
        found = False
        # min_val can be small. Let's check 1 to 1000.
        for min_val in range(1, 1000):
            max_val = min_val + R
            res = has_valid_assignment(min_val, max_val)
            if res is not None:
                # Verify distinctness and divisibility
                if len(set(res)) == n:
                    if all(res[i] % divisors[i] == 0 for i in range(n)):
                        if all(isinstance(x, int) for x in res):
                            best_result = res
                            best_range = R
                            found = True
                            break
        if found:
            break
            
    if best_result is None:
        # Fallback to seed solution
        return divisors[:]
        
    return best_result
