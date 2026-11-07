import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s to be safe against 60s limit
    
    n = 12
    max_m = 1000000
    period = 27720  # LCM(1, 2, ..., 12)
    
    best_L = 0
    best_r = 0
    
    # We search for the best residue r in [0, period)
    # f(12, m) depends only on m % period
    for r in range(period):
        # Check deadline
        if time.time() > deadline:
            break
        
        # If we already found the theoretical maximum, we can stop
        if best_L == 144:
            break
        
        found_L = None
        
        # L must be at least n to fit n distinct numbers
        # Max L is 144, so we search up to 144
        for L in range(n, 145):
            # Build candidate lists for each i in 1..12
            # candidates[i] holds x in (r, r+L] such that i divides x
            candidates = [[] for _ in range(n + 1)]
            
            for i in range(1, n + 1):
                # First multiple of i strictly greater than r
                k = r // i + 1
                x = k * i
                while x <= r + L:
                    candidates[i].append(x)
                    x += i
            
            # Quick check: if any i has no candidates, skip
            # Also, if total candidates < n, skip
            if any(len(candidates[i]) == 0 for i in range(1, n + 1)):
                continue
                
            # Order indices by number of candidates (ascending) for better pruning
            order = sorted(range(1, n + 1), key=lambda i: len(candidates[i]))
            
            # Backtracking search for perfect matching
            # assigned: list of chosen numbers
            def dfs(pos, assigned):
                if pos == n:
                    return True
                i = order[pos]
                for x in candidates[i]:
                    # Check if x is already used
                    # Since n=12, linear search is fast
                    if x not in assigned:
                        assigned.append(x)
                        if dfs(pos + 1, assigned):
                            return True
                        assigned.pop()
                return False
            
            if dfs(0, []):
                found_L = L
                break
        
        if found_L is not None:
            if found_L > best_L:
                best_L = found_L
                best_r = r
    
    # best_r is in [0, 27719], which is <= 1000000
    # Return best_r
    return int(best_r)
